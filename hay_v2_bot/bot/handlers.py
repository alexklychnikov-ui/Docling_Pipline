import os
import time
import tempfile
from pathlib import Path

import telebot
from haystack.dataclasses import ChatMessage

from hay_v2_bot.pipelines import get_context_for_user, build_file_summary, get_document_texts_for_summary


def register_handlers(
    bot: telebot.TeleBot,
    document_store,
    text_embedder,
    doc_embedder,
    retriever,
    agent,
    ingestion_pipeline,
    logger=None,
):
    log = logger or (lambda msg: None)

    @bot.message_handler(commands=["start"])
    def cmd_start(message):
        bot.reply_to(
            message,
            "Привет! Я помощник с доступом к твоим документам: загружай PDF или DOCX — я сохраню контент и смогу отвечать по ним. Также могу рассказать факт о собаках или показать случайную собаку с описанием породы. Напиши что-нибудь или пришли файл.",
        )

    @bot.message_handler(content_types=["document"])
    def on_document(message):
        user_id = message.from_user.id
        chat_id = message.chat.id
        doc = message.document
        file_id = doc.file_id
        filename = doc.file_name or "document"
        log(f"[file] user_id={user_id} filename={filename} file_id={file_id}")
        bot.send_message(
            chat_id,
            "Файл получен. Запускаю анализ и сохранение. Это может занять немного времени…",
        )
        try:
            tg_file = bot.get_file(file_id)
            data = bot.download_file(tg_file.file_path)
            suffix = Path(filename).suffix or ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="hayv2_") as f:
                f.write(data)
                tmp_path = f.name
            try:
                ingestion_pipeline.run(
                    {"loader": {"paths": [tmp_path], "user_id": str(user_id), "filename": filename}}
                )
                texts = get_document_texts_for_summary(tmp_path)
                summary = build_file_summary(texts)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            bot.send_message(
                chat_id,
                "Готово. Я изучил этот файл, теперь можем его обсудить.",
            )
            bot.send_message(chat_id, summary)
            log(f"[file] user_id={user_id} filename={filename} done, summary_len={len(summary)}")
        except Exception as e:
            log(f"[file] user_id={user_id} error: {e}")
            # Обрезаем сообщение об ошибке для Telegram (лимит 4096 символов)
            error_msg = str(e)
            if len(error_msg) > 4000:
                error_msg = error_msg[:4000] + "\n... (сообщение обрезано)"
            bot.send_message(chat_id, f"Ошибка при обработке файла: {error_msg}")

    @bot.message_handler(func=lambda m: True)
    def on_message(message):
        user_id = message.from_user.id
        chat_id = message.chat.id
        text = (message.text or "").strip()
        if not text:
            return
        t0 = time.perf_counter()
        log(f"[run] user_id={user_id} chat_id={chat_id} query_len={len(text)} query={text[:80]!r}...")
        try:
            embedded = text_embedder.run(text=text)
            query_emb = embedded.get("embedding")
            if query_emb is not None and isinstance(query_emb, list) and len(query_emb) > 0:
                vec = query_emb[0] if isinstance(query_emb[0], list) else query_emb
            else:
                vec = None
            t1 = time.perf_counter()
            log(f"[run] user_id={user_id} embed done in {t1 - t0:.2f}s")
            context_str = get_context_for_user(retriever, str(user_id), vec, top_k=15, logger=log) if vec else ""
            if context_str:
                user_content = f"Контекст предыдущего диалога и загруженных документов:\n{context_str}\n\nТекущее сообщение пользователя: {text}"
            else:
                user_content = text
            messages = [ChatMessage.from_user(user_content)]
            result = agent.run(messages=messages)
            t2 = time.perf_counter()
            replies = result.get("messages") or []
            reply_text = replies[-1].text if replies else "Не удалось сформировать ответ."
            log(f"[run] user_id={user_id} agent done in {t2 - t1:.2f}s messages={len(replies)} reply_len={len(reply_text)}")
            bot.send_message(chat_id, reply_text)

            ts = time.time()
            from haystack import Document
            to_store = [
                Document(content=f"user: {text}", meta={"user_id": str(user_id), "timestamp": ts}),
                Document(content=f"assistant: {reply_text}", meta={"user_id": str(user_id), "timestamp": ts + 0.01}),
            ]
            out = doc_embedder.run(documents=to_store)
            docs_with_emb = out.get("documents") or to_store
            document_store.write_documents(docs_with_emb)
            t3 = time.perf_counter()
            log(f"[run] user_id={user_id} stored {len(docs_with_emb)} docs in {t3 - t2:.2f}s total_run={t3 - t0:.2f}s")
        except Exception as e:
            log(f"[run] user_id={user_id} Error: {e}")
            bot.send_message(chat_id, f"Произошла ошибка: {e}")

