"""
Telegram-бот: персональный помощник на Haystack Agent + Pinecone (контекст по косинусному сходству).
Инструменты: случайный факт о собаках (Dog API), случайная картинка собаки + описание породы через OpenAI Vision.
"""
import os
import time
import urllib.request
import json
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import telebot
from haystack import Document, component
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools.component_tool import ComponentTool
from haystack.utils import Secret

from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever

WORK_LOG_PATH = os.path.join(os.path.dirname(__file__), "WORK_LOG.txt")

def log_work(msg: str):
    with open(WORK_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536

def _pinecone_api_key():
    return os.getenv("PINECONE_API_KEY") or os.getenv("PYNECONE_API_KEY")

def get_document_store():
    return PineconeDocumentStore(
        index=os.getenv("PINECONE_INDEX_NAME", "tgdialog"),
        metric="cosine",
        dimension=EMBEDDING_DIM,
        spec={"serverless": {"region": "us-east-1", "cloud": "aws"}},
    )


@component
class DogFactTool:
    """Возвращает случайный факт о собаках (Dog API by kinduff)."""

    @component.output_types(result=str)
    def run(self) -> dict:
        try:
            with urllib.request.urlopen("https://dogapi.dog/api/v2/facts?limit=1", timeout=10) as r:
                data = json.loads(r.read().decode())
            facts = data.get("data", [])
            if facts and "attributes" in facts[0]:
                body = facts[0]["attributes"].get("body", "No fact available.")
                return {"result": body}
            return {"result": "Не удалось получить факт."}
        except Exception as e:
            return {"result": f"Ошибка API: {e}"}


@component
class DogImageDescribeTool:
    """Получает случайную картинку собаки (dog.ceo), отправляет в OpenAI Vision и возвращает описание породы и предысторию."""

    @component.output_types(result=str)
    def run(self) -> dict:
        try:
            with urllib.request.urlopen("https://dog.ceo/api/breeds/image/random", timeout=10) as r:
                data = json.loads(r.read().decode())
            image_url = data.get("message")
            if not image_url:
                return {"result": "Не удалось получить ссылку на изображение."}
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Опиши собаку на фото: укажи породу (или предположение), краткую предысторию породы и как она получилась. Ответь на русском, кратко и по делу."},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                max_tokens=500,
            )
            text = (resp.choices[0].message.content or "").strip()
            return {"result": f"Фото: {image_url}\n\n{text}"}
        except Exception as e:
            return {"result": f"Ошибка: {e}"}


dog_fact_tool = ComponentTool(
    name="dog_fact",
    description="Получить случайный интересный факт о собаках. Вызывай, когда пользователь просит факт о собаках или хочет что-то интересное про собак.",
    component=DogFactTool(),
    outputs_to_string={"source": "result"},
)

dog_image_tool = ComponentTool(
    name="dog_image_describe",
    description="Получить случайную фотографию собаки и описание породы с краткой историей. Вызывай, когда пользователь просит картинку собаки или описание породы по фото.",
    component=DogImageDescribeTool(),
    outputs_to_string={"source": "result"},
)


def build_agent():
    generator = OpenAIChatGenerator(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
    )
    return Agent(
        chat_generator=generator,
        tools=[dog_fact_tool, dog_image_tool],
        system_prompt="""Ты — умный персональный помощник в Telegram. Ты ведёшь диалог естественно и учитываешь контекст предыдущих сообщений пользователя.
Используй инструменты dog_fact и dog_image_describe по запросу пользователя (факты о собаках, картинки собак, описание пород).
Отвечай кратко, по-русски, дружелюбно. Если тебе передали контекст предыдущего диалога — опирайся на него и продолжай разговор как настоящий помощник.""",
        exit_conditions=["text"],
        max_agent_steps=10,
    )


def get_context_for_user(document_store, text_embedder, retriever, user_id: str, query_embedding: list, top_k: int = 15, logger=None):
    log = logger or log_work
    if query_embedding is None:
        log(f"[retrieve] user_id={user_id} query_embedding=None, skip")
        return ""
    filters = {"field": "user_id", "operator": "==", "value": str(user_id)}
    docs = retriever.run(query_embedding=query_embedding, filters=filters, top_k=top_k)
    documents = docs.get("documents") or []
    log(f"[retrieve] user_id={user_id} found {len(documents)} docs (top_k={top_k})")
    if not documents:
        return ""
    sorted_docs = sorted(documents, key=lambda d: d.meta.get("timestamp", 0))
    ts_min = sorted_docs[0].meta.get("timestamp")
    ts_max = sorted_docs[-1].meta.get("timestamp")
    context_len = sum(len(d.content) for d in sorted_docs)
    log(f"[retrieve] user_id={user_id} context: {len(sorted_docs)} docs, {context_len} chars, timestamp range [{ts_min}, {ts_max}]")
    return "\n".join(d.content for d in sorted_docs)


def main():
    log_work("Start: загрузка .env и инициализация Pinecone + embedders")
    api_key = _pinecone_api_key()
    if not api_key:
        log_work("ERROR: PINECONE_API_KEY (или PYNECONE_API_KEY) не задан")
        raise SystemExit("Задай PINECONE_API_KEY в .env")
    os.environ["PINECONE_API_KEY"] = api_key

    document_store = get_document_store()
    doc_embedder = OpenAIDocumentEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
    )
    text_embedder = OpenAITextEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
    )
    retriever = PineconeEmbeddingRetriever(document_store=document_store, top_k=15)

    agent = build_agent()
    agent.warm_up()
    log_work("Agent и Pinecone готовы")

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        log_work("ERROR: TELEGRAM_BOT_TOKEN не задан")
        raise SystemExit("Задай TELEGRAM_BOT_TOKEN в .env")
    bot = telebot.TeleBot(token)

    @bot.message_handler(commands=["start"])
    def cmd_start(message):
        bot.reply_to(message, "Привет! Я твой персональный помощник. Могу рассказать факты о собаках, показать случайную собаку с описанием породы и просто общаться с учётом контекста. Напиши что-нибудь.")

    @bot.message_handler(func=lambda m: True)
    def on_message(message):
        user_id = message.from_user.id
        chat_id = message.chat.id
        text = (message.text or "").strip()
        if not text:
            return
        t0 = time.perf_counter()
        log_work(f"[run] user_id={user_id} chat_id={chat_id} query_len={len(text)} query={text[:80]!r}...")
        try:
            embedded = text_embedder.run(text=text)
            query_emb = embedded.get("embedding")
            if query_emb is not None and isinstance(query_emb, list) and len(query_emb) > 0:
                vec = query_emb[0] if isinstance(query_emb[0], list) else query_emb
            else:
                vec = None
            t1 = time.perf_counter()
            log_work(f"[run] user_id={user_id} embed done in {t1 - t0:.2f}s")
            context_str = get_context_for_user(document_store, text_embedder, retriever, str(user_id), vec) if vec else ""
            if context_str:
                user_content = f"Контекст предыдущего диалога:\n{context_str}\n\nТекущее сообщение пользователя: {text}"
            else:
                user_content = text
            messages = [ChatMessage.from_user(user_content)]
            result = agent.run(messages=messages)
            t2 = time.perf_counter()
            replies = result.get("messages") or []
            reply_text = replies[-1].text if replies else "Не удалось сформировать ответ."
            num_agent_messages = len(replies)
            log_work(f"[run] user_id={user_id} agent done in {t2 - t1:.2f}s messages={num_agent_messages} reply_len={len(reply_text)}")
            bot.send_message(chat_id, reply_text)

            ts = time.time()
            to_store = [
                Document(content=f"user: {text}", meta={"user_id": str(user_id), "timestamp": ts}),
                Document(content=f"assistant: {reply_text}", meta={"user_id": str(user_id), "timestamp": ts + 0.01}),
            ]
            out = doc_embedder.run(documents=to_store)
            docs_with_emb = out.get("documents") or to_store
            document_store.write_documents(docs_with_emb)
            t3 = time.perf_counter()
            log_work(f"[run] user_id={user_id} stored {len(docs_with_emb)} docs in {t3 - t2:.2f}s total_run={t3 - t0:.2f}s")
        except Exception as e:
            log_work(f"[run] user_id={user_id} Error: {e}")
            bot.send_message(chat_id, f"Произошла ошибка: {e}")

    log_work("Polling started")
    bot.infinity_polling()


if __name__ == "__main__":
    main()
