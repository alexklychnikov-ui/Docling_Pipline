import os
from pathlib import Path

import telebot

from hay_v2_bot.config import WORK_LOG_PATH, TELEGRAM_BOT_TOKEN, ROOT_DIR
from hay_v2_bot.components import get_document_store, get_doc_embedder, get_text_embedder
from hay_v2_bot.pipelines import build_ingestion_pipeline, build_agent, get_context_for_user
from hay_v2_bot.bot.handlers import register_handlers
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from datetime import datetime


def _log_work(msg: str):
    line = f"[{datetime.now().isoformat()}] {msg}"
    print(line)
    with open(WORK_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_bot():
    import platform
    # Настраиваем HuggingFace кэш в папку проекта (решает проблему с правами на Windows)
    hf_cache_dir = ROOT_DIR / ".hf_cache"
    hf_cache_dir.mkdir(exist_ok=True)
    # Принудительно отключаем симлинки и используем копирование файлов
    os.environ["HF_HUB_CACHE"] = str(hf_cache_dir)
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
    if platform.system() == "Windows":
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    
    api_key = os.getenv("PINECONE_API_KEY") or os.getenv("PYNECONE_API_KEY")
    if not api_key:
        _log_work("ERROR: PINECONE_API_KEY (или PYNECONE_API_KEY) не задан")
        raise SystemExit("Задай PINECONE_API_KEY в .env")
    os.environ["PINECONE_API_KEY"] = api_key

    _log_work("Start: инициализация Pinecone, embedders, pipelines, agent")
    document_store = get_document_store()
    doc_embedder = get_doc_embedder()
    text_embedder = get_text_embedder()
    retriever = PineconeEmbeddingRetriever(document_store=document_store, top_k=15)

    ingestion_pipeline = build_ingestion_pipeline(document_store, doc_embedder)

    agent = build_agent()
    agent.warm_up()
    _log_work("Agent и Pinecone готовы")

    if not TELEGRAM_BOT_TOKEN:
        _log_work("ERROR: TELEGRAM_BOT_TOKEN не задан")
        raise SystemExit("Задай TELEGRAM_BOT_TOKEN в .env")
    
    # Отключаем прокси для запросов к Telegram (иначе ProxyError при sendMessage)
    # OpenAI идёт через PROXY_BASE_URL в коде, системный прокси здесь не нужен
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(key, None)
    
    bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
    register_handlers(
        bot=bot,
        document_store=document_store,
        text_embedder=text_embedder,
        doc_embedder=doc_embedder,
        retriever=retriever,
        agent=agent,
        ingestion_pipeline=ingestion_pipeline,
        logger=_log_work,
    )
    _log_work("Polling started")
    bot.infinity_polling()
