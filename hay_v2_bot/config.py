import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv()

# Embedding
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536

# Docling chunker tokenizer (модель из transformers: bert, gpt2 и т.д. Не sentence-transformers!)
CHUNKER_TOKENIZER = os.getenv("CHUNKER_TOKENIZER", "bert-base-uncased")

# Pinecone
def _pinecone_api_key():
    return os.getenv("PINECONE_API_KEY") or os.getenv("PYNECONE_API_KEY")

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "tgdialog")

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# OpenAI (все запросы через прокси proxyapi.ru)
# Приоритет: PROXY_API_KEY для прокси, затем OPENAI_API_KEY
OPENAI_API_KEY = os.getenv("PROXY_API_KEY") or os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PROXY_BASE_URL = os.getenv("PROXY_BASE_URL", "https://openai.api.proxyapi.ru/v1")

# Paths
ROOT_DIR = Path(__file__).resolve().parent
WORK_LOG_PATH = ROOT_DIR / "WORK_LOG.txt"
