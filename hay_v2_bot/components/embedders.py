from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.utils import Secret

from hay_v2_bot.config import EMBEDDING_DIM, EMBEDDING_MODEL, PROXY_BASE_URL, OPENAI_API_KEY


def get_doc_embedder():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY или PROXY_API_KEY должен быть задан в .env")
    return OpenAIDocumentEmbedder(
        api_key=Secret.from_token(OPENAI_API_KEY),
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
        api_base_url=PROXY_BASE_URL,
    )


def get_text_embedder():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY или PROXY_API_KEY должен быть задан в .env")
    return OpenAITextEmbedder(
        api_key=Secret.from_token(OPENAI_API_KEY),
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
        api_base_url=PROXY_BASE_URL,
    )
