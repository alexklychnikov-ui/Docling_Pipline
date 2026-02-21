import os

from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from hay_v2_bot.config import EMBEDDING_DIM, PINECONE_INDEX_NAME


def _pinecone_api_key():
    return os.getenv("PINECONE_API_KEY") or os.getenv("PYNECONE_API_KEY")


def get_document_store():
    api_key = _pinecone_api_key()
    if api_key:
        os.environ["PINECONE_API_KEY"] = api_key
    return PineconeDocumentStore(
        index=PINECONE_INDEX_NAME,
        metric="cosine",
        dimension=EMBEDDING_DIM,
        spec={"serverless": {"region": "us-east-1", "cloud": "aws"}},
    )
