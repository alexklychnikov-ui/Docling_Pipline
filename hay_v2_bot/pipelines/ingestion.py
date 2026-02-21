"""
Пайплайн индексации: Docling (без docling-haystack) -> эмбеддинг -> Pinecone.
"""

import os
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker, HierarchicalChunker
from haystack import Pipeline
from haystack.components.writers import DocumentWriter

from hay_v2_bot.config import CHUNKER_TOKENIZER, ROOT_DIR
from hay_v2_bot.components import get_doc_embedder, DoclingLoader


def _setup_hf_cache():
    """Устанавливает кэш HuggingFace в папку проекта."""
    import platform
    hf_cache_dir = ROOT_DIR / ".hf_cache"
    hf_cache_dir.mkdir(exist_ok=True)
    if "HF_HUB_CACHE" not in os.environ:
        os.environ["HF_HUB_CACHE"] = str(hf_cache_dir)
    # Принудительно отключаем симлинки на Windows
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
    if platform.system() == "Windows":
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"


def _make_chunker():
    _setup_hf_cache()
    try:
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
        from transformers import AutoTokenizer
        tokenizer = HuggingFaceTokenizer(tokenizer=AutoTokenizer.from_pretrained(CHUNKER_TOKENIZER))
        return HybridChunker(tokenizer=tokenizer)
    except Exception:
        return HierarchicalChunker()


def build_ingestion_pipeline(document_store, doc_embedder=None):
    if doc_embedder is None:
        doc_embedder = get_doc_embedder()
    loader = DoclingLoader()
    writer = DocumentWriter(document_store=document_store)

    pipe = Pipeline()
    pipe.add_component("loader", loader)
    pipe.add_component("embedder", doc_embedder)
    pipe.add_component("writer", writer)
    pipe.connect("loader.documents", "embedder.documents")
    pipe.connect("embedder.documents", "writer.documents")
    return pipe


def get_document_texts_for_summary(file_path: str, max_chars: int = 12000) -> list[str]:
    """Читает файл через Docling, возвращает тексты чанков для резюме."""
    converter = DocumentConverter()
    result = converter.convert(file_path)
    doc = result.document
    chunker = _make_chunker()
    chunks = list(chunker.chunk(dl_doc=doc))
    texts = []
    total = 0
    for ch in chunks:
        c = (ch.text or "").strip()
        if not c:
            continue
        if total + len(c) > max_chars:
            texts.append(c[: max_chars - total])
            break
        texts.append(c)
        total += len(c)
    return texts
