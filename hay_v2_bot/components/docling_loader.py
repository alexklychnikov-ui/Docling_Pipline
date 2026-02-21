"""
Загрузка и чанкинг документов через Docling (без docling-haystack).
Поддерживает Python 3.10+ и 3.13.
"""

import os
from pathlib import Path

from haystack import Document, component

from hay_v2_bot.config import CHUNKER_TOKENIZER, ROOT_DIR


def _setup_hf_cache():
    """Устанавливает кэш HuggingFace в папку проекта и принудительно отключает симлинки."""
    import platform
    hf_cache_dir = ROOT_DIR / ".hf_cache"
    hf_cache_dir.mkdir(exist_ok=True)
    # Устанавливаем кэш
    os.environ["HF_HUB_CACHE"] = str(hf_cache_dir)
    # Принудительно отключаем симлинки на Windows
    if platform.system() == "Windows":
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
        # Monkey patch: перехватываем os.symlink для копирования вместо симлинков
        try:
            import shutil
            _original_symlink = getattr(os, 'symlink', None)
            if _original_symlink:
                def _copy_instead_of_symlink(src, dst, target_is_directory=False):
                    # Копируем файл/папку вместо создания симлинка
                    try:
                        src_path = Path(src)
                        dst_path = Path(dst)
                        if dst_path.exists():
                            if dst_path.is_dir():
                                shutil.rmtree(dst_path)
                            else:
                                dst_path.unlink()
                        if src_path.is_dir():
                            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                        else:
                            shutil.copy2(src_path, dst_path)
                        return True
                    except Exception:
                        return False
                os.symlink = _copy_instead_of_symlink
        except Exception:
            pass


def _make_chunker():
    _setup_hf_cache()
    from docling.chunking import HybridChunker, HierarchicalChunker
    # Не вызываем HybridChunker() без аргументов — по умолчанию подтягивается sentence-transformers
    try:
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
        from transformers import AutoTokenizer
        tokenizer_obj = AutoTokenizer.from_pretrained(CHUNKER_TOKENIZER)
        tokenizer = HuggingFaceTokenizer(tokenizer=tokenizer_obj)
        return HybridChunker(tokenizer=tokenizer)
    except Exception as e:
        print(f"[WARN] Токенизатор {CHUNKER_TOKENIZER} недоступен: {e}")
        print(f"[INFO] Использую HierarchicalChunker (без токенизатора)")
        return HierarchicalChunker()


def _docling_path_to_documents(path: str, user_id: str, filename: str) -> list[Document]:
    from docling.document_converter import DocumentConverter
    converter = DocumentConverter()
    try:
        result = converter.convert(path)
        doc = result.document
    except Exception as e:
        # Если обработка не удалась из-за памяти или других ошибок
        print(f"[WARN] Ошибка при конвертации {filename}: {e}")
        # Возвращаем пустой список или минимальный документ
        return [
            Document(
                content=f"Документ {filename} не удалось полностью обработать из-за ошибки: {e}",
                meta={"user_id": str(user_id), "filename": filename, "chunk_index": 0, "error": str(e)},
            )
        ]
    
    chunker = _make_chunker()
    try:
        chunks = list(chunker.chunk(dl_doc=doc))
    except Exception as e:
        print(f"[WARN] Ошибка при чанкинге {filename}: {e}")
        # Пробуем получить хотя бы часть документа
        chunks = []
    
    out = []
    for i, ch in enumerate(chunks):
        try:
            text = (chunker.contextualize(chunk=ch) if hasattr(chunker, "contextualize") else ch.text) or ch.text
            if text and text.strip():
                out.append(
                    Document(
                        content=text,
                        meta={"user_id": str(user_id), "filename": filename, "chunk_index": i},
                    )
                )
        except Exception as e:
            print(f"[WARN] Ошибка при обработке чанка {i} из {filename}: {e}")
            continue
    
    if not out:
        # Если ничего не получилось, возвращаем хотя бы сообщение об ошибке
        out.append(
            Document(
                content=f"Документ {filename} обработан, но не удалось извлечь текст. Возможно, документ содержит только изображения или произошла ошибка памяти.",
                meta={"user_id": str(user_id), "filename": filename, "chunk_index": 0},
            )
        )
    
    return out


@component
class DoclingLoader:
    """Конвертирует файлы через Docling в чанки и отдаёт Haystack Document (без docling-haystack)."""

    @component.output_types(documents=list[Document])
    def run(self, paths: list[str], user_id: str, filename: str) -> dict:
        all_docs = []
        for path in paths:
            all_docs.extend(_docling_path_to_documents(path, user_id, filename))
        return {"documents": all_docs}


def docling_path_to_documents(path: str, user_id: str, filename: str) -> list[Document]:
    """Вспомогательная функция: один файл -> список Haystack Document."""
    return _docling_path_to_documents(path, user_id, filename)
