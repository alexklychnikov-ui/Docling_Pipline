from haystack import Document, component


@component
class DocumentMetaAdder:
    """Добавляет user_id, filename и chunk_index в meta каждого документа."""

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document], user_id: str, filename: str) -> dict:
        if not documents:
            return {"documents": []}
        out = []
        for i, doc in enumerate(documents):
            meta = dict(doc.meta) if doc.meta else {}
            meta["user_id"] = str(user_id)
            meta["filename"] = filename
            meta["chunk_index"] = i
            out.append(Document(content=doc.content, meta=meta))
        return {"documents": out}
