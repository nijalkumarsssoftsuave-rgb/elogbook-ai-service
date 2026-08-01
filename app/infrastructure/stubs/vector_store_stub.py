from app.domain.models import Embedding, RetrievedChunk


class VectorStoreStub:
    """Returns one fixed dense-retrieval chunk, ignoring the query embedding. Stands in
    for a real vector store (Milvus or OpenSearch — still TBD, see CLAUDE.md) searched
    with BGE-M3 embeddings.

    Its chunk_id intentionally does not overlap the BM25 fixture corpus: coupling a dummy
    stub to the real fixture's id scheme would be a hidden dependency to unwind later.
    RRF's dedup behaviour is proven directly in tests/application/test_fusion.py instead.
    """

    async def search(self, embedding: Embedding, top_k: int = 5) -> list[RetrievedChunk]:
        chunks = [
            RetrievedChunk(
                chunk_id="dense-stub-chunk-1",
                document_id="stub-document-1",
                text="This is a stub dense-retrieval chunk standing in for a real vector store.",
                score=0.42,
                metadata={"source_title": "Stub Vector Source", "source": "vector_store_stub"},
            )
        ]
        return chunks[:top_k]
