from app.domain.models import Embedding, RetrievedChunk
from app.domain.retrieval import EffectiveSearchScope
from app.infrastructure.retrieval.fixture_corpus import SHIFT_LOGS


class VectorStoreStub:
    """Returns one fixed dense-retrieval chunk, ignoring the query embedding. Stands in
    for a real vector store (Milvus or OpenSearch — still TBD, see CLAUDE.md) searched
    with BGE-M3 embeddings.

    Its chunk_id intentionally does not overlap the BM25 fixture corpus: coupling a dummy
    stub to the real fixture's id scheme would be a hidden dependency to unwind later.
    RRF's dedup behaviour is proven directly in tests/application/test_fusion.py instead.
    """

    # The stub chunk has to belong somewhere for source restriction to mean anything. It
    # sits in the general shift-logs source, the one every role can read.
    _SOURCE_ID = SHIFT_LOGS

    async def search(
        self,
        embedding: Embedding,
        top_k: int = 5,
        source_id: str | None = None,
        scope: EffectiveSearchScope | None = None,
    ) -> list[RetrievedChunk]:
        if source_id is not None and source_id != self._SOURCE_ID:
            return []
        chunks = [
            RetrievedChunk(
                chunk_id="dense-stub-chunk-1",
                document_id="stub-document-1",
                text="This is a stub dense-retrieval chunk standing in for a real vector store.",
                score=0.42,
                metadata={
                    "source_title": "Stub Vector Source",
                    "source": "vector_store_stub",
                    "source_id": self._SOURCE_ID,
                },
            )
        ]
        # The stub chunk declares no organisational attributes, so any filter on them
        # excludes it -- which is fail-closed matching working as intended, not a gap.
        # A real vector store would push this into the query as a metadata clause.
        if scope is not None:
            chunks = [chunk for chunk in chunks if scope.permits(chunk.metadata)]
        return chunks[:top_k]
