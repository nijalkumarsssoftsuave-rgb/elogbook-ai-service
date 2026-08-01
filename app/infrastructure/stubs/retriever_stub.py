from app.domain.models import Question, RetrievedChunk


class RetrieverStub:
    """Returns fixed dummy chunks. Stands in for hybrid (BGE-M3 + BM25 + RRF) retrieval."""

    async def retrieve(self, question: Question, top_k: int = 5) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id="stub-chunk-1",
                document_id="stub-document-1",
                text="This is a stub retrieved chunk standing in for real hybrid retrieval.",
                score=0.42,
                metadata={"source": "stub"},
            )
        ][:top_k]
