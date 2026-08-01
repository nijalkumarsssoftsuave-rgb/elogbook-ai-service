from app.domain.models import Question, RetrievedChunk


class RerankerStub:
    """Identity pass-through: returns the chunks unchanged, in the order fusion produced.
    Stands in for BGE-Reranker-v2-M3 cross-encoder reranking, which needs a model
    checkpoint that does not exist yet. Unchanged ordering means "not reranked", not
    "reranked and already optimal".
    """

    async def rerank(
        self, question: Question, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        return list(chunks)
