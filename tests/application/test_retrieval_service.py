from app.application.retrieval_service import RetrievalService
from app.domain.models import Embedding, Question, RetrievedChunk


class FakeEmbedding:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    async def embed(self, text: str) -> Embedding:
        self.calls.append("embed")
        return Embedding(vector=[0.1, 0.2], model="fake")


class FakeVectorStore:
    def __init__(self, calls: list[str], chunks: list[RetrievedChunk] | None = None) -> None:
        self.calls = calls
        self._chunks = chunks
        self.received_top_k: int | None = None

    async def search(self, embedding: Embedding, top_k: int = 5) -> list[RetrievedChunk]:
        self.calls.append("dense_search")
        self.received_top_k = top_k
        if self._chunks is not None:
            return self._chunks
        return [RetrievedChunk(chunk_id="shared-1", document_id="d1", text="dense", score=0.9)]


class FakeKeywordRetriever:
    def __init__(self, calls: list[str], chunks: list[RetrievedChunk] | None = None) -> None:
        self.calls = calls
        self._chunks = chunks

    async def search(self, query_text: str, top_k: int = 5) -> list[RetrievedChunk]:
        self.calls.append("sparse_search")
        if self._chunks is not None:
            return self._chunks
        return [RetrievedChunk(chunk_id="shared-1", document_id="d1", text="sparse", score=5.0)]


class FakeReranker:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.received_chunks: list[RetrievedChunk] = []

    async def rerank(
        self, question: Question, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        self.calls.append("rerank")
        self.received_chunks = chunks
        return chunks


def _question() -> Question:
    return Question(text="what happened on the last shift?", user_id="u1", language="en")


async def test_retrieve_runs_embed_then_both_searches_then_rerank() -> None:
    calls: list[str] = []
    reranker = FakeReranker(calls)
    service = RetrievalService(
        FakeEmbedding(calls), FakeVectorStore(calls), FakeKeywordRetriever(calls), reranker
    )

    await service.retrieve(_question(), top_k=5)

    assert calls[0] == "embed"
    # gather does not guarantee ordering between the two searches, only that both ran.
    assert set(calls[1:3]) == {"dense_search", "sparse_search"}
    assert calls[3] == "rerank"


async def test_retrieve_deduplicates_a_chunk_returned_by_both_methods() -> None:
    calls: list[str] = []
    service = RetrievalService(
        FakeEmbedding(calls),
        FakeVectorStore(calls),
        FakeKeywordRetriever(calls),
        FakeReranker(calls),
    )

    results = await service.retrieve(_question(), top_k=5)

    assert len(results) == 1
    assert results[0].chunk_id == "shared-1"


async def test_retrieve_truncates_to_top_k_after_reranking() -> None:
    calls: list[str] = []
    dense = [
        RetrievedChunk(chunk_id=f"d{i}", document_id="doc", text="t", score=0.5) for i in range(4)
    ]
    sparse = [
        RetrievedChunk(chunk_id=f"s{i}", document_id="doc", text="t", score=0.5) for i in range(4)
    ]
    reranker = FakeReranker(calls)
    service = RetrievalService(
        FakeEmbedding(calls),
        FakeVectorStore(calls, chunks=dense),
        FakeKeywordRetriever(calls, chunks=sparse),
        reranker,
    )

    results = await service.retrieve(_question(), top_k=2)

    assert len(results) == 2
    # The reranker still saw the full fused candidate pool before truncation.
    assert len(reranker.received_chunks) == 8


async def test_retrieve_over_fetches_candidates_beyond_the_requested_top_k() -> None:
    calls: list[str] = []
    vector_store = FakeVectorStore(calls)
    service = RetrievalService(
        FakeEmbedding(calls), vector_store, FakeKeywordRetriever(calls), FakeReranker(calls)
    )

    await service.retrieve(_question(), top_k=3)

    assert vector_store.received_top_k == RetrievalService._CANDIDATE_POOL_SIZE
