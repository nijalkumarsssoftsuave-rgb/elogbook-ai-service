from app.application.retrieval_service import RetrievalService
from app.domain.models import (
    Embedding,
    Question,
    RetrievalCandidates,
    RetrievedChunk,
    SourceSearchRequest,
)
from app.domain.permission import SearchScope, Source

# The service is handed an already-resolved SearchScope; resolution itself is covered in
# tests/application/test_permission_resolver.py, and the per-method retrievers live
# behind the multi-source retriever, covered in its own test module.

ALL_SOURCES = [
    Source(source_id="shift-logs", display_name="Shift Logs"),
    Source(source_id="incidents", display_name="Incident Reports"),
]
FULL_SCOPE = SearchScope(sources=ALL_SOURCES)


class FakeEmbedding:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    async def embed(self, text: str) -> Embedding:
        self.calls.append("embed")
        return Embedding(vector=[0.1, 0.2], model="fake")


class FakeMultiSourceRetriever:
    def __init__(
        self,
        calls: list[str],
        dense: list[RetrievedChunk] | None = None,
        sparse: list[RetrievedChunk] | None = None,
    ) -> None:
        self.calls = calls
        self._dense = dense
        self._sparse = sparse
        self.received: SourceSearchRequest | None = None

    async def search(self, request: SourceSearchRequest) -> RetrievalCandidates:
        self.calls.append("multi_source_search")
        self.received = request
        return RetrievalCandidates(
            searched_source_ids=request.search_scope.source_ids,
            dense=self._dense
            if self._dense is not None
            else [RetrievedChunk(chunk_id="shared-1", document_id="d1", text="dense", score=0.9)],
            sparse=self._sparse
            if self._sparse is not None
            else [RetrievedChunk(chunk_id="shared-1", document_id="d1", text="sparse", score=5.0)],
        )


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


def _question(language: str = "en") -> Question:
    return Question(text="what happened on the last shift?", user_id="u1", language=language)


def _build(
    calls: list[str],
    dense: list[RetrievedChunk] | None = None,
    sparse: list[RetrievedChunk] | None = None,
    reranker: FakeReranker | None = None,
    retriever: FakeMultiSourceRetriever | None = None,
) -> RetrievalService:
    return RetrievalService(
        FakeEmbedding(calls),
        retriever or FakeMultiSourceRetriever(calls, dense, sparse),
        reranker or FakeReranker(calls),
    )


async def test_retrieve_embeds_then_searches_then_reranks() -> None:
    calls: list[str] = []

    await _build(calls).retrieve(_question(), FULL_SCOPE)

    assert calls == ["embed", "multi_source_search", "rerank"]


async def test_the_search_scope_is_handed_to_the_retriever_unchanged() -> None:
    """The entitlement reaches the retriever intact.

    Since ES-338 it arrives combined with the caller's filters rather than raw, so the
    assertion is on the sources and the permitted values rather than on object identity --
    but the point is unchanged: this service does not narrow, widen or reinterpret what the
    permission resolver decided.
    """
    calls: list[str] = []
    retriever = FakeMultiSourceRetriever(calls)
    scope = SearchScope(sources=[ALL_SOURCES[0]], department_ids=["maintenance"])

    await _build(calls, retriever=retriever).retrieve(_question(), scope)

    assert retriever.received is not None
    handed = retriever.received.search_scope
    assert handed.sources == scope.sources
    assert handed.area_ids == scope.area_ids
    assert handed.department_ids == scope.department_ids
    assert handed.company_ids == scope.company_ids
    assert handed.is_satisfiable is True


async def test_an_empty_scope_retrieves_nothing_without_searching() -> None:
    """Fails closed, and cheaply: no embedding call, no search, no evidence. This guard
    lives here rather than at the caller because, now that the scope arrives from outside,
    this is the last place that can refuse.
    """
    calls: list[str] = []

    results = await _build(calls).retrieve(_question(), SearchScope())

    assert results == []
    assert calls == []


async def test_a_scope_with_filters_but_no_sources_is_still_empty() -> None:
    """Filters narrow a search set; they cannot conjure one. Treating this as searchable
    would mean an unentitled caller reaching an unrestricted index.
    """
    calls: list[str] = []

    results = await _build(calls).retrieve(
        _question(), SearchScope(sources=[], department_ids=["maintenance"])
    )

    assert results == []
    assert calls == []


async def test_the_query_is_embedded_once_regardless_of_source_count() -> None:
    calls: list[str] = []

    await _build(calls).retrieve(_question(), FULL_SCOPE)

    assert calls.count("embed") == 1


async def test_the_question_language_reaches_the_search_request() -> None:
    """The keyword index is mixed-language and matches across languages on shared tokens,
    so the question's language still has to travel down.
    """
    calls: list[str] = []
    retriever = FakeMultiSourceRetriever(calls)
    service = _build(calls, retriever=retriever)

    await service.retrieve(_question("ar"), FULL_SCOPE)

    assert retriever.received is not None
    assert retriever.received.language == "ar"


# --- regression: the fusion tail is untouched ------------------------------------------


async def test_retrieve_deduplicates_a_chunk_returned_by_both_methods() -> None:
    calls: list[str] = []

    results = await _build(calls).retrieve(_question(), FULL_SCOPE)

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

    results = await _build(calls, dense=dense, sparse=sparse, reranker=reranker).retrieve(
        _question(), FULL_SCOPE, top_k=2
    )

    assert len(results) == 2
    # The reranker still saw the full fused candidate pool before truncation.
    assert len(reranker.received_chunks) == 8


async def test_retrieve_over_fetches_candidates_beyond_the_requested_top_k() -> None:
    calls: list[str] = []
    retriever = FakeMultiSourceRetriever(calls)
    service = _build(calls, retriever=retriever)

    await service.retrieve(_question(), FULL_SCOPE, top_k=3)

    assert retriever.received is not None
    assert retriever.received.limit_per_source == RetrievalService._CANDIDATE_POOL_SIZE


async def test_exactly_two_ranked_lists_reach_fusion() -> None:
    """Phase 5 stays untouched only if the number of lists handed to RRF does not change.
    Fusing per-source lists instead would compress ranks and move every retrieval metric.
    """
    calls: list[str] = []
    dense = [RetrievedChunk(chunk_id="d1", document_id="doc", text="t", score=0.9)]
    sparse = [RetrievedChunk(chunk_id="s1", document_id="doc", text="t", score=5.0)]
    reranker = FakeReranker(calls)

    await _build(calls, dense=dense, sparse=sparse, reranker=reranker).retrieve(_question(), FULL_SCOPE)

    # Both survive fusion with a rank-1 RRF contribution each, which is only true if they
    # arrived as two separate lists.
    assert {chunk.chunk_id for chunk in reranker.received_chunks} == {"d1", "s1"}
    assert all(chunk.score == 1.0 / 61 for chunk in reranker.received_chunks)
