from app.domain.models import Embedding, RetrievedChunk, Source, SourceSearchRequest
from app.infrastructure.retrieval.bm25_keyword_retriever import BM25KeywordRetriever
from app.infrastructure.retrieval.fixture_corpus import INCIDENTS, SAFETY, SHIFT_LOGS
from app.infrastructure.retrieval.multi_source_retriever import MultiSourceRetriever
from app.infrastructure.stubs.embedding_stub import EmbeddingStub
from app.infrastructure.stubs.vector_store_stub import VectorStoreStub

ALL_SOURCES = [
    Source(source_id=SHIFT_LOGS, display_name="Shift Logs"),
    Source(source_id=INCIDENTS, display_name="Incident Reports"),
    Source(source_id=SAFETY, display_name="Safety and Visitor Records"),
]
SHIFT_LOGS_ONLY = [ALL_SOURCES[0]]

# Matches log-006 (incidents) strongly and log-003 / log-004 (shift-logs / safety) weakly,
# so it reaches across all three sources -- which is what makes the merge observable.
QUERY = "fire alarm during the night shift maintenance safety"


class RecordingKeywordRetriever:
    """Wraps the real BM25 retriever and records which source each call asked for."""

    def __init__(self) -> None:
        self._inner = BM25KeywordRetriever()
        self.requested_source_ids: list[str | None] = []

    async def search(
        self,
        query_text: str,
        top_k: int = 5,
        language: str | None = None,
        source_id: str | None = None,
    ) -> list[RetrievedChunk]:
        self.requested_source_ids.append(source_id)
        return await self._inner.search(query_text, top_k, language, source_id)


async def _request(sources: list[Source], query: str = QUERY) -> SourceSearchRequest:
    return SourceSearchRequest(
        query_text=query,
        query_embedding=await EmbeddingStub().embed(query),
        language="en",
        sources=sources,
        limit_per_source=20,
    )


def _build(keyword: RecordingKeywordRetriever | None = None) -> MultiSourceRetriever:
    return MultiSourceRetriever(VectorStoreStub(), keyword or RecordingKeywordRetriever())


# --- multiple sources queried -----------------------------------------------------------


async def test_every_permitted_source_is_queried() -> None:
    keyword = RecordingKeywordRetriever()

    candidates = await _build(keyword).search(await _request(ALL_SOURCES))

    assert sorted(keyword.requested_source_ids) == sorted([SHIFT_LOGS, INCIDENTS, SAFETY])
    assert candidates.searched_source_ids == [SHIFT_LOGS, INCIDENTS, SAFETY]


async def test_a_permitted_source_that_matches_nothing_still_counts_as_searched() -> None:
    """"Where did we look?" and "where did we find something?" are different questions.
    Only reporting the second would make an empty source indistinguishable from one the
    caller was never allowed to see.
    """
    candidates = await _build().search(await _request(ALL_SOURCES, query="zzzz qqqq nothing"))

    assert candidates.searched_source_ids == [SHIFT_LOGS, INCIDENTS, SAFETY]
    assert candidates.sparse == []


# --- unauthorized sources excluded --------------------------------------------------------


async def test_sources_outside_the_request_are_never_searched() -> None:
    keyword = RecordingKeywordRetriever()

    candidates = await _build(keyword).search(await _request(SHIFT_LOGS_ONLY))

    assert keyword.requested_source_ids == [SHIFT_LOGS]
    assert candidates.searched_source_ids == [SHIFT_LOGS]


async def test_no_candidate_comes_from_an_unpermitted_source() -> None:
    """The strong form: not merely "we asked for the right source", but "nothing from a
    forbidden one is in the result", which is the property that actually protects data.
    """
    candidates = await _build().search(await _request(SHIFT_LOGS_ONLY))

    assert candidates.sparse  # the query does match shift-logs documents
    for chunk in [*candidates.dense, *candidates.sparse]:
        assert chunk.metadata["source_id"] == SHIFT_LOGS


async def test_the_incident_report_is_reachable_with_access_and_not_without() -> None:
    """Same query, two scopes -- the difference is the authorization, nothing else."""
    with_access = await _build().search(await _request(ALL_SOURCES))
    without_access = await _build().search(await _request(SHIFT_LOGS_ONLY))

    assert "log-006" in [chunk.chunk_id for chunk in with_access.sparse]
    assert "log-006" not in [chunk.chunk_id for chunk in without_access.sparse]


# --- empty permission scope ----------------------------------------------------------------


async def test_no_sources_means_nothing_is_searched_at_all() -> None:
    """Fails closed. An empty source list must not fall through to an unrestricted search,
    which is the failure mode this whole ticket exists to prevent.
    """
    keyword = RecordingKeywordRetriever()

    candidates = await _build(keyword).search(await _request([]))

    assert keyword.requested_source_ids == []
    assert candidates.searched_source_ids == []
    assert candidates.dense == []
    assert candidates.sparse == []


# --- single source ---------------------------------------------------------------------------


async def test_a_single_source_returns_its_own_documents_in_score_order() -> None:
    candidates = await _build().search(await _request(SHIFT_LOGS_ONLY))

    scores = [chunk.score for chunk in candidates.sparse]
    assert scores == sorted(scores, reverse=True)


# --- multiple source merge ---------------------------------------------------------------------


async def test_candidates_from_every_source_are_merged_into_one_ranked_list() -> None:
    candidates = await _build().search(await _request(ALL_SOURCES))

    merged_sources = {chunk.metadata["source_id"] for chunk in candidates.sparse}
    assert len(merged_sources) > 1

    scores = [chunk.score for chunk in candidates.sparse]
    assert scores == sorted(scores, reverse=True)


async def test_the_merge_reconstructs_the_unrestricted_ranking() -> None:
    """The property that keeps the downstream fusion tail untouched: searching each source
    separately and merging by score gives exactly the list a single unrestricted search
    would have produced. If this ever fails, every retrieval threshold moves with it.
    """
    unrestricted = await BM25KeywordRetriever().search(QUERY, top_k=20, language="en")
    candidates = await _build().search(await _request(ALL_SOURCES))

    assert [chunk.chunk_id for chunk in candidates.sparse] == [
        chunk.chunk_id for chunk in unrestricted
    ]


async def test_the_two_methods_stay_in_separate_lists() -> None:
    """Fusion is what reconciles dense and keyword results, so they must arrive unmixed."""
    candidates = await _build().search(await _request(ALL_SOURCES))

    assert [chunk.chunk_id for chunk in candidates.dense] == ["dense-stub-chunk-1"]
    assert "dense-stub-chunk-1" not in [chunk.chunk_id for chunk in candidates.sparse]


async def test_the_merged_lists_respect_the_per_source_limit() -> None:
    request = SourceSearchRequest(
        query_text=QUERY,
        query_embedding=Embedding(vector=[0.1], model="test"),
        language="en",
        sources=ALL_SOURCES,
        limit_per_source=2,
    )

    candidates = await _build().search(request)

    assert len(candidates.sparse) <= 2
