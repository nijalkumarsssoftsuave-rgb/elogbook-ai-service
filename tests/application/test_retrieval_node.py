from datetime import date

from app.application.qa.nodes.retrieval import RetrievalNode
from app.domain.models import Question, RetrievedChunk
from app.domain.permission import PermissionScope, SearchScope, Source
from app.domain.query import QueryFilters

CHUNK = RetrievedChunk(chunk_id="log-001", document_id="doc-log-001", text="t", score=1.0)
SHIFT_LOGS = Source(source_id="shift-logs", display_name="Shift Logs")
RESOLVED = SearchScope(sources=[SHIFT_LOGS], department_ids=["maintenance"])


class FakePermissionResolver:
    def __init__(self, scope: SearchScope = RESOLVED) -> None:
        self._scope = scope
        self.received: list[PermissionScope] = []

    async def resolve(self, scope: PermissionScope) -> SearchScope:
        self.received.append(scope)
        return self._scope


class FakeRetrievalService:
    def __init__(self) -> None:
        self.received: list[tuple[Question, SearchScope, int]] = []
        self.filters: list[QueryFilters | None] = []

    async def retrieve(
        self,
        question: Question,
        search_scope: SearchScope,
        top_k: int = 5,
        filters: QueryFilters | None = None,
    ) -> list[RetrievedChunk]:
        self.received.append((question, search_scope, top_k))
        self.filters.append(filters)
        return [CHUNK]


def _question(language: str = "en") -> Question:
    return Question(
        text="what happened on the last shift?",
        user_id="u1",
        roles=["contractor"],
        language=language,
    )


def _build(
    resolver: FakePermissionResolver | None = None,
    service: FakeRetrievalService | None = None,
) -> tuple[RetrievalNode, FakePermissionResolver, FakeRetrievalService]:
    resolver = resolver or FakePermissionResolver()
    service = service or FakeRetrievalService()
    return RetrievalNode(resolver, service), resolver, service  # type: ignore[arg-type]


async def test_the_node_resolves_permissions_before_retrieving() -> None:
    """The step ES-328 moved up out of RetrievalService: RBAC becomes a search set here."""
    node, resolver, service = _build()

    await node.run(_question(), PermissionScope.from_roles(["contractor"]), top_k=3)

    assert [scope.roles for scope in resolver.received] == [["contractor"]]
    assert service.received[0][1] == RESOLVED


async def test_the_node_resolves_on_every_call_rather_than_trusting_a_caller() -> None:
    """This is the guard that replaces what the old signature enforced structurally.

    While RetrievalService took a PermissionScope it was impossible to retrieve with a
    hand-made source list. Now that it accepts a SearchScope, the thing keeping RBAC in the
    path is that this node always resolves -- so that is what gets pinned.
    """
    node, resolver, service = _build()

    await node.run(_question(), PermissionScope.from_roles(["viewer"]))
    await node.run(_question(), PermissionScope.from_roles(["contractor"]))

    assert len(resolver.received) == 2
    assert len(service.received) == 2
    assert all(scope is RESOLVED for _, scope, _ in service.received)


async def test_an_empty_resolution_still_reaches_the_service() -> None:
    """The node does not second-guess an empty search set. Refusing to search it is the
    service's job, and having exactly one place fail closed is the point.
    """
    node, _, service = _build(resolver=FakePermissionResolver(SearchScope()))

    await node.run(_question(), PermissionScope())

    assert service.received[0][1].is_empty


async def test_the_question_and_top_k_travel_unchanged() -> None:
    node, _, service = _build()

    await node.run(_question("ar"), PermissionScope.from_roles(["viewer"]), top_k=7)

    question, _, top_k = service.received[0]
    assert question.language == "ar"
    assert question.text == "what happened on the last shift?"
    assert top_k == 7


async def test_the_node_returns_the_services_chunks_unchanged() -> None:
    """Thin by design: no filtering, no reranking, no reshaping on the way back."""
    node, _, _ = _build()

    chunks = await node.run(_question(), PermissionScope.from_roles(["viewer"]))

    assert chunks == [CHUNK]


# --- ES-337: the caller's filters reach retrieval -----------------------------------------------

_FILTERS = QueryFilters(
    area_ids=["north"], date_from=date(2026, 7, 1), date_to=date(2026, 7, 31)
)


async def test_filters_reach_the_retrieval_service_unchanged() -> None:
    node, _, service = _build()

    await node.run(_question(), PermissionScope.from_roles(["viewer"]), filters=_FILTERS)

    assert service.filters == [_FILTERS]
    assert service.filters[0].area_ids == ["north"]
    assert service.filters[0].date_from == date(2026, 7, 1)
    assert service.filters[0].date_to == date(2026, 7, 31)


async def test_a_request_with_no_filters_still_reaches_retrieval() -> None:
    """The node supplies an empty QueryFilters rather than None, so nothing downstream has
    to decide what a missing filter set means.
    """
    node, _, service = _build()

    await node.run(_question(), PermissionScope.from_roles(["viewer"]))

    assert service.filters[0] == QueryFilters()
    assert service.filters[0].is_empty


async def test_filters_are_not_merged_into_the_resolved_search_scope() -> None:
    """The distinction ES-337 exists to protect. A SearchScope is an entitlement resolved
    from the caller's roles; filters are a preference stated in the request. Copying one
    into the other -- the fields share names -- is how a request would come to widen its own
    access instead of narrowing it.
    """
    node, _, service = _build()

    await node.run(
        _question(),
        PermissionScope.from_roles(["viewer"]),
        filters=QueryFilters(area_ids=["south"], department_ids=["logistics"]),
    )

    _, resolved_scope, _ = service.received[0]
    assert resolved_scope == RESOLVED
    assert resolved_scope.area_ids == []
    assert resolved_scope.department_ids == ["maintenance"]


async def test_the_node_does_not_inspect_or_rewrite_the_filters() -> None:
    """It forwards. Anything that reads a filter here would be a second place deciding what
    filters mean, and the two would drift.
    """
    filters = QueryFilters(area_ids=["north"], company_ids=["acme-industrial"])
    node, _, service = _build()

    await node.run(_question(), PermissionScope.from_roles(["viewer"]), filters=filters)

    assert service.filters[0] is filters
