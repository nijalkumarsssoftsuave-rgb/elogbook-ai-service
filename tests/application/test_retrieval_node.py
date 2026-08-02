from app.application.qa.nodes.retrieval import RetrievalNode
from app.domain.models import (
    PermissionScope,
    Question,
    RetrievedChunk,
    SearchScope,
    Source,
)

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

    async def retrieve(
        self, question: Question, search_scope: SearchScope, top_k: int = 5
    ) -> list[RetrievedChunk]:
        self.received.append((question, search_scope, top_k))
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
