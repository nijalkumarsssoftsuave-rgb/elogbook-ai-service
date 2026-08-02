from app.application.qa.nodes.retrieval import RetrievalNode
from app.domain.models import (
    PermissionScope,
    Question,
    RetrievalSearchContext,
    RetrievedChunk,
)

CHUNK = RetrievedChunk(chunk_id="log-001", document_id="doc-log-001", text="t", score=1.0)


class FakeRetrievalService:
    def __init__(self) -> None:
        self.received: list[RetrievalSearchContext] = []

    async def retrieve(self, context: RetrievalSearchContext) -> list[RetrievedChunk]:
        self.received.append(context)
        return [CHUNK]


def _question(language: str = "en") -> Question:
    return Question(
        text="what happened on the last shift?",
        user_id="u1",
        roles=["contractor"],
        language=language,
    )


async def test_the_node_builds_a_search_context_from_the_question_scope_and_top_k() -> None:
    """Phase 2's whole job: accept the scope, build the request, invoke the service."""
    service = FakeRetrievalService()
    scope = PermissionScope.from_roles(["contractor"])

    await RetrievalNode(service).run(_question(), scope, top_k=3)

    context = service.received[0]
    assert context.question.text == "what happened on the last shift?"
    assert context.permission_scope == scope
    assert context.top_k == 3


async def test_the_permission_scope_is_passed_through_untouched() -> None:
    """The node carries the scope; it does not interpret it. Resolving roles into sources
    belongs to the resolver, and doing any of it here would mean two places decide access.
    """
    service = FakeRetrievalService()
    scope = PermissionScope.from_roles(["contractor", "viewer"])

    await RetrievalNode(service).run(_question(), scope)

    assert service.received[0].permission_scope.roles == ["contractor", "viewer"]


async def test_an_empty_scope_is_forwarded_rather_than_widened() -> None:
    """The tempting bug: treating "no roles" as "no restriction". The node must hand the
    empty scope down and let retrieval fail closed on it.
    """
    service = FakeRetrievalService()

    await RetrievalNode(service).run(_question(), PermissionScope())

    assert service.received[0].permission_scope.roles == []


async def test_the_node_returns_the_services_chunks_unchanged() -> None:
    """Thin by design: no filtering, no reranking, no reshaping on the way back."""
    service = FakeRetrievalService()

    chunks = await RetrievalNode(service).run(_question(), PermissionScope.from_roles(["viewer"]))

    assert chunks == [CHUNK]


async def test_the_question_language_survives_the_hand_off() -> None:
    service = FakeRetrievalService()

    await RetrievalNode(service).run(_question("ar"), PermissionScope.from_roles(["viewer"]))

    assert service.received[0].question.language == "ar"
