import pytest

from app.application.audit_service import AuditService
from app.application.citation.citation_resolver import CitationResolver
from app.application.citation.citation_validator import CitationValidator
from app.application.dto import QueryRequestDTO
from app.application.guardrail_service import GuardrailService
from app.application.language_detection_service import LanguageDetectionService
from app.application.qa.nodes.generation import GenerationNode
from app.application.qa.nodes.retrieval import RetrievalNode
from app.application.qa_service import QAApplicationService
from app.application.retrieval_service import RetrievalService
from app.domain.citation import Citation
from app.domain.exceptions import UnsupportedLanguageError
from app.domain.models import (
    AuditRecord,
    DetectedLanguage,
    Embedding,
    GeneratedAnswer,
    GenerationRequest,
    GroundedAnswer,
    PermissionScope,
    Question,
    RetrievalCandidates,
    RetrievedChunk,
    SearchScope,
    Source,
    SourceSearchRequest,
)

# The services under test are concrete classes, so the fakes sit one level down at the
# port boundary: every collaborator below is a real service wrapping fake ports.

EVIDENCE = RetrievedChunk(
    chunk_id="log-001",
    document_id="doc-log-001",
    text="Morning shift equipment check completed.",
    score=1.0,
    metadata={"source_title": "Morning Shift Equipment Log"},
)


class FakeLanguageDetector:
    def __init__(self, calls: list[str], code: str = "en") -> None:
        self.calls = calls
        self.code = code

    async def detect(self, text: str) -> DetectedLanguage:
        self.calls.append("detect")
        return DetectedLanguage(code=self.code, confidence=0.99)


class FakeEmbedding:
    async def embed(self, text: str) -> Embedding:
        return Embedding(vector=[0.1], model="fake")


class FakePermissionResolver:
    """Resolves to one source; the resolution rules themselves are covered in
    tests/application/test_permission_resolver.py.
    """

    async def resolve(self, scope: PermissionScope) -> SearchScope:
        return SearchScope(sources=[Source(source_id="shift-logs", display_name="Shift Logs")])


class FakeMultiSourceRetriever:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    async def search(self, request: SourceSearchRequest) -> RetrievalCandidates:
        self.calls.append("retrieve")
        return RetrievalCandidates(
            searched_source_ids=request.search_scope.source_ids,
            dense=[],
            sparse=[EVIDENCE],
        )


class FakeReranker:
    async def rerank(
        self, question: Question, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        return chunks


class FakeGuardrail:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    async def check_input(self, question: Question) -> None:
        self.calls.append("check_input")

    async def check_retrieved_chunks(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        self.calls.append("check_retrieved_chunks")
        return chunks

    async def check_output(self, answer: GroundedAnswer) -> GroundedAnswer:
        self.calls.append("check_output")
        return answer


class FakeModelClient:
    """Returns each configured completion in turn, repeating the last one thereafter."""

    def __init__(self, calls: list[str], completions: list[str]) -> None:
        self.calls = calls
        self._completions = completions
        self.attempts = 0

    async def generate(self, request: GenerationRequest) -> str:
        self.calls.append("generate")
        completion = self._completions[min(self.attempts, len(self._completions) - 1)]
        self.attempts += 1
        return completion


class FakeCache:
    def __init__(self, calls: list[str], hit: GroundedAnswer | None = None) -> None:
        self.calls = calls
        self._hit = hit
        self.stored: list[GroundedAnswer] = []

    async def get(self, key: str) -> GroundedAnswer | None:
        self.calls.append("cache_get")
        return self._hit

    async def set(self, key: str, value: GroundedAnswer, ttl_seconds: int = 300) -> None:
        self.calls.append("cache_set")
        self.stored.append(value)


class FakeAudit:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.recorded: list[AuditRecord] = []

    async def record(self, record: AuditRecord) -> None:
        self.calls.append("record_query")
        self.recorded.append(record)


class PhantomGenerationNode:
    """Returns an answer citing a chunk that is not among the retrieved evidence.

    The real GenerationNode cannot produce this: it builds references only from the labels
    it issued, for chunks it was handed. A stand-in is the only way to reach the
    orchestrator's UNRESOLVED_CITATION branch, and that branch is worth reaching -- it is
    what stops a partially resolved answer going out if the chunk list ever moves underneath
    an answer.
    """

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    async def generate(
        self, question: Question, context_chunks: list[RetrievedChunk]
    ) -> GeneratedAnswer:
        self.calls.append("generate")
        return GeneratedAnswer(
            answer_text="The check was completed [c1].",
            citations=[Citation(citation_id="c1", chunk_id="log-nowhere", order=1)],
        )


# A completion citing the evidence label the node issues for the single retrieved chunk,
# and one citing nothing.
GROUNDED_COMPLETION = "The check was completed [c1]."
UNGROUNDED_COMPLETION = "I believe so, but cannot point at any source."


def _build_service(
    calls: list[str],
    completions: list[str],
    cache: FakeCache,
    audit: FakeAudit,
    language_code: str = "en",
    generation_node: GenerationNode | PhantomGenerationNode | None = None,
) -> QAApplicationService:
    return QAApplicationService(
        LanguageDetectionService(
            detector=FakeLanguageDetector(calls, code=language_code), supported_languages=["en"]
        ),
        GuardrailService(FakeGuardrail(calls)),
        RetrievalNode(
            FakePermissionResolver(),
            RetrievalService(
                FakeEmbedding(), FakeMultiSourceRetriever(calls), FakeReranker()
            ),
        ),
        generation_node or GenerationNode(model_client=FakeModelClient(calls, completions)),
        CitationResolver(),
        CitationValidator(),
        AuditService(cache, audit),
    )


@pytest.fixture
def request_dto() -> QueryRequestDTO:
    return QueryRequestDTO(
        query="What happened on the last shift?",
        user_id="u1",
        roles=["viewer"],
        correlation_id="cid-1",
        top_k=3,
        permission_scope=PermissionScope.from_roles(["viewer"]),
    )


async def test_execute_runs_the_pipeline_in_order_on_a_cache_miss(
    request_dto: QueryRequestDTO,
) -> None:
    calls: list[str] = []
    cache, audit = FakeCache(calls), FakeAudit(calls)
    service = _build_service(calls, [GROUNDED_COMPLETION], cache, audit)

    result = await service.execute(request_dto)

    assert calls == [
        "detect",
        "cache_get",
        "check_input",
        "retrieve",
        "check_retrieved_chunks",
        "generate",
        "check_output",
        "record_query",
        "cache_set",
    ]
    assert result.cache_hit is False
    assert result.refused is False
    assert [citation.chunk_id for citation in result.citations] == [EVIDENCE.chunk_id]
    # Resolution ran: the reference has become a full citation, carrying detail only the
    # retrieved chunk could supply.
    assert result.citations[0].citation_id == "c1"
    assert result.citations[0].source_title == "Morning Shift Equipment Log"


async def test_execute_returns_the_cached_answer_without_retrieving_or_generating(
    request_dto: QueryRequestDTO,
) -> None:
    calls: list[str] = []
    cached = GroundedAnswer(answer_text="cached answer")
    cache, audit = FakeCache(calls, hit=cached), FakeAudit(calls)
    service = _build_service(calls, [GROUNDED_COMPLETION], cache, audit)

    result = await service.execute(request_dto)

    assert calls == ["detect", "cache_get", "record_query"]
    assert result.cache_hit is True
    assert result.answer_text == "cached answer"


async def test_execute_retries_generation_once_when_the_first_answer_is_uncited(
    request_dto: QueryRequestDTO,
) -> None:
    calls: list[str] = []
    cache, audit = FakeCache(calls), FakeAudit(calls)
    service = _build_service(
        calls, [UNGROUNDED_COMPLETION, GROUNDED_COMPLETION], cache, audit
    )

    result = await service.execute(request_dto)

    assert calls.count("generate") == 2
    assert result.refused is False
    assert [citation.chunk_id for citation in result.citations] == [EVIDENCE.chunk_id]
    assert cache.stored  # a valid answer on the retry is still cached


async def test_execute_refuses_after_two_failed_attempts_and_does_not_cache_the_refusal(
    request_dto: QueryRequestDTO,
) -> None:
    calls: list[str] = []
    cache, audit = FakeCache(calls), FakeAudit(calls)
    service = _build_service(calls, [UNGROUNDED_COMPLETION], cache, audit)

    result = await service.execute(request_dto)

    assert calls.count("generate") == 2  # the retry is bounded, not unlimited
    assert result.refused is True
    assert result.is_grounded is False
    assert result.answer_text == GroundedAnswer.REFUSAL_TEXT
    assert result.citations == []
    assert result.citations == []  # a refusal resolves nothing
    assert "record_query" in calls  # refusals are still audited
    assert "cache_set" not in calls  # but never cached
    assert audit.recorded[0].answer.refused is True


async def test_execute_refuses_rather_than_returning_an_answer_with_a_citation_that_did_not_resolve(
    request_dto: QueryRequestDTO,
) -> None:
    """The decision ES-331 adds. Resolution drops the unmatched reference, so without the
    validator downstream this answer would go out with its claim intact and its supporting
    citation quietly missing. Instead the whole response is rejected.
    """
    calls: list[str] = []
    cache, audit = FakeCache(calls), FakeAudit(calls)
    service = _build_service(
        calls, [], cache, audit, generation_node=PhantomGenerationNode(calls)
    )

    result = await service.execute(request_dto)

    assert calls.count("generate") == 2  # retried first, then refused
    assert result.refused is True
    assert result.answer_text == GroundedAnswer.REFUSAL_TEXT
    assert result.citations == []
    # The answer the model actually produced never reaches the caller, not even in part.
    assert "The check was completed" not in result.answer_text
    assert "cache_set" not in calls


async def test_execute_rejects_an_unsupported_language_before_anything_else(
    request_dto: QueryRequestDTO,
) -> None:
    calls: list[str] = []
    cache, audit = FakeCache(calls), FakeAudit(calls)
    service = _build_service(calls, [GROUNDED_COMPLETION], cache, audit, language_code="fr")

    with pytest.raises(UnsupportedLanguageError):
        await service.execute(request_dto)

    assert calls == ["detect"]
