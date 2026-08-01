import pytest

from app.application.dto import QueryRequestDTO
from app.application.qa_service import QAApplicationService
from app.domain.models import GroundedAnswer, RetrievedChunk


class FakeRetriever:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    async def retrieve(self, question, top_k: int = 5) -> list[RetrievedChunk]:
        self.calls.append("retrieve")
        return [RetrievedChunk(chunk_id="c1", document_id="d1", text="chunk", score=0.9)]


class FakeModelClient:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    async def generate(self, question, context_chunks) -> GroundedAnswer:
        self.calls.append("generate")
        return GroundedAnswer(answer_text="answer", citations=[], confidence=0.8, is_grounded=True)


class FakeGuardrail:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    async def check_input(self, question) -> None:
        self.calls.append("check_input")

    async def check_output(self, answer: GroundedAnswer) -> GroundedAnswer:
        self.calls.append("check_output")
        return answer


class FakeCache:
    def __init__(self, calls: list[str], hit: GroundedAnswer | None = None) -> None:
        self.calls = calls
        self._hit = hit
        self.stored: GroundedAnswer | None = None

    async def get(self, key: str) -> GroundedAnswer | None:
        self.calls.append("cache_get")
        return self._hit

    async def set(self, key: str, value: GroundedAnswer, ttl_seconds: int = 300) -> None:
        self.calls.append("cache_set")
        self.stored = value


class FakeAudit:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.recorded: list[tuple] = []

    async def record_query(self, question, answer, correlation_id: str) -> None:
        self.calls.append("audit")
        self.recorded.append((question, answer, correlation_id))


@pytest.fixture
def request_dto() -> QueryRequestDTO:
    return QueryRequestDTO(
        query="What happened on the last shift?",
        user_id="u1",
        roles=["viewer"],
        correlation_id="cid-1",
        top_k=3,
    )


async def test_execute_calls_ports_in_order_on_cache_miss(request_dto: QueryRequestDTO) -> None:
    calls: list[str] = []
    retriever = FakeRetriever(calls)
    model_client = FakeModelClient(calls)
    guardrail = FakeGuardrail(calls)
    cache = FakeCache(calls, hit=None)
    audit = FakeAudit(calls)

    service = QAApplicationService(retriever, model_client, guardrail, cache, audit)
    result = await service.execute(request_dto)

    assert calls == [
        "cache_get",
        "check_input",
        "retrieve",
        "generate",
        "check_output",
        "cache_set",
        "audit",
    ]
    assert result.cache_hit is False
    assert result.answer_text == "answer"
    assert audit.recorded[0][2] == "cid-1"


async def test_execute_skips_retrieval_and_generation_on_cache_hit(
    request_dto: QueryRequestDTO,
) -> None:
    calls: list[str] = []
    cached_answer = GroundedAnswer(
        answer_text="cached answer", citations=[], confidence=0.9, is_grounded=True
    )
    retriever = FakeRetriever(calls)
    model_client = FakeModelClient(calls)
    guardrail = FakeGuardrail(calls)
    cache = FakeCache(calls, hit=cached_answer)
    audit = FakeAudit(calls)

    service = QAApplicationService(retriever, model_client, guardrail, cache, audit)
    result = await service.execute(request_dto)

    assert calls == ["cache_get", "audit"]
    assert result.cache_hit is True
    assert result.answer_text == "cached answer"
