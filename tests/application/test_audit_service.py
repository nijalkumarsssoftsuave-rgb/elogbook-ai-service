from app.application.audit_service import AuditService
from app.domain.models import GroundedAnswer, Question


class FakeCache:
    def __init__(self, hit: GroundedAnswer | None = None) -> None:
        self._hit = hit
        self.requested_keys: list[str] = []
        self.stored: list[tuple[str, GroundedAnswer]] = []

    async def get(self, key: str) -> GroundedAnswer | None:
        self.requested_keys.append(key)
        return self._hit

    async def set(self, key: str, value: GroundedAnswer, ttl_seconds: int = 300) -> None:
        self.stored.append((key, value))


class FakeAudit:
    def __init__(self) -> None:
        self.recorded: list[tuple[Question, GroundedAnswer, str]] = []

    async def record_query(
        self, question: Question, answer: GroundedAnswer, correlation_id: str
    ) -> None:
        self.recorded.append((question, answer, correlation_id))


def _question(text: str = "what happened on the last shift?") -> Question:
    return Question(text=text, user_id="u1", language="en")


def _answer(refused: bool = False) -> GroundedAnswer:
    return GroundedAnswer(answer_text="an answer", refused=refused)


async def test_get_cached_returns_the_cached_answer() -> None:
    cached = _answer()
    service = AuditService(FakeCache(hit=cached), FakeAudit())

    assert await service.get_cached(_question()) is cached


async def test_finalize_audits_and_caches_a_genuine_answer() -> None:
    cache, audit = FakeCache(), FakeAudit()
    service = AuditService(cache, audit)
    answer = _answer()

    await service.finalize(_question(), answer, "cid-1")

    assert len(audit.recorded) == 1
    assert audit.recorded[0][2] == "cid-1"
    assert [stored_answer for _, stored_answer in cache.stored] == [answer]


async def test_finalize_audits_a_refusal_but_never_caches_it() -> None:
    cache, audit = FakeCache(), FakeAudit()
    service = AuditService(cache, audit)

    await service.finalize(_question(), _answer(refused=True), "cid-1")

    assert len(audit.recorded) == 1
    assert cache.stored == []


async def test_record_cache_hit_audits_without_writing_to_the_cache() -> None:
    cache, audit = FakeCache(), FakeAudit()
    service = AuditService(cache, audit)

    await service.record_cache_hit(_question(), _answer(), "cid-1")

    assert len(audit.recorded) == 1
    assert cache.stored == []


async def test_the_same_question_reads_and_writes_the_same_cache_key() -> None:
    cache, audit = FakeCache(), FakeAudit()
    service = AuditService(cache, audit)

    await service.get_cached(_question())
    await service.finalize(_question(), _answer(), "cid-1")

    assert cache.requested_keys[0] == cache.stored[0][0]


async def test_different_questions_use_different_cache_keys() -> None:
    cache, audit = FakeCache(), FakeAudit()
    service = AuditService(cache, audit)

    await service.get_cached(_question("first question"))
    await service.get_cached(_question("second question"))

    assert cache.requested_keys[0] != cache.requested_keys[1]
