from app.application.audit_service import AuditService
from app.domain.models import (
    AuditRecord,
    GroundedAnswer,
    QueryOrigin,
    QueryProvenance,
    Question,
)

TEXT_PROVENANCE = QueryProvenance()
VOICE_PROVENANCE = QueryProvenance(
    origin=QueryOrigin.VOICE, audio_duration_seconds=11.0, transcription_duration_seconds=1.4
)


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
        self.recorded: list[AuditRecord] = []

    async def record(self, record: AuditRecord) -> None:
        self.recorded.append(record)


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

    await service.finalize(_question(), answer, "cid-1", TEXT_PROVENANCE)

    assert len(audit.recorded) == 1
    assert audit.recorded[0].correlation_id == "cid-1"
    assert [stored_answer for _, stored_answer in cache.stored] == [answer]


async def test_finalize_audits_a_refusal_but_never_caches_it() -> None:
    cache, audit = FakeCache(), FakeAudit()
    service = AuditService(cache, audit)

    await service.finalize(_question(), _answer(refused=True), "cid-1", TEXT_PROVENANCE)

    assert len(audit.recorded) == 1
    assert cache.stored == []


async def test_record_cache_hit_audits_without_writing_to_the_cache() -> None:
    cache, audit = FakeCache(), FakeAudit()
    service = AuditService(cache, audit)

    await service.record_cache_hit(_question(), _answer(), "cid-1", TEXT_PROVENANCE)

    assert len(audit.recorded) == 1
    assert cache.stored == []


async def test_the_same_question_reads_and_writes_the_same_cache_key() -> None:
    cache, audit = FakeCache(), FakeAudit()
    service = AuditService(cache, audit)

    await service.get_cached(_question())
    await service.finalize(_question(), _answer(), "cid-1", TEXT_PROVENANCE)

    assert cache.requested_keys[0] == cache.stored[0][0]


async def test_different_questions_use_different_cache_keys() -> None:
    cache, audit = FakeCache(), FakeAudit()
    service = AuditService(cache, audit)

    await service.get_cached(_question("first question"))
    await service.get_cached(_question("second question"))

    assert cache.requested_keys[0] != cache.requested_keys[1]


async def test_the_record_carries_the_provenance_it_was_given() -> None:
    cache, audit = FakeCache(), FakeAudit()
    service = AuditService(cache, audit)

    await service.finalize(_question(), _answer(), "cid-1", VOICE_PROVENANCE)

    record = audit.recorded[0]
    assert record.provenance.origin is QueryOrigin.VOICE
    assert record.provenance.audio_duration_seconds == 11.0
    assert record.provenance.transcription_duration_seconds == 1.4


async def test_a_cache_hit_is_audited_with_its_provenance_too() -> None:
    """A spoken question answered from cache is still a spoken question. Losing the origin
    on the cheap path would make voice usage look lower than it is.
    """
    cache, audit = FakeCache(), FakeAudit()
    service = AuditService(cache, audit)

    await service.record_cache_hit(_question(), _answer(), "cid-1", VOICE_PROVENANCE)

    assert audit.recorded[0].provenance.origin is QueryOrigin.VOICE


async def test_the_record_language_tracks_the_question_it_describes() -> None:
    """Computed rather than stored, so the audit row's language column cannot drift away
    from the question it belongs to.
    """
    cache, audit = FakeCache(), FakeAudit()
    service = AuditService(cache, audit)
    arabic_question = Question(text="a question", user_id="u1", language="ar")

    await service.finalize(arabic_question, _answer(), "cid-1", TEXT_PROVENANCE)

    record = audit.recorded[0]
    assert record.language == "ar"
    # And it survives serialization, which is what a real SQL adapter will write.
    assert record.model_dump()["language"] == "ar"


async def test_provenance_does_not_reach_the_cache_key() -> None:
    """The quiet failure this guards against: if provenance ever entered the key, a spoken
    question would stop sharing cache entries with the identical typed one, and ES-325's
    voice/text guarantee would rot without any test failing.
    """
    cache, audit = FakeCache(), FakeAudit()
    service = AuditService(cache, audit)

    await service.finalize(_question(), _answer(), "cid-1", TEXT_PROVENANCE)
    await service.finalize(_question(), _answer(), "cid-2", VOICE_PROVENANCE)

    typed_key, voice_key = (key for key, _ in cache.stored)
    assert typed_key == voice_key
