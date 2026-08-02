import pytest

from app.application.dto import QueryRequestDTO, QueryResultDTO, TranscribeRequestDTO
from app.application.stt_service import STTApplicationService
from app.domain.exceptions import AudioRejectionReason, InvalidAudioError
from app.domain.models import AudioRequest, QueryOrigin, Transcript

# The orchestrator's collaborators are both concrete services, so the fakes here stand in
# for them directly -- their own behaviour is covered in test_transcription_service.py and
# test_qa_service.py.

TRANSCRIPT = Transcript(
    text="What caused the fire alarm during the night shift?",
    language="en",
    confidence=0.9,
    model="fake-asr",
)


class FakeTranscriptionService:
    def __init__(
        self,
        calls: list[str],
        error: Exception | None = None,
        transcript: Transcript = TRANSCRIPT,
    ) -> None:
        self.calls = calls
        self._error = error
        self._transcript = transcript
        self.received: list[AudioRequest] = []

    async def transcribe(self, audio: AudioRequest) -> Transcript:
        self.calls.append("transcribe")
        if self._error is not None:
            raise self._error
        self.received.append(audio)
        return self._transcript


class FakeQAService:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.received: list[QueryRequestDTO] = []

    async def execute(self, request: QueryRequestDTO) -> QueryResultDTO:
        self.calls.append("qa_execute")
        self.received.append(request)
        return QueryResultDTO(answer_text="A stub answer.", citations=[])


def _build_service(
    calls: list[str],
    transcription_error: Exception | None = None,
    transcript: Transcript = TRANSCRIPT,
) -> tuple[STTApplicationService, FakeTranscriptionService, FakeQAService]:
    transcription = FakeTranscriptionService(calls, error=transcription_error, transcript=transcript)
    qa = FakeQAService(calls)
    # The fakes satisfy the same call signatures as the concrete services; the type
    # ignores keep mypy honest about that being a test-only substitution.
    service = STTApplicationService(transcription, qa)  # type: ignore[arg-type]
    return service, transcription, qa


@pytest.fixture
def request_dto() -> TranscribeRequestDTO:
    return TranscribeRequestDTO(
        audio_bytes=b"pretend-this-is-a-wav-file",
        filename="question.wav",
        content_type="audio/wav",
        language_hint="en",
        user_id="u1",
        roles=["viewer"],
        correlation_id="cid-1",
        top_k=3,
    )


async def test_execute_transcribes_then_asks_the_qa_pipeline(
    request_dto: TranscribeRequestDTO,
) -> None:
    calls: list[str] = []
    service, _, _ = _build_service(calls)

    result = await service.execute(request_dto)

    assert calls == ["transcribe", "qa_execute"]
    assert result.transcript == TRANSCRIPT
    assert result.answer.answer_text == "A stub answer."


async def test_the_transcript_becomes_the_query_and_caller_context_is_carried_over(
    request_dto: TranscribeRequestDTO,
) -> None:
    calls: list[str] = []
    service, _, qa = _build_service(calls)

    await service.execute(request_dto)

    asked = qa.received[0]
    assert asked.query == TRANSCRIPT.text
    assert asked.user_id == "u1"
    assert asked.roles == ["viewer"]
    assert asked.correlation_id == "cid-1"
    assert asked.top_k == 3


async def test_audio_metadata_is_assembled_from_the_upload(
    request_dto: TranscribeRequestDTO,
) -> None:
    calls: list[str] = []
    service, transcription, _ = _build_service(calls)

    await service.execute(request_dto)

    metadata = transcription.received[0].metadata
    assert metadata.filename == "question.wav"
    assert metadata.content_type == "audio/wav"
    # Size is measured from the bytes we actually hold, never taken on trust from the
    # client's declared length.
    assert metadata.size_bytes == len(request_dto.audio_bytes)


async def test_the_query_carries_voice_provenance_to_the_audit_trail(
    request_dto: TranscribeRequestDTO,
) -> None:
    calls: list[str] = []
    timed = TRANSCRIPT.model_copy(update={"duration_seconds": 11.0})
    service, _, qa = _build_service(calls, transcript=timed)

    await service.execute(request_dto)

    provenance = qa.received[0].provenance
    assert provenance.origin is QueryOrigin.VOICE
    assert provenance.audio_duration_seconds == 11.0
    # Wall-clock: asserted on presence and sign, never on a magic number.
    assert provenance.transcription_duration_seconds is not None
    assert provenance.transcription_duration_seconds >= 0


async def test_an_engine_that_reports_no_duration_records_none_rather_than_zero(
    request_dto: TranscribeRequestDTO,
) -> None:
    """The stub speech backend cannot know a duration without decoding. Zero would read as
    "an empty recording"; None reads as "not measured", which is the truth.
    """
    calls: list[str] = []
    service, _, qa = _build_service(calls)  # TRANSCRIPT has duration_seconds=None

    await service.execute(request_dto)

    assert qa.received[0].provenance.audio_duration_seconds is None


async def test_a_padded_transcript_is_trimmed_before_it_becomes_a_question(
    request_dto: TranscribeRequestDTO,
) -> None:
    """Stray whitespace would not change the answer -- BM25 ignores it -- but it does
    change the cache key, so a voice question would miss the entry its identical typed
    twin just warmed. Both speech adapters happen to strip today; this makes it a rule.
    """
    calls: list[str] = []
    padded = TRANSCRIPT.model_copy(update={"text": f"  {TRANSCRIPT.text}  "})
    service, _, qa = _build_service(calls, transcript=padded)

    await service.execute(request_dto)

    assert qa.received[0].query == TRANSCRIPT.text


async def test_the_transcript_returned_is_the_question_that_was_asked(
    request_dto: TranscribeRequestDTO,
) -> None:
    """Reporting one string while having answered a different one would make a wrong
    answer impossible to diagnose from the response alone.
    """
    calls: list[str] = []
    padded = TRANSCRIPT.model_copy(update={"text": f"\n{TRANSCRIPT.text}\t"})
    service, _, qa = _build_service(calls, transcript=padded)

    result = await service.execute(request_dto)

    assert result.transcript.text == qa.received[0].query


async def test_the_language_hint_is_not_forwarded_to_the_qa_pipeline(
    request_dto: TranscribeRequestDTO,
) -> None:
    """The hint steers the speech engine only. Past the seam the pipeline detects language
    from the transcript, exactly as it does for typed input -- so QueryRequestDTO has
    nowhere to put a hint, and this pins that it stays that way.
    """
    calls: list[str] = []
    service, _, qa = _build_service(calls)

    await service.execute(request_dto)

    assert not hasattr(qa.received[0], "language_hint")


async def test_a_rejected_upload_never_reaches_the_qa_pipeline(
    request_dto: TranscribeRequestDTO,
) -> None:
    calls: list[str] = []
    service, _, qa = _build_service(
        calls,
        transcription_error=InvalidAudioError(
            AudioRejectionReason.EMPTY_AUDIO, "The uploaded audio file is empty."
        ),
    )

    with pytest.raises(InvalidAudioError):
        await service.execute(request_dto)

    assert calls == ["transcribe"]
    assert qa.received == []
