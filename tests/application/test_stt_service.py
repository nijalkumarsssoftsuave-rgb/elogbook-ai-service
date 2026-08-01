import pytest

from app.application.dto import QueryRequestDTO, QueryResultDTO, TranscribeRequestDTO
from app.application.stt_service import STTApplicationService
from app.domain.exceptions import AudioRejectionReason, InvalidAudioError
from app.domain.models import AudioRequest, Transcript

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
    def __init__(self, calls: list[str], error: Exception | None = None) -> None:
        self.calls = calls
        self._error = error
        self.received: list[AudioRequest] = []

    async def transcribe(self, audio: AudioRequest) -> Transcript:
        self.calls.append("transcribe")
        if self._error is not None:
            raise self._error
        self.received.append(audio)
        return TRANSCRIPT


class FakeQAService:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.received: list[QueryRequestDTO] = []

    async def execute(self, request: QueryRequestDTO) -> QueryResultDTO:
        self.calls.append("qa_execute")
        self.received.append(request)
        return QueryResultDTO(answer_text="A stub answer.", citations=[])


def _build_service(
    calls: list[str], transcription_error: Exception | None = None
) -> tuple[STTApplicationService, FakeTranscriptionService, FakeQAService]:
    transcription = FakeTranscriptionService(calls, error=transcription_error)
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
