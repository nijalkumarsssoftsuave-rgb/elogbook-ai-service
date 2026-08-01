import pytest

from app.application.transcription_service import TranscriptionService
from app.domain.exceptions import (
    AudioRejectionReason,
    InvalidAudioError,
    UnsupportedLanguageError,
)
from app.domain.models import AudioMetadata, AudioRequest, LanguageHint, Transcript

ALLOWED_CONTENT_TYPES = ["audio/wav", "audio/mpeg", "audio/ogg"]
MAX_AUDIO_BYTES = 1_000
SUPPORTED_LANGUAGES = ["en", "ar"]


class FakeSpeechToText:
    """Records what it was handed, so tests can assert the hint reaches the engine."""

    def __init__(self, text: str = "What happened on the night shift?") -> None:
        self._text = text
        self.received: list[AudioRequest] = []

    async def transcribe(self, audio: AudioRequest) -> Transcript:
        self.received.append(audio)
        return Transcript(text=self._text, language="en", confidence=0.9, model="fake")


def _audio(
    *,
    content: bytes = b"audio-bytes",
    filename: str = "question.wav",
    content_type: str = "audio/wav",
    size_bytes: int | None = None,
    language_hint: str | None = None,
) -> AudioRequest:
    return AudioRequest(
        content=content,
        metadata=AudioMetadata(
            filename=filename,
            content_type=content_type,
            size_bytes=len(content) if size_bytes is None else size_bytes,
        ),
        language_hint=LanguageHint.from_optional(language_hint),
    )


def _build_service(speech_to_text: FakeSpeechToText) -> TranscriptionService:
    return TranscriptionService(
        speech_to_text=speech_to_text,
        allowed_content_types=ALLOWED_CONTENT_TYPES,
        max_audio_bytes=MAX_AUDIO_BYTES,
        supported_languages=SUPPORTED_LANGUAGES,
    )


async def test_a_valid_upload_is_transcribed() -> None:
    engine = FakeSpeechToText()

    transcript = await _build_service(engine).transcribe(_audio())

    assert transcript.text == "What happened on the night shift?"
    assert len(engine.received) == 1


async def test_the_language_hint_is_passed_through_to_the_engine() -> None:
    engine = FakeSpeechToText()

    await _build_service(engine).transcribe(_audio(language_hint=" AR "))

    hint = engine.received[0].language_hint
    assert hint is not None
    assert hint.code == "ar"  # normalized once, in the domain model


async def test_an_empty_file_is_rejected_before_the_engine_is_called() -> None:
    engine = FakeSpeechToText()

    with pytest.raises(InvalidAudioError) as excinfo:
        await _build_service(engine).transcribe(_audio(content=b""))

    assert excinfo.value.reason is AudioRejectionReason.EMPTY_AUDIO
    assert engine.received == []


async def test_an_oversized_file_is_rejected_with_both_numbers() -> None:
    engine = FakeSpeechToText()

    with pytest.raises(InvalidAudioError) as excinfo:
        await _build_service(engine).transcribe(_audio(size_bytes=MAX_AUDIO_BYTES + 1))

    assert excinfo.value.reason is AudioRejectionReason.FILE_TOO_LARGE
    # Both numbers travel with the error so the caller can size the next attempt.
    assert excinfo.value.details["size_bytes"] == MAX_AUDIO_BYTES + 1
    assert excinfo.value.details["max_bytes"] == MAX_AUDIO_BYTES
    assert engine.received == []


async def test_a_file_exactly_at_the_limit_is_accepted() -> None:
    """The ceiling is inclusive: a file of exactly max_bytes is not too large."""
    engine = FakeSpeechToText()

    await _build_service(engine).transcribe(_audio(size_bytes=MAX_AUDIO_BYTES))

    assert len(engine.received) == 1


async def test_an_unsupported_content_type_is_rejected() -> None:
    engine = FakeSpeechToText()

    with pytest.raises(InvalidAudioError) as excinfo:
        await _build_service(engine).transcribe(
            _audio(filename="notes.txt", content_type="text/plain")
        )

    assert excinfo.value.reason is AudioRejectionReason.UNSUPPORTED_CONTENT_TYPE
    assert excinfo.value.details["content_type"] == "text/plain"
    assert engine.received == []


async def test_an_unsupported_language_hint_is_rejected() -> None:
    engine = FakeSpeechToText()

    with pytest.raises(UnsupportedLanguageError) as excinfo:
        await _build_service(engine).transcribe(_audio(language_hint="fr"))

    assert excinfo.value.language_code == "fr"
    assert engine.received == []


async def test_silence_is_rejected_rather_than_queried_as_an_empty_question() -> None:
    """A blank transcript would otherwise reach the QA pipeline as an empty query and
    come back with a confidently retrieved answer to a question nobody asked.
    """
    engine = FakeSpeechToText(text="   ")

    with pytest.raises(InvalidAudioError) as excinfo:
        await _build_service(engine).transcribe(_audio())

    assert excinfo.value.reason is AudioRejectionReason.NO_SPEECH_DETECTED


@pytest.mark.parametrize(
    ("declared", "filename", "expected"),
    [
        ("audio/wav", "q.wav", "audio/wav"),
        ("AUDIO/WAV", "q.wav", "audio/wav"),  # normalized
        ("", "q.wav", "audio/wav"),  # nothing declared -> extension
        ("application/octet-stream", "q.mp3", "audio/mpeg"),  # curl's default
        ("application/octet-stream", "q.bin", "application/octet-stream"),  # unknown
        ("audio/mpeg", "q.txt", "audio/mpeg"),  # declared type wins over extension
    ],
)
def test_content_type_resolution(declared: str, filename: str, expected: str) -> None:
    assert TranscriptionService.resolve_content_type(declared, filename) == expected


async def test_an_extension_only_upload_is_accepted() -> None:
    """Proves the fallback pays off end-to-end: curl sends application/octet-stream for a
    perfectly ordinary .wav, and rejecting that would break the common case.
    """
    engine = FakeSpeechToText()

    await _build_service(engine).transcribe(
        _audio(filename="question.wav", content_type="application/octet-stream")
    )

    assert len(engine.received) == 1
