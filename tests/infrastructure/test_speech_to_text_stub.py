from app.domain.models import AudioMetadata, AudioRequest, LanguageHint
from app.infrastructure.stubs.speech_to_text_stub import SpeechToTextStub

# Arabic appears only as a module-level constant so assertion lines stay pure ASCII: a
# failing assertion prints its operands, and this project is developed on a cp1252 console
# where printing Arabic raises.
_ARABIC_ALARM_QUESTION = "ما سبب إنذار الحريق في الوردية الليلية؟"


def _audio(language_hint: str | None = None) -> AudioRequest:
    return AudioRequest(
        content=b"not really audio",
        metadata=AudioMetadata(
            filename="question.wav", content_type="audio/wav", size_bytes=16
        ),
        language_hint=LanguageHint.from_optional(language_hint),
    )


async def test_no_hint_transcribes_as_english() -> None:
    transcript = await SpeechToTextStub().transcribe(_audio())

    assert transcript.language == "en"
    assert transcript.text == "What caused the fire alarm during the night shift?"


async def test_arabic_hint_returns_the_arabic_question() -> None:
    transcript = await SpeechToTextStub().transcribe(_audio("ar"))

    assert transcript.language == "ar"
    assert transcript.text == _ARABIC_ALARM_QUESTION


async def test_a_language_with_no_canned_transcript_falls_back_to_english() -> None:
    """`supported_languages` is configurable, so the stub can be handed a hint it has no
    canned text for. Falling back beats raising: the stub is a development shortcut and
    should never be the thing that breaks a request.
    """
    transcript = await SpeechToTextStub().transcribe(_audio("fr"))

    assert transcript.language == "en"


async def test_duration_is_not_invented() -> None:
    transcript = await SpeechToTextStub().transcribe(_audio())

    assert transcript.duration_seconds is None
    assert transcript.confidence == 0.9
    assert transcript.model == "faster-whisper-large-v3-turbo-stub"
