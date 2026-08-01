import pytest

from app.api.dependencies import build_speech_to_text_port
from app.core.config import Settings
from app.infrastructure.model_serving.speech.faster_whisper_adapter import FasterWhisperAdapter
from app.infrastructure.stubs.speech_to_text_stub import SpeechToTextStub
from tests.conftest import TEST_ALGORITHM, TEST_SECRET


def _settings(**overrides: object) -> Settings:
    return Settings(
        service_jwt_secret=TEST_SECRET,
        service_jwt_algorithm=TEST_ALGORITHM,
        **overrides,  # type: ignore[arg-type]
    )


def test_the_stub_is_the_default_backend() -> None:
    """A plain checkout must run without the multi-gigabyte model stack installed."""
    assert isinstance(build_speech_to_text_port(_settings()), SpeechToTextStub)


def test_the_faster_whisper_backend_is_selectable() -> None:
    port = build_speech_to_text_port(
        _settings(stt_backend="faster_whisper", stt_model="tiny", stt_device="cpu")
    )

    assert isinstance(port, FasterWhisperAdapter)


def test_selecting_faster_whisper_does_not_load_a_model() -> None:
    """Wiring runs at import/startup time. If constructing the adapter loaded the model,
    a machine without it -- or without the `stt` extra -- could not start the service at
    all, including the endpoints that have nothing to do with speech.
    """
    port = build_speech_to_text_port(
        _settings(stt_backend="faster_whisper", stt_model="does-not-exist")
    )

    assert isinstance(port, FasterWhisperAdapter)
    # Reaching into private state is the point here: "nothing was loaded" has no public
    # surface, and asserting it any less directly would not actually prove it.
    assert port._model is None


def test_an_unknown_backend_is_rejected_rather_than_silently_stubbed() -> None:
    with pytest.raises(ValueError, match="Unknown STT_BACKEND"):
        build_speech_to_text_port(_settings(stt_backend="whispr"))
