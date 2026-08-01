import asyncio
import threading
import time
from collections.abc import Iterable

import pytest

from app.domain.exceptions import TranscriptionFailedError, TranscriptionFailureReason
from app.domain.models import AudioMetadata, AudioRequest, LanguageHint
from app.infrastructure.model_serving.speech.faster_whisper_adapter import (
    FasterWhisperAdapter,
    Segment,
    TranscriptionInfo,
)


class FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeInfo:
    def __init__(
        self, language: str = "en", language_probability: float = 0.98, duration: float = 3.5
    ) -> None:
        self.language = language
        self.language_probability = language_probability
        self.duration = duration


class FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel.

    Like the real thing, it hands back a *generator* of segments rather than a list, so
    tests exercise the same laziness the adapter has to cope with. The generator records
    which thread drained it, which is what makes the off-the-event-loop guarantee provable
    rather than merely intended.
    """

    def __init__(
        self,
        texts: list[str] | None = None,
        info: FakeInfo | None = None,
        sleep_seconds: float = 0.0,
    ) -> None:
        self._texts = texts if texts is not None else [" Hello there.", " How are you?"]
        self._info = info or FakeInfo()
        self._sleep_seconds = sleep_seconds
        self.calls: list[dict[str, object]] = []
        self.drained_on_thread: int | None = None

    def transcribe(
        self, audio: object, language: str | None = None, **kwargs: object
    ) -> tuple[Iterable[Segment], TranscriptionInfo]:
        self.calls.append({"language": language, "audio_type": type(audio).__name__})

        def segments() -> Iterable[Segment]:
            # The real engine does its work here, on iteration -- not in transcribe().
            self.drained_on_thread = threading.get_ident()
            if self._sleep_seconds:
                time.sleep(self._sleep_seconds)
            for text in self._texts:
                yield FakeSegment(text)

        return segments(), self._info


class CountingFactory:
    def __init__(self, model: FakeWhisperModel | None = None, error: Exception | None = None):
        self.model = model or FakeWhisperModel()
        self.error = error
        self.calls = 0

    def __call__(self) -> FakeWhisperModel:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.model


def _audio(content: bytes = b"fake-wav-bytes", language_hint: str | None = None) -> AudioRequest:
    return AudioRequest(
        content=content,
        metadata=AudioMetadata(
            filename="question.wav", content_type="audio/wav", size_bytes=len(content)
        ),
        language_hint=LanguageHint.from_optional(language_hint),
    )


def _build(factory: CountingFactory, **kwargs: object) -> FasterWhisperAdapter:
    return FasterWhisperAdapter(
        model_name="large-v3-turbo",
        model_factory=factory,
        **kwargs,  # type: ignore[arg-type]
    )


# --- model loading ------------------------------------------------------------------


def test_the_model_is_not_loaded_when_the_adapter_is_constructed() -> None:
    """Construction happens during DI wiring, before any request. Loading there would
    make a missing model break the whole service instead of one endpoint.
    """
    factory = CountingFactory()

    _build(factory)

    assert factory.calls == 0


async def test_the_model_is_loaded_once_across_concurrent_transcriptions() -> None:
    factory = CountingFactory()
    adapter = _build(factory)

    await asyncio.gather(*(adapter.transcribe(_audio()) for _ in range(5)))

    assert factory.calls == 1


async def test_warmup_loads_the_model_up_front() -> None:
    factory = CountingFactory()
    adapter = _build(factory)

    await adapter.warmup()

    assert factory.calls == 1


async def test_a_load_failure_becomes_a_model_unavailable_error() -> None:
    factory = CountingFactory(error=RuntimeError("no such model directory"))
    adapter = _build(factory)

    with pytest.raises(TranscriptionFailedError) as excinfo:
        await adapter.transcribe(_audio())

    assert excinfo.value.reason is TranscriptionFailureReason.MODEL_UNAVAILABLE
    assert excinfo.value.details["model"] == "large-v3-turbo"
    # The underlying cause survives in the message, so an operator can act on it.
    assert "no such model directory" in str(excinfo.value)


async def test_a_failed_load_is_retried_rather_than_remembered() -> None:
    """A missing network mount is usually temporary. Caching the failure would keep the
    endpoint down long after the cause was fixed.
    """
    factory = CountingFactory(error=RuntimeError("mount not ready"))
    adapter = _build(factory)

    with pytest.raises(TranscriptionFailedError):
        await adapter.transcribe(_audio())

    factory.error = None
    transcript = await adapter.transcribe(_audio())

    assert factory.calls == 2
    assert transcript.text == "Hello there. How are you?"


# --- the compute actually leaves the event loop ---------------------------------------


async def test_the_segment_generator_is_drained_off_the_event_loop() -> None:
    """faster-whisper does its work when the segments are iterated, not when transcribe()
    returns. If the join happened back on the event loop, one long recording would stall
    every other request -- so this pins where the draining happens.
    """
    model = FakeWhisperModel()
    adapter = _build(CountingFactory(model))

    await adapter.transcribe(_audio())

    assert model.drained_on_thread is not None
    assert model.drained_on_thread != threading.get_ident()


# --- language hint ---------------------------------------------------------------------


async def test_a_language_hint_is_passed_to_the_engine() -> None:
    model = FakeWhisperModel(info=FakeInfo(language="ar", language_probability=0.95))
    adapter = _build(CountingFactory(model))

    transcript = await adapter.transcribe(_audio(language_hint="ar"))

    assert model.calls[0]["language"] == "ar"
    assert transcript.language == "ar"


async def test_without_a_hint_the_engine_detects_the_language_itself() -> None:
    model = FakeWhisperModel(info=FakeInfo(language="ar", language_probability=0.87))
    adapter = _build(CountingFactory(model))

    transcript = await adapter.transcribe(_audio())

    assert model.calls[0]["language"] is None
    # The detected language wins -- we report what the engine heard, not what was assumed.
    assert transcript.language == "ar"
    assert transcript.confidence == 0.87


# --- mapping onto the domain model ------------------------------------------------------


async def test_transcription_info_is_mapped_onto_the_transcript() -> None:
    model = FakeWhisperModel(
        texts=[" Fire alarm", " during the night shift."],
        info=FakeInfo(language="en", language_probability=0.99, duration=7.25),
    )
    adapter = _build(CountingFactory(model))

    transcript = await adapter.transcribe(_audio())

    assert transcript.text == "Fire alarm during the night shift."
    assert transcript.language == "en"
    assert transcript.confidence == 0.99
    # ES-323's stub had to leave this None; a real decoder finally knows the answer.
    assert transcript.duration_seconds == 7.25
    assert transcript.model == "large-v3-turbo"


async def test_the_upload_is_streamed_to_the_engine_without_touching_disk() -> None:
    model = FakeWhisperModel()
    adapter = _build(CountingFactory(model))

    await adapter.transcribe(_audio())

    assert model.calls[0]["audio_type"] == "BytesIO"


# --- timeout ----------------------------------------------------------------------------


async def test_a_transcription_past_the_deadline_raises_a_timeout() -> None:
    adapter = _build(
        CountingFactory(FakeWhisperModel(sleep_seconds=1.0)), timeout_seconds=0.05
    )

    with pytest.raises(TranscriptionFailedError) as excinfo:
        await adapter.transcribe(_audio())

    assert excinfo.value.reason is TranscriptionFailureReason.TIMEOUT
    assert excinfo.value.details["timeout_seconds"] == 0.05


async def test_the_caller_is_released_at_the_deadline_not_when_the_engine_finishes() -> None:
    """The point of the timeout is that the request comes back quickly. The worker thread
    does keep running -- asyncio cannot kill it -- which is recorded in the adapter's
    docstring, but the caller must not be held hostage to it.
    """
    adapter = _build(
        CountingFactory(FakeWhisperModel(sleep_seconds=1.0)), timeout_seconds=0.05
    )

    started = time.monotonic()
    with pytest.raises(TranscriptionFailedError):
        await adapter.transcribe(_audio())
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
