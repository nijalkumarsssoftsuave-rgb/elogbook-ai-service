import asyncio
import io
from collections.abc import Callable, Iterable
from typing import Any, Protocol

from app.domain.exceptions import TranscriptionFailedError, TranscriptionFailureReason
from app.domain.models import AudioRequest, Transcript


class Segment(Protocol):
    """The one field we read off a faster-whisper segment."""

    text: str


class TranscriptionInfo(Protocol):
    """The three fields we read off faster-whisper's TranscriptionInfo."""

    language: str
    language_probability: float
    duration: float


class SpeechModel(Protocol):
    """The slice of faster_whisper.WhisperModel this adapter actually uses.

    Declared locally so mypy can check our call site even though `faster-whisper` is an
    optional extra and is normally not installed, and so tests can substitute a fake model
    without importing the real package.
    """

    def transcribe(
        self, audio: Any, language: str | None = None, **kwargs: Any
    ) -> tuple[Iterable[Segment], TranscriptionInfo]: ...


class FasterWhisperAdapter:
    """Transcribes audio with faster-whisper (CTranslate2), the SpeechToTextPort
    implementation that replaces SpeechToTextStub in a real deployment.

    Three things drive the shape of this class:

    * **The model loads lazily, once.** Loading takes seconds and, on a cold machine,
      downloads gigabytes. Doing it in __init__ would mean the whole service fails to
      start because one endpoint's model is missing. The lock makes concurrent first
      requests load once instead of racing; a *failed* load is deliberately not
      remembered, so a transient mount or network problem does not poison the adapter for
      the rest of the process's life.

    * **Transcription runs in a worker thread.** WhisperModel.transcribe is synchronous
      and compute-bound, so leaving it on the event loop would stall every other request
      for the duration of the audio.

    * **The segment generator is drained inside that same thread.** faster-whisper returns
      segments lazily -- `transcribe()` itself returns almost immediately and the real work
      happens on iteration. Joining the segments anywhere else would put the compute back
      on the event loop and leave the timeout guarding a call that does nothing.

    Known limitation: asyncio.wait_for cancels the *await*, not the thread. On a timeout
    the caller gets an answer promptly, but the worker keeps running to completion in the
    background. Actually reclaiming that CPU needs a process pool or a deadline enforced by
    the inference server itself, which is a separate piece of work.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        compute_type: str = "default",
        timeout_seconds: float = 120.0,
        local_files_only: bool = False,
        download_root: str | None = None,
        model_factory: Callable[[], SpeechModel] | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._timeout_seconds = timeout_seconds
        self._local_files_only = local_files_only
        self._download_root = download_root
        # Injectable so the loading, timeout and mapping behaviour can be tested without
        # downloading a multi-gigabyte model.
        self._model_factory = model_factory or self._load_model
        self._model: SpeechModel | None = None
        self._load_lock = asyncio.Lock()

    async def transcribe(self, audio: AudioRequest) -> Transcript:
        model = await self._ensure_model()
        language = audio.language_hint.code if audio.language_hint else None

        try:
            text, info = await asyncio.wait_for(
                asyncio.to_thread(self._transcribe_blocking, model, audio.content, language),
                timeout=self._timeout_seconds,
            )
        except TimeoutError as error:
            raise TranscriptionFailedError(
                TranscriptionFailureReason.TIMEOUT,
                f"Transcription exceeded the {self._timeout_seconds} second limit.",
                {
                    "timeout_seconds": self._timeout_seconds,
                    "size_bytes": audio.metadata.size_bytes,
                },
            ) from error

        return Transcript(
            text=text,
            # The detected language, not the hint: when no hint is given whisper works it
            # out, and when a hint is given it is echoed back here anyway.
            language=info.language,
            confidence=info.language_probability,
            duration_seconds=info.duration,
            model=self._model_name,
        )

    async def warmup(self) -> None:
        """Loads the model ahead of the first request.

        Lets a deployment pay the load cost at startup, deliberately, instead of making one
        unlucky caller wait for it.
        """
        await self._ensure_model()

    def _transcribe_blocking(
        self, model: SpeechModel, content: bytes, language: str | None
    ) -> tuple[str, TranscriptionInfo]:
        """Runs entirely inside the worker thread, generator included.

        faster-whisper decodes a file-like object directly via PyAV, so the upload never
        touches the filesystem on its way to the model.
        """
        segments, info = model.transcribe(io.BytesIO(content), language=language)
        text = "".join(segment.text for segment in segments).strip()
        return text, info

    async def _ensure_model(self) -> SpeechModel:
        if self._model is not None:
            return self._model

        async with self._load_lock:
            # Re-checked under the lock: several requests can queue here while the first
            # one loads, and they should reuse its result rather than load again.
            if self._model is None:
                self._model = await asyncio.to_thread(self._call_model_factory)
            return self._model

    def _call_model_factory(self) -> SpeechModel:
        try:
            return self._model_factory()
        except TranscriptionFailedError:
            # Already carries a reason and a usable message (a missing extra, say) -- do
            # not bury it inside a second one.
            raise
        # Deliberately broad: a missing file, a bad device name and a CUDA failure all
        # reach the caller as the same "the model would not load" answer.
        except Exception as error:
            # Nothing is assigned to self._model here, so the next request retries rather
            # than inheriting a permanent failure.
            raise TranscriptionFailedError(
                TranscriptionFailureReason.MODEL_UNAVAILABLE,
                f"Speech model '{self._model_name}' could not be loaded: {error}",
                {"model": self._model_name, "device": self._device},
            ) from error

    def _load_model(self) -> SpeechModel:
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:  # pragma: no cover - depends on the install profile
            raise TranscriptionFailedError(
                TranscriptionFailureReason.MODEL_UNAVAILABLE,
                "faster-whisper is not installed. Run `uv sync --extra stt` on a machine "
                "that serves the speech model.",
                {"model": self._model_name},
            ) from error

        model: SpeechModel = WhisperModel(
            self._model_name,
            device=self._device,
            compute_type=self._compute_type,
            local_files_only=self._local_files_only,
            download_root=self._download_root,
        )
        return model
