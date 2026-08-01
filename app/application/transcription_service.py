from pathlib import PurePosixPath

from app.application.ports import SpeechToTextPort
from app.domain.exceptions import (
    AudioRejectionReason,
    InvalidAudioError,
    UnsupportedLanguageError,
)
from app.domain.models import AudioRequest, Transcript

# What a browser or curl sends when it has no idea what the file is. Treated as "not
# declared" rather than as a type in its own right, so we fall back to the extension.
_UNDECLARED_CONTENT_TYPE = "application/octet-stream"

# Only used when the client declares nothing useful. Deliberately small: it maps the
# extensions people actually record with onto the canonical types the allow-list holds,
# and the allow-list stays the single place that decides what is accepted.
_EXTENSION_CONTENT_TYPES: dict[str, str] = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".flac": "audio/flac",
}


def enforce_size_limit(size_bytes: int, max_audio_bytes: int) -> None:
    """Raises if an upload is over the ceiling.

    Lives at module level because the ceiling is enforced twice: cheaply at the transport
    edge from the size the multipart parser reports, so a huge upload is never read into
    memory, and then authoritatively here from the bytes we actually hold. Two call sites,
    one rule, one error message.
    """
    if size_bytes > max_audio_bytes:
        raise InvalidAudioError(
            AudioRejectionReason.FILE_TOO_LARGE,
            f"The uploaded audio file exceeds the {max_audio_bytes} byte limit.",
            {"size_bytes": size_bytes, "max_bytes": max_audio_bytes},
        )


class TranscriptionService:
    """Turns an audio upload into text, refusing input it cannot use.

    This service owns the whole audio policy -- accepted formats, size ceiling, language
    hints, silence -- so the orchestrator above it never sees a content type or a byte
    count. Validating inside the capability mirrors LanguageDetectionService, which
    likewise detects and validates in one call rather than leaving a half-checked value
    for its caller to finish checking.

    A language hint is not a detected language. It tells the speech model what to expect;
    the language the QA pipeline then acts on is detected from the transcript itself. If
    a caller hints "ar" but the audio was English, detection wins downstream.
    """

    def __init__(
        self,
        speech_to_text: SpeechToTextPort,
        allowed_content_types: list[str],
        max_audio_bytes: int,
        supported_languages: list[str],
    ) -> None:
        self._speech_to_text = speech_to_text
        self._allowed_content_types = allowed_content_types
        self._max_audio_bytes = max_audio_bytes
        self._supported_languages = supported_languages

    async def transcribe(self, audio: AudioRequest) -> Transcript:
        self._validate_audio(audio)
        self._validate_language_hint(audio)

        transcript = await self._speech_to_text.transcribe(audio)
        if not transcript.text.strip():
            raise InvalidAudioError(
                AudioRejectionReason.NO_SPEECH_DETECTED,
                "No speech could be detected in the uploaded audio.",
                {"filename": audio.metadata.filename},
            )
        return transcript

    def _validate_audio(self, audio: AudioRequest) -> None:
        metadata = audio.metadata

        if metadata.size_bytes <= 0:
            raise InvalidAudioError(
                AudioRejectionReason.EMPTY_AUDIO,
                "The uploaded audio file is empty.",
                {"filename": metadata.filename},
            )

        enforce_size_limit(metadata.size_bytes, self._max_audio_bytes)

        content_type = self.resolve_content_type(metadata.content_type, metadata.filename)
        if content_type not in self._allowed_content_types:
            raise InvalidAudioError(
                AudioRejectionReason.UNSUPPORTED_CONTENT_TYPE,
                f"Audio content type '{content_type}' is not supported.",
                {
                    "content_type": content_type,
                    "allowed_content_types": self._allowed_content_types,
                },
            )

    def _validate_language_hint(self, audio: AudioRequest) -> None:
        hint = audio.language_hint
        if hint is not None and hint.code not in self._supported_languages:
            raise UnsupportedLanguageError(hint.code, self._supported_languages)

    @staticmethod
    def resolve_content_type(declared: str, filename: str) -> str:
        """Decides what type an upload really is.

        What the client declares wins. Browsers and command-line tools routinely send
        nothing at all, or `application/octet-stream`, for a perfectly ordinary .wav --
        so in that case we fall back to the filename extension rather than rejecting a
        valid recording. Content sniffing from the bytes themselves belongs with the real
        decoder, which has to parse the container anyway.
        """
        normalized = declared.strip().lower()
        if normalized and normalized != _UNDECLARED_CONTENT_TYPE:
            return normalized

        suffix = PurePosixPath(filename).suffix.lower()
        return _EXTENSION_CONTENT_TYPES.get(suffix, normalized)
