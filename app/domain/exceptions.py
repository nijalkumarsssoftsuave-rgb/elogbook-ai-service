from enum import StrEnum
from typing import Any


class AudioRejectionReason(StrEnum):
    UNSUPPORTED_CONTENT_TYPE = "unsupported_content_type"
    EMPTY_AUDIO = "empty_audio"
    FILE_TOO_LARGE = "file_too_large"
    NO_SPEECH_DETECTED = "no_speech_detected"


class InvalidAudioError(Exception):
    """Raised when an upload cannot be turned into a usable question.

    That covers audio we refuse before decoding (wrong format, empty, too large) and
    audio that decoded to nothing at all. Silence belongs here even though the file
    itself was well-formed: from the caller's point of view both mean "this recording
    produced no question to answer".
    """

    def __init__(
        self, reason: AudioRejectionReason, message: str, details: dict[str, Any] | None = None
    ) -> None:
        self.reason = reason
        self.details = details or {}
        super().__init__(message)


class UnsupportedLanguageError(Exception):
    """Raised when the detected query language is not in the configured allow-list."""

    def __init__(self, language_code: str, supported_languages: list[str]) -> None:
        self.language_code = language_code
        self.supported_languages = supported_languages
        super().__init__(
            f"Detected language '{language_code}' is not supported. "
            f"Supported languages: {', '.join(supported_languages)}."
        )
