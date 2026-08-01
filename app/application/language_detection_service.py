from app.application.ports import LanguageDetectorPort
from app.domain.exceptions import UnsupportedLanguageError
from app.domain.models import DetectedLanguage


class LanguageDetectionService:
    """Detects the query language and validates it against the configured allow-list.
    Sits ahead of the rest of the QA pipeline (ES-320): unsupported languages are
    rejected before any cache/guardrail/retrieval work happens.
    """

    def __init__(self, detector: LanguageDetectorPort, supported_languages: list[str]) -> None:
        self._detector = detector
        self._supported_languages = supported_languages

    async def detect_and_validate(self, text: str) -> DetectedLanguage:
        detected = await self._detector.detect(text)
        if detected.code not in self._supported_languages:
            raise UnsupportedLanguageError(detected.code, self._supported_languages)
        return detected
