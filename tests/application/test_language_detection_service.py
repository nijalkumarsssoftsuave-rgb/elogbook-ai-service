import pytest

from app.application.language_detection_service import LanguageDetectionService
from app.domain.exceptions import UnsupportedLanguageError
from app.domain.models import DetectedLanguage


class FakeLanguageDetector:
    def __init__(self, code: str, confidence: float = 0.99) -> None:
        self.code = code
        self.confidence = confidence

    async def detect(self, text: str) -> DetectedLanguage:
        return DetectedLanguage(code=self.code, confidence=self.confidence)


async def test_detect_and_validate_returns_supported_language() -> None:
    service = LanguageDetectionService(
        detector=FakeLanguageDetector(code="en"), supported_languages=["en"]
    )

    result = await service.detect_and_validate("What happened on the last shift?")

    assert result.code == "en"
    assert result.confidence == 0.99


async def test_detect_and_validate_rejects_unsupported_language() -> None:
    service = LanguageDetectionService(
        detector=FakeLanguageDetector(code="fr", confidence=0.95), supported_languages=["en"]
    )

    with pytest.raises(UnsupportedLanguageError) as exc_info:
        await service.detect_and_validate("Bonjour, que s'est-il passé ?")

    assert exc_info.value.language_code == "fr"
    assert exc_info.value.supported_languages == ["en"]
