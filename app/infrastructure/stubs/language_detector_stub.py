from app.domain.models import DetectedLanguage


class LanguageDetectorStub:
    """Returns a fixed dummy detection result. Stands in for fastText language ID."""

    async def detect(self, text: str) -> DetectedLanguage:
        return DetectedLanguage(code="en", confidence=0.99)
