from app.domain.models import DetectedLanguage

# The Arabic block, its two supplements, and both Presentation Forms blocks. Presentation
# forms are included because text pasted from PDFs and older Windows tooling still carries
# them, and they are unambiguously Arabic script.
_ARABIC_RANGES: tuple[tuple[int, int], ...] = (
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)


class ScriptLanguageDetector:
    """Identifies Arabic versus Latin script by codepoint, reporting the script ratio as
    the confidence.

    This is real detection for the English/Arabic pair the service actually supports, but
    it still stands in for fastText's 176-language identification: it classifies *script*,
    not *language*. It cannot tell English from French (both Latin) or Arabic from Persian
    or Urdu (all Arabic script), which is exactly why the reported confidence is capped
    below certainty.
    """

    # Majority script wins. A tie means at least half the letters are Arabic, which we
    # treat as an Arabic query.
    _ARABIC_RATIO_THRESHOLD = 0.5

    # Ceiling on reported confidence. Even entirely Arabic script could be Persian, and
    # entirely Latin script could be French: the script evidence is strong but never
    # conclusive about the language. Refusing to emit 1.0 keeps that honest for any
    # downstream consumer that thresholds on this number.
    _MAX_CONFIDENCE = 0.9

    def __init__(self, default_language: str = "en") -> None:
        # What non-Arabic (and evidence-free) text is reported as. A parameter rather than
        # a hardcoded constant so the fallback is visible at the wiring point.
        self._default_language = default_language

    async def detect(self, text: str) -> DetectedLanguage:
        # Only letters carry script evidence. Digits and punctuation are script-neutral
        # and would dilute the ratio unpredictably; combining marks (Arabic diacritics)
        # are category Mn and not alphabetic, so vowelled text scores the same as plain.
        letters = [character for character in text if character.isalpha()]
        if not letters:
            # No script evidence at all. We report the default language with zero
            # confidence rather than inventing an "und" code, because "und" would fail
            # the supported-language gate and turn a harmless query like "14:32?" into a
            # 400. Zero confidence is the honest signal.
            return DetectedLanguage(code=self._default_language, confidence=0.0)

        arabic_letters = sum(1 for character in letters if self._is_arabic(character))
        arabic_ratio = arabic_letters / len(letters)
        if arabic_ratio >= self._ARABIC_RATIO_THRESHOLD:
            return DetectedLanguage(
                code="ar", confidence=min(arabic_ratio, self._MAX_CONFIDENCE)
            )
        return DetectedLanguage(
            code=self._default_language,
            confidence=min(1 - arabic_ratio, self._MAX_CONFIDENCE),
        )

    @staticmethod
    def _is_arabic(character: str) -> bool:
        code_point = ord(character)
        return any(start <= code_point <= end for start, end in _ARABIC_RANGES)
