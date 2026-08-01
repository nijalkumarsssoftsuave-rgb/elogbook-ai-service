import pytest

from app.infrastructure.language.script_language_detector import ScriptLanguageDetector

_ARABIC_QUERY = "ما سبب إنذار الحريق في الوردية الليلية؟"
_ENGLISH_QUERY = "What caused the fire alarm during the night shift?"


async def test_english_text_is_detected_as_english() -> None:
    result = await ScriptLanguageDetector().detect(_ENGLISH_QUERY)

    assert result.code == "en"


async def test_arabic_text_is_detected_as_arabic() -> None:
    result = await ScriptLanguageDetector().detect(_ARABIC_QUERY)

    assert result.code == "ar"


async def test_confidence_is_capped_below_certainty() -> None:
    """Script evidence is never conclusive about language, so the detector must not
    claim certainty even for unambiguous input.
    """
    english = await ScriptLanguageDetector().detect(_ENGLISH_QUERY)
    arabic = await ScriptLanguageDetector().detect(_ARABIC_QUERY)

    assert english.confidence == pytest.approx(0.9)
    assert arabic.confidence == pytest.approx(0.9)


async def test_mixed_script_reports_an_honest_partial_confidence() -> None:
    """An Arabic sentence containing a Latin acronym is still Arabic, but the detector
    should not pretend the evidence was unanimous.
    """
    result = await ScriptLanguageDetector().detect("وصل الفريق لفحص HVAC اليوم")

    assert result.code == "ar"
    assert 0.5 <= result.confidence < 0.9


async def test_diacritics_do_not_affect_the_script_ratio() -> None:
    """Arabic diacritics are combining marks, not letters, so vowelled and plain text
    must score identically.
    """
    plain = await ScriptLanguageDetector().detect("انذار الحريق")
    vowelled = await ScriptLanguageDetector().detect("إنْذَار الحَرِيق")

    assert plain.code == vowelled.code == "ar"
    assert plain.confidence == pytest.approx(vowelled.confidence)


@pytest.mark.parametrize("text", ["", "   ", "12345", "!!! ???", "14:32"])
async def test_text_without_letters_returns_the_default_with_zero_confidence(
    text: str,
) -> None:
    """No script evidence. Reported as the default language rather than an unsupported
    code, so a harmless numeric query is not rejected outright.
    """
    result = await ScriptLanguageDetector().detect(text)

    assert result.code == "en"
    assert result.confidence == 0.0


async def test_default_language_is_configurable() -> None:
    result = await ScriptLanguageDetector(default_language="fr").detect(_ENGLISH_QUERY)

    assert result.code == "fr"


async def test_arabic_presentation_forms_are_recognised_as_arabic() -> None:
    """Text pasted from PDFs often carries presentation forms rather than base letters."""
    result = await ScriptLanguageDetector().detect("ﻻ ﻳﻮﺟﺪ")

    assert result.code == "ar"


async def test_latin_script_non_english_is_classified_as_english() -> None:
    """A known limitation, pinned deliberately: this detector reads script, not language,
    so every Latin-script language reports as the default. Only a real language-ID model
    (fastText, per CLAUDE.md) distinguishes them -- at which point this test should be
    updated to expect "fr".
    """
    result = await ScriptLanguageDetector().detect("Bonjour, que s'est-il passe hier ?")

    assert result.code == "en"
