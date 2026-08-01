"""Tests for the shared tokenizer.

Arabic is written as \\uXXXX escapes with a comment naming each character. These tests
are *about* specific codepoints, so escapes state the intent precisely, keep the source
pure ASCII (this project is developed on a cp1252 console, where pytest printing Arabic
in a failure message would itself raise), and stop bidirectional reordering from making
the diff unreadable.
"""

import re

import pytest

from app.infrastructure.retrieval.fixture_corpus import DEFAULT_FIXTURE_CORPUS
from app.infrastructure.retrieval.text_normalization import normalize, tokenize

# The pre-ES-322 tokenizer, kept as an oracle. English must tokenize identically forever;
# pinning it here is stronger than having checked once during the change.
_LEGACY_TOKEN_RE = re.compile(r"[a-z0-9]+")

_ENGLISH_QUERIES = [
    "fire alarm triggered during the night shift",
    "forklift near miss in aisle 7",
    "shift maintenance safety inspection report",
    "zzzz qqqq nonexistent gibberish",
    "shift forklift",
    "Boiler 3 pressure 145 psi at 09:15",
    "What happened at the loading dock?",
]

# "AL-HAREEQ" fully vowelled: ALEF LAM SUKUN HAH FATHA REH KASRA YEH QAF DAMMA.
_VOWELLED = "الْحَرِيقُ"
_UNVOWELLED = "الحريق"


@pytest.mark.parametrize("document", DEFAULT_FIXTURE_CORPUS)
def test_english_documents_tokenize_exactly_as_the_legacy_tokenizer_did(document) -> None:
    if getattr(document, "language", "en") != "en":
        pytest.skip("the legacy tokenizer only ever handled English")
    assert tokenize(document.text) == _LEGACY_TOKEN_RE.findall(document.text.lower())


@pytest.mark.parametrize("query", _ENGLISH_QUERIES)
def test_english_queries_tokenize_exactly_as_the_legacy_tokenizer_did(query: str) -> None:
    assert tokenize(query) == _LEGACY_TOKEN_RE.findall(query.lower())


def test_arabic_text_produces_tokens_at_all() -> None:
    """The original blocker: the ASCII tokenizer returned [] for any Arabic input."""
    assert tokenize(_UNVOWELLED)


def test_diacritics_do_not_split_a_word_into_several_tokens() -> None:
    """Tashkeel are category Mn, which `\\w` does not match, so an unstripped vowelled
    word shatters into one token per letter cluster.
    """
    assert len(tokenize(_VOWELLED)) == 1
    assert tokenize(_VOWELLED) == tokenize(_UNVOWELLED)


def test_tatweel_is_removed_from_inside_a_word() -> None:
    # HAH TATWEEL TATWEEL REH YEH QAF vs HAH REH YEH QAF
    assert tokenize("حــريق") == tokenize("حريق")


@pytest.mark.parametrize("variant", ["أ", "إ", "آ", "ٱ"])
def test_hamza_carrier_variants_fold_to_plain_alef(variant: str) -> None:
    assert normalize(variant) == "ا"


def test_alef_maksura_folds_to_yeh() -> None:
    assert normalize("ى") == "ي"


def test_ta_marbuta_folds_to_heh() -> None:
    assert normalize("ة") == "ه"


def test_arabic_presentation_forms_fold_to_base_letters() -> None:
    """Pins the NFKC step: text pasted from PDFs still carries presentation forms."""
    # LAM WITH ALEF ligature -> LAM + ALEF
    assert normalize("ﻻ") == "لا"


def test_underscore_splits_tokens() -> None:
    """Pins the choice of `[^\\W_]+` over `\\w+`, which would fuse this into one token."""
    assert tokenize("2024_ok") == ["2024", "ok"]


def test_definite_article_is_deliberately_not_stripped() -> None:
    """A documentation test. Normalization is orthographic; the "al-" prefix is
    morphological and needs a real stemmer, which is out of scope. This pins the scope
    boundary so a future stemming ticket has to consciously delete it.
    """
    assert tokenize("الحريق") != tokenize("حريق")


@pytest.mark.parametrize("text", ["", "   ", "!!! ???", "-- , --"])
def test_text_without_word_characters_produces_no_tokens(text: str) -> None:
    assert tokenize(text) == []
