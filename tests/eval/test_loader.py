import pytest

from app.infrastructure.retrieval.fixture_corpus import DEFAULT_FIXTURE_CORPUS
from tests.eval.loader import (
    ARABIC_DATASET_PATH,
    ENGLISH_DATASET_PATH,
    ThresholdBound,
    load_dataset,
    load_thresholds,
    thresholds_for,
    validate_against_corpus,
)

_CORPUS_IDS = {document.chunk_id for document in DEFAULT_FIXTURE_CORPUS}
_DATASET_PATHS = [ENGLISH_DATASET_PATH, ARABIC_DATASET_PATH]


@pytest.mark.parametrize("path", _DATASET_PATHS)
def test_dataset_parses(path) -> None:
    assert load_dataset(path).cases


@pytest.mark.parametrize("path", _DATASET_PATHS)
def test_dataset_is_utf8_without_a_byte_order_mark(path) -> None:
    """A BOM makes json.loads fail with an unhelpful "Expecting value: line 1 column 1",
    and it is the most likely accident from a Windows editor.
    """
    raw = path.read_bytes()

    assert not raw.startswith(b"\xef\xbb\xbf")
    raw.decode("utf-8")


@pytest.mark.parametrize("path", _DATASET_PATHS)
def test_dataset_gold_ids_all_exist_in_the_corpus(path) -> None:
    """A typo'd gold id would score 0.0 for that case and look exactly like a retrieval
    regression rather than a dataset bug.
    """
    validate_against_corpus(load_dataset(path), _CORPUS_IDS)


def test_arabic_dataset_actually_contains_arabic() -> None:
    """Guards against a transliterated placeholder being committed unnoticed."""
    dataset = load_dataset(ARABIC_DATASET_PATH)

    assert any(any(ord(char) > 0x0590 for char in case.query) for case in dataset.cases)


def test_arabic_dataset_parallels_the_english_one_case_for_case() -> None:
    english = load_dataset(ENGLISH_DATASET_PATH)
    arabic = load_dataset(ARABIC_DATASET_PATH)
    english_ids = {case.case_id for case in english.cases}

    assert len(arabic.cases) == len(english.cases)
    for case in arabic.cases:
        assert case.parallel_case_id in english_ids, case.case_id


def test_validate_against_corpus_names_the_unknown_ids() -> None:
    dataset = load_dataset(ENGLISH_DATASET_PATH)

    with pytest.raises(ValueError, match="log-001"):
        validate_against_corpus(dataset, {"something-else"})


def test_thresholds_parse_and_cover_both_languages() -> None:
    thresholds = load_thresholds()

    assert set(thresholds.languages) >= {"en", "ar"}


@pytest.mark.parametrize("language", ["en", "ar"])
def test_language_thresholds_are_populated(language: str) -> None:
    """Placeholder floors of 0.0 would pass silently forever. Once a baseline exists,
    every configured floor should be a real measured number.
    """
    thresholds = thresholds_for(language)

    assert thresholds.retrieval, f"no retrieval thresholds configured for {language}"
    bounds = [
        bound
        for by_k in thresholds.retrieval.values()
        for by_metric in by_k.values()
        for bound in by_metric.values()
    ]
    assert bounds
    assert all(bound.min is None or bound.min > 0.0 for bound in bounds)


def test_a_threshold_needs_exactly_one_direction() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        ThresholdBound()

    with pytest.raises(ValueError, match="exactly one"):
        ThresholdBound(min=0.5, max=0.9)
