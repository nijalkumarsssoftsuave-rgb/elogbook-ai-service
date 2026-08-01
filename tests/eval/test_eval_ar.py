"""Arabic accuracy regression suite."""

import pytest

from tests.eval.assertions import assert_no_regression
from tests.eval.loader import thresholds_for, validate_against_corpus
from tests.eval.runner import EvaluationRunner
from tests.eval.schema import EvaluationDataset, EvaluationReport, Surface
from tests.eval.surfaces import EvaluationSurfaces


@pytest.fixture(scope="module")
async def arabic_report(
    evaluation_runner: EvaluationRunner, arabic_dataset: EvaluationDataset
) -> EvaluationReport:
    return await evaluation_runner.run(arabic_dataset, thresholds=thresholds_for("ar"))


def test_dataset_gold_ids_exist_in_the_corpus(
    arabic_dataset: EvaluationDataset, evaluation_surfaces: EvaluationSurfaces
) -> None:
    validate_against_corpus(arabic_dataset, evaluation_surfaces.corpus_chunk_ids)


async def test_no_case_errored(arabic_report: EvaluationReport) -> None:
    """Named separately from the threshold check: an errored case must never be mistaken
    for a low score.
    """
    assert arabic_report.errors == []


async def test_every_case_produced_a_result(
    arabic_report: EvaluationReport, arabic_dataset: EvaluationDataset
) -> None:
    scored = {
        result.case_id
        for result in arabic_report.case_results
        if result.surface is Surface.KEYWORD_ONLY
    }

    assert scored == {case.case_id for case in arabic_dataset.cases}


async def test_arabic_accuracy_meets_its_thresholds(
    arabic_report: EvaluationReport,
) -> None:
    assert_no_regression(arabic_report)


async def test_answer_quality_is_not_silently_fabricated(
    arabic_report: EvaluationReport,
) -> None:
    """Guards the decision that no answer-quality score is invented from the stub."""
    assert arabic_report.answer_quality.computed is False
    assert arabic_report.answer_quality.faithfulness is None


async def test_english_and_arabic_are_scored_independently(
    evaluation_runner: EvaluationRunner,
    english_dataset: EvaluationDataset,
    arabic_dataset: EvaluationDataset,
) -> None:
    """The ticket's actual deliverable, made executable: each language is measured against
    its own corpus and its own thresholds, so neither can mask a regression in the other.
    Deliberately does NOT assert the two scores are close -- they legitimately differ.
    """
    english = await evaluation_runner.run(english_dataset, thresholds=thresholds_for("en"))
    arabic = await evaluation_runner.run(arabic_dataset, thresholds=thresholds_for("ar"))

    assert english.language == "en"
    assert arabic.language == "ar"
    assert english.threshold_outcomes and arabic.threshold_outcomes

    english_cited = {
        chunk_id
        for result in english.case_results
        for chunk_id in result.retrieved_chunk_ids
        if chunk_id.startswith("log-")
    }
    arabic_cited = {
        chunk_id
        for result in arabic.case_results
        for chunk_id in result.retrieved_chunk_ids
        if chunk_id.startswith("log-")
    }

    # Each language retrieved only its own documents: the scores are genuinely separate
    # measurements rather than two views of one mixed result set.
    assert not any(chunk_id.startswith("log-ar-") for chunk_id in english_cited)
    assert all(chunk_id.startswith("log-ar-") for chunk_id in arabic_cited)
