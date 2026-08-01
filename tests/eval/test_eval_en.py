"""English accuracy regression suite."""

import pytest

from tests.eval.assertions import assert_no_regression
from tests.eval.loader import thresholds_for, validate_against_corpus
from tests.eval.runner import EvaluationRunner
from tests.eval.schema import EvaluationDataset, EvaluationReport, Surface
from tests.eval.surfaces import EvaluationSurfaces


@pytest.fixture(scope="module")
async def english_report(
    evaluation_runner: EvaluationRunner, english_dataset: EvaluationDataset
) -> EvaluationReport:
    return await evaluation_runner.run(english_dataset, thresholds=thresholds_for("en"))


def test_dataset_gold_ids_exist_in_the_corpus(
    english_dataset: EvaluationDataset, evaluation_surfaces: EvaluationSurfaces
) -> None:
    validate_against_corpus(english_dataset, evaluation_surfaces.corpus_chunk_ids)


async def test_no_case_errored(english_report: EvaluationReport) -> None:
    """Named separately from the threshold check: an errored case must never be mistaken
    for a low score.
    """
    assert english_report.errors == []


async def test_every_case_produced_a_result(
    english_report: EvaluationReport, english_dataset: EvaluationDataset
) -> None:
    scored = {
        result.case_id
        for result in english_report.case_results
        if result.surface is Surface.KEYWORD_ONLY
    }

    assert scored == {case.case_id for case in english_dataset.cases}


async def test_english_accuracy_meets_its_thresholds(
    english_report: EvaluationReport,
) -> None:
    assert_no_regression(english_report)


async def test_answer_quality_is_not_silently_fabricated(
    english_report: EvaluationReport,
) -> None:
    """Guards the decision that no answer-quality score is invented from the stub."""
    assert english_report.answer_quality.computed is False
    assert english_report.answer_quality.faithfulness is None
