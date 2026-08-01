import pytest
from pydantic import ValidationError

from tests.eval.schema import (
    ContractAggregate,
    EvaluationCase,
    EvaluationDataset,
    EvaluationReport,
    ThresholdOutcome,
)


def _case(case_id: str = "en-001", **overrides) -> dict:
    payload = {
        "case_id": case_id,
        "query": "What happened on the last shift?",
        "language": "en",
        "relevant_chunk_ids": ["log-001"],
    }
    payload.update(overrides)
    return payload


def _dataset(**overrides) -> dict:
    payload = {
        "name": "example",
        "language": "en",
        "version": "1",
        "description": "example dataset",
        "cases": [_case()],
    }
    payload.update(overrides)
    return payload


def test_duplicate_case_ids_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate case_id"):
        EvaluationDataset.model_validate(_dataset(cases=[_case("en-001"), _case("en-001")]))


def test_a_case_declaring_a_different_language_is_rejected() -> None:
    with pytest.raises(ValidationError, match="language other than"):
        EvaluationDataset.model_validate(_dataset(cases=[_case(language="ar")]))


def test_an_unknown_key_is_rejected_rather_than_ignored() -> None:
    """A typo in hand-edited JSON must fail loudly. Silently ignoring it would make the
    case score zero and look like a retrieval regression.
    """
    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(_case(relevant_chunks=["log-001"]))


def test_an_empty_dataset_is_rejected() -> None:
    with pytest.raises(ValidationError):
        EvaluationDataset.model_validate(_dataset(cases=[]))


def test_expects_answer_reflects_whether_gold_chunks_were_given() -> None:
    assert EvaluationCase.model_validate(_case()).expects_answer is True
    assert EvaluationCase.model_validate(_case(relevant_chunk_ids=[])).expects_answer is False


def test_cases_are_immutable() -> None:
    case = EvaluationCase.model_validate(_case())

    with pytest.raises(ValidationError):
        case.query = "something else"


def test_dataset_partitions_answerable_and_out_of_corpus_cases() -> None:
    dataset = EvaluationDataset.model_validate(
        _dataset(cases=[_case("en-001"), _case("en-002", relevant_chunk_ids=[])])
    )

    assert [case.case_id for case in dataset.answerable_cases] == ["en-001"]
    assert [case.case_id for case in dataset.out_of_corpus_cases] == ["en-002"]


def test_stable_report_drops_the_timestamp_so_baselines_diff_only_on_numbers() -> None:
    report = EvaluationReport(
        dataset_name="example",
        dataset_version="1",
        language="en",
        top_k=5,
        k_values=[1, 3, 5],
        case_count=1,
        corpus_size=16,
        corpus_languages=["ar", "en"],
        contract=ContractAggregate(
            case_count=1,
            answerable_case_count=1,
            out_of_corpus_case_count=0,
            grounded_rate=1.0,
            refusal_rate=0.0,
            citation_validity_rate=1.0,
            citation_corpus_coverage=1.0,
            cited_relevant_rate=1.0,
            correct_refusal_rate=0.0,
            false_refusal_rate=0.0,
            empty_retrieval_rate=0.0,
        ),
    )

    assert report.generated_at is not None
    assert report.stable().generated_at is None


def test_threshold_outcome_describes_a_failure_with_its_shortfall() -> None:
    outcome = ThresholdOutcome(
        metric_path="keyword_only.k=1.mrr_at_k", floor=0.68, actual=0.5385, passed=False
    )

    description = outcome.describe()

    assert "FAIL" in description
    assert ">= 0.6800" in description
    assert "0.5385" in description
    assert "-0.1415" in description
