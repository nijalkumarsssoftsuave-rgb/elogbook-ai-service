"""Runner mechanics, exercised against a tiny hand-built corpus rather than the real
datasets, so behaviour is tested independently of the real accuracy numbers.
"""

import pytest

from app.api import dependencies
from app.infrastructure.retrieval.fixture_corpus import FixtureDocument
from tests.eval.runner import EvaluationRunner
from tests.eval.schema import EvaluationDataset, Surface
from tests.eval.surfaces import EvaluationSurfaces, build_surfaces

_TINY_CORPUS = [
    FixtureDocument("t-001", "doc-t-001", "conveyor belt inspection completed", {}, "en"),
    FixtureDocument("t-002", "doc-t-002", "forklift collision near the aisle", {}, "en"),
]

_TINY_DATASET = EvaluationDataset.model_validate(
    {
        "name": "tiny",
        "language": "en",
        "version": "1",
        "description": "runner mechanics only",
        "cases": [
            {
                "case_id": "t-a",
                "query": "conveyor belt inspection",
                "language": "en",
                "relevant_chunk_ids": ["t-001"],
            },
            {
                "case_id": "t-b",
                "query": "forklift collision",
                "language": "en",
                "relevant_chunk_ids": ["t-002"],
            },
            {
                "case_id": "t-c",
                "query": "quarterly revenue forecast",
                "language": "en",
                "relevant_chunk_ids": [],
            },
        ],
    }
)


@pytest.fixture
def tiny_runner() -> EvaluationRunner:
    return EvaluationRunner(surfaces=build_surfaces(corpus=_TINY_CORPUS))


async def test_every_case_is_scored_on_each_ranked_surface(
    tiny_runner: EvaluationRunner,
) -> None:
    report = await tiny_runner.run(_TINY_DATASET)

    for surface in (Surface.KEYWORD_ONLY, Surface.RETRIEVAL_RAW):
        scored = {r.case_id for r in report.case_results if r.surface is surface}
        assert scored == {"t-a", "t-b", "t-c"}
    assert len(report.contract_results) == 3


async def test_out_of_corpus_cases_are_excluded_from_retrieval_aggregates(
    tiny_runner: EvaluationRunner,
) -> None:
    """Scoring retrieval against no gold set is undefined, so those cases must not drag
    the averages down. They are counted through the contract metrics instead.
    """
    report = await tiny_runner.run(_TINY_DATASET)

    out_of_corpus = next(r for r in report.case_results if r.case_id == "t-c")
    assert out_of_corpus.metrics == []
    assert all(
        aggregate.case_count == 2
        for aggregate in report.retrieval[Surface.KEYWORD_ONLY]
    )
    assert report.contract.out_of_corpus_case_count == 1


async def test_two_runs_produce_identical_reports_apart_from_the_timestamp(
    tiny_runner: EvaluationRunner,
) -> None:
    first = await tiny_runner.run(_TINY_DATASET)
    second = await tiny_runner.run(_TINY_DATASET)

    assert first.stable().model_dump() == second.stable().model_dump()


async def test_a_failing_case_is_recorded_without_aborting_the_run(
    tiny_runner: EvaluationRunner,
) -> None:
    """A report with one named error and the remaining numbers is far more useful than a
    traceback and nothing.
    """

    async def explode(*args, **kwargs):
        raise RuntimeError("simulated retrieval failure")

    tiny_runner._surfaces.keyword_retriever.search = explode  # type: ignore[method-assign]

    report = await tiny_runner.run(_TINY_DATASET)

    # Every surface reads through the keyword retriever, so all three fail for all three
    # cases. The point is that the run still completes and names each failure.
    assert len(report.errors) == 9
    assert all("simulated retrieval failure" in error for error in report.errors)
    assert all(error.startswith(("t-a/", "t-b/", "t-c/")) for error in report.errors)

    # The report is still well-formed: every case is present, carrying its own error.
    assert len(report.contract_results) == 3
    assert all(result.error for result in report.contract_results)
    assert all(result.retrieved_chunk_ids == [] for result in report.case_results)


def test_surfaces_match_production_wiring(evaluation_surfaces: EvaluationSurfaces) -> None:
    """The evaluation builds its own object graph, because the DI composing functions
    return `Depends` placeholders outside a request. This guards the cost of that choice:
    if production swaps a stub for a real adapter, the evaluation must not keep silently
    scoring the old one.
    """
    retrieval = evaluation_surfaces.retrieval_service

    assert type(retrieval._embedding) is type(dependencies.get_embedding_port())
    assert type(retrieval._vector_store) is type(dependencies.get_vector_store_port())
    assert type(retrieval._keyword_retriever) is type(dependencies.get_keyword_retriever_port())
    assert type(retrieval._reranker) is type(dependencies.get_reranker_port())
