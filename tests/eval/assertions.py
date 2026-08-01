"""The regression assertion.

A bare `assert actual >= floor` is unusable in a suite someone else will debug months
from now: it says nothing about which metric moved, by how much, or what to do next.
"""

from tests.eval.schema import EvaluationReport, Surface


def _cases_with_no_hit(report: EvaluationReport) -> list[str]:
    return sorted(
        {
            result.case_id
            for result in report.case_results
            if result.surface is Surface.KEYWORD_ONLY
            and any(metric.k == 5 and metric.hit_rate == 0.0 for metric in result.metrics)
        }
    )


def assert_no_regression(report: EvaluationReport) -> None:
    failures = report.failed_thresholds
    if not failures:
        return

    passed = len(report.threshold_outcomes) - len(failures)
    lines = [
        "",
        (f"Evaluation regression: {report.dataset_name} v{report.dataset_version} "
        f"(language={report.language}, {report.case_count} cases, "
        f"corpus={report.corpus_size} chunks)"),
        "",
        *(f"  {outcome.describe()}" for outcome in failures),
        "",
        f"  {passed} of {len(report.threshold_outcomes)} thresholds passed.",
    ]

    missed = _cases_with_no_hit(report)
    if missed:
        lines += ["", f"  Cases retrieving no relevant chunk at k=5: {', '.join(missed)}"]

    stem = f"{report.dataset_name}.v{report.dataset_version}"
    lines += [
        "",
        (f"  Inspect:  uv run python -m tests.eval.run --language {report.language} "
        "--format md"),
        f"  Then diff against tests/eval/baseline/{stem}.report.md",
        "  Only regenerate the baseline once you have confirmed the drop is intended.",
        "",
    ]
    raise AssertionError("\n".join(lines))
