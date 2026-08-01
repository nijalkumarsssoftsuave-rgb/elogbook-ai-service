"""Command-line entry point for the evaluation suite.

    uv run python -m tests.eval.run --all
    uv run python -m tests.eval.run --language en --format md
    uv run python -m tests.eval.run --all --write-baseline
    uv run python -m tests.eval.run --language en --print-thresholds

Must be run from the repository root: `python -m` puts the working directory on the path,
and `tests` is a source-tree package rather than something hatchling ships in the wheel.

Exit codes: 0 all thresholds passed, 1 a threshold failed, 2 a case errored or a dataset
could not be loaded.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from tests.eval.loader import (
    BASELINE_DIR,
    REPORTS_DIR,
    dataset_path_for,
    load_dataset,
    thresholds_for,
    validate_against_corpus,
)
from tests.eval.report import to_json, to_markdown, write_report
from tests.eval.runner import EvaluationRunner
from tests.eval.schema import EvaluationReport
from tests.eval.surfaces import build_surfaces

_LANGUAGES = ("en", "ar")


async def _run_language(language: str) -> EvaluationReport:
    surfaces = build_surfaces()
    dataset = load_dataset(dataset_path_for(language))
    validate_against_corpus(dataset, surfaces.corpus_chunk_ids)
    runner = EvaluationRunner(surfaces=surfaces)
    return await runner.run(dataset, thresholds=thresholds_for(language))


def _print_thresholds(report: EvaluationReport) -> None:
    """Emits measured values in the shape thresholds.json expects, so the floors can be
    pasted in and rounded down rather than transcribed by hand.
    """
    retrieval = {
        surface.value: {
            str(aggregate.k): {
                "hit_rate_at_k": {"min": round(aggregate.hit_rate_at_k, 4)},
                "recall_at_k": {"min": round(aggregate.recall_at_k, 4)},
                "mrr_at_k": {"min": round(aggregate.mrr_at_k, 4)},
                "ndcg_at_k": {"min": round(aggregate.ndcg_at_k, 4)},
            }
            for aggregate in aggregates
        }
        for surface, aggregates in report.retrieval.items()
    }
    measured = {
        "retrieval": retrieval,
        "contract": {
            "citation_validity_rate": {"min": round(report.contract.citation_validity_rate, 4)},
            "cited_relevant_rate": {"min": round(report.contract.cited_relevant_rate, 4)},
            "grounded_rate": {"min": round(report.contract.grounded_rate, 4)},
            "false_refusal_rate": {"max": round(report.contract.false_refusal_rate, 4)},
        },
    }
    print(f"--- measured values for '{report.language}' (round floors DOWN before use) ---")
    print(json.dumps(measured, indent=2))


def main(argv: list[str] | None = None) -> int:
    # The console on Windows is cp1252 and cannot encode Arabic; the reports themselves
    # are always written UTF-8 regardless.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Run the EN/AR retrieval evaluation.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--language", choices=_LANGUAGES)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--format", choices=("md", "json"), default="md")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--print-thresholds", action="store_true")
    args = parser.parse_args(argv)

    if not (Path.cwd() / "pyproject.toml").is_file():
        parser.error("run from the repository root: uv run python -m tests.eval.run ...")

    languages = list(_LANGUAGES) if args.all else [args.language]
    exit_code = 0

    for language in languages:
        report = asyncio.run(_run_language(language))

        if args.print_thresholds:
            _print_thresholds(report)
        else:
            print(to_markdown(report) if args.format == "md" else to_json(report))

        stem = f"{report.dataset_name}.v{report.dataset_version}"
        if args.write_baseline:
            json_path, markdown_path = write_report(report.stable(), BASELINE_DIR, stem)
            print(f"[baseline written] {json_path}\n[baseline written] {markdown_path}")
        else:
            write_report(report, REPORTS_DIR, stem)

        if report.errors:
            exit_code = max(exit_code, 2)
        elif report.failed_thresholds:
            exit_code = max(exit_code, 1)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
