"""Serialization of an EvaluationReport to JSON and Markdown.

Both are written UTF-8 with `ensure_ascii=False`, so Arabic appears as Arabic rather than
escape sequences. Per-case tables print case and chunk ids rather than query text: that
keeps the Markdown readable in a mixed left-to-right / right-to-left diff, and avoids the
cp1252 console entirely.
"""

import json
from pathlib import Path

from tests.eval.schema import EvaluationReport, Surface

_SURFACE_BLURB = {
    Surface.KEYWORD_ONLY: "BM25 keyword search -- the one real retrieval component",
    Surface.RETRIEVAL_RAW: "RetrievalService: RRF over dense + keyword, then reranking",
}


def to_json(report: EvaluationReport) -> str:
    return json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2)


def to_markdown(report: EvaluationReport) -> str:
    lines: list[str] = [
        (f"# Evaluation report — {report.dataset_name} v{report.dataset_version} "
        f"({report.language})"),
        "",
        (f"Corpus: {report.corpus_size} chunks "
        f"({', '.join(report.corpus_languages)}) · "
        f"{report.case_count} cases "
        f"({report.contract.answerable_case_count} answerable, "
        f"{report.contract.out_of_corpus_case_count} out-of-corpus) · "
        f"top_k={report.top_k}"),
        "",
        "## What this measures",
        "",
        "Retrieval accuracy against known-correct chunk ids, and the pipeline's",
        "structural contract. **Answer quality is not measured** — the LLM, embeddings,",
        "dense vector search and the reranker are stubs.",
        "",
    ]

    for surface, aggregates in report.retrieval.items():
        lines += [
            f"## Retrieval — {surface.value}",
            "",
            f"_{_SURFACE_BLURB.get(surface, '')}_",
            "",
            "| k | hit@k | recall@k | precision@k | MRR@k | nDCG@k |",
            "|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {a.k} | {a.hit_rate_at_k:.4f} | {a.recall_at_k:.4f} | "
            f"{a.precision_at_k:.4f} | {a.mrr_at_k:.4f} | {a.ndcg_at_k:.4f} |"
            for a in aggregates
        ]
        if surface is Surface.RETRIEVAL_RAW:
            lines += [
                "",
                "> MRR on this surface is capped and precision@1 pinned at 0: the dense",
                "> stub returns one fixed chunk that ties the top keyword hit on RRF score",
                "> and wins the tie by insertion order, so rank 1 is always the stub.",
                "> Compare against `keyword_only` for the un-polluted retrieval number.",
            ]
        lines.append("")

    contract = report.contract
    lines += [
        "## Pipeline contract",
        "",
        "| metric | value | note |",
        "|---|---|---|",
        (f"| citation_validity_rate | {contract.citation_validity_rate:.4f} | "
        "every citation points at a retrieved chunk |"),
        (f"| citation_corpus_coverage | {contract.citation_corpus_coverage:.4f} | "
        "below 1.0 because the dense stub's chunk is in no corpus |"),
        (f"| cited_relevant_rate | {contract.cited_relevant_rate:.4f} | "
        "answers citing at least one known-correct source |"),
        f"| grounded_rate | {contract.grounded_rate:.4f} | |",
        f"| refusal_rate | {contract.refusal_rate:.4f} | |",
        (f"| correct_refusal_rate | {contract.correct_refusal_rate:.4f} | "
        "**known gap** — the service cannot refuse today |"),
        f"| false_refusal_rate | {contract.false_refusal_rate:.4f} | must stay at 0 |",
        f"| empty_retrieval_rate | {contract.empty_retrieval_rate:.4f} | |",
        "",
        "## Answer quality",
        "",
        f"Computed: **{report.answer_quality.computed}**. Requires a real model client, a",
        "judge reachable inside the deployment boundary, and per-case ground truth.",
        "See `AnswerQualityMetrics` in `tests/eval/schema.py`.",
        "",
    ]

    if report.threshold_outcomes:
        passed = len(report.threshold_outcomes) - len(report.failed_thresholds)
        lines += [
            "## Thresholds",
            "",
            f"{passed} of {len(report.threshold_outcomes)} passed.",
            "",
        ]
        lines += [f"    {outcome.describe()}" for outcome in report.failed_thresholds]
        if report.failed_thresholds:
            lines.append("")

    lines += [
        "## Per-case detail (keyword_only)",
        "",
        "| case | gold | retrieved (top 5) | hit@5 | RR |",
        "|---|---|---|---|---|",
    ]
    for result in report.case_results:
        if result.surface is not Surface.KEYWORD_ONLY:
            continue
        at_5 = next((m for m in result.metrics if m.k == 5), None)
        hit = f"{at_5.hit_rate:.0f}" if at_5 else "—"
        rr = f"{at_5.reciprocal_rank:.3f}" if at_5 else "—"
        gold = ", ".join(result.relevant_chunk_ids) or "(none — expects refusal)"
        retrieved = ", ".join(result.retrieved_chunk_ids[:5]) or "(nothing)"
        lines.append(f"| {result.case_id} | {gold} | {retrieved} | {hit} | {rr} |")

    if report.errors:
        lines += ["", "## Errors", ""] + [f"- {error}" for error in report.errors]

    lines += ["", "## Notes", ""] + [f"- {note}" for note in report.notes] + [""]
    return "\n".join(lines)


def write_report(report: EvaluationReport, directory: Path, stem: str) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{stem}.report.json"
    markdown_path = directory / f"{stem}.report.md"
    json_path.write_text(to_json(report), encoding="utf-8", newline="\n")
    markdown_path.write_text(to_markdown(report), encoding="utf-8", newline="\n")
    return json_path, markdown_path
