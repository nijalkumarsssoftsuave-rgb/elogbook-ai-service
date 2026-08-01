# Evaluation suite (ES-322) — independent EN / AR accuracy

Measures English and Arabic retrieval accuracy separately, and turns those numbers into
regression tests so a future model or corpus change cannot quietly make things worse.

```bash
uv run pytest tests/eval                              # the regression suite
uv run python -m tests.eval.run --all                 # human-readable report
uv run python -m tests.eval.run --language ar --format json
uv run python -m tests.eval.run --all --write-baseline
uv run python -m tests.eval.run --language en --print-thresholds
```

Run from the repository root. Exit codes: `0` all thresholds passed, `1` a threshold
failed, `2` a case errored or a dataset failed to load.

## What is measured, and what is not

**Measured.** Retrieval accuracy against known-correct chunk ids, and the pipeline's
structural contract — do citations resolve to retrieved chunks, does the service refuse
when it should.

**Not measured: answer quality.** The LLM, the embedding model, the dense vector store
and the reranker are all stubs. `ModelClientStub` returns one fixed sentence citing every
chunk it was handed, so any "answer accuracy" derived from it would score the stub, not
the system. `AnswerQualityMetrics` reserves the fields and stays `computed=False`. An
absent number is honest; a fabricated one is not.

## Three surfaces

A single end-to-end measurement would misattribute stub artifacts to retrieval quality,
so results are read from three places:

| Surface | Read from | Answers |
|---|---|---|
| `keyword_only` | `BM25KeywordRetriever.search` | How good is the one real retrieval component? **This is the primary EN-vs-AR comparison.** |
| `retrieval_raw` | `RetrievalService.retrieve` | What does a caller actually get, after fusion and reranking? |
| `pipeline_contract` | `QAApplicationService.execute` | Does the pipeline honour its contract? Contract metrics only. |

Retrieval accuracy is never computed from pipeline citations: that list is a function of
the model stub citing everything it is given, so it would be a laundered copy of
`retrieval_raw` dressed up as an answer-level measurement.

## Known gaps, and the tripwires that watch them

`test_pipeline_tripwires.py` encodes these as tests that **fail when the pipeline
improves**, each with a message saying what to do next. A red build there is good news.

1. **Rank 1 of `retrieval_raw` is always `dense-stub-chunk-1`.** The dense stub returns
   one fixed chunk that ties the top keyword hit on RRF score and wins by insertion
   order. This caps MRR at 0.5 and pins precision@1 at 0.0 on that surface — which is why
   `thresholds.json` has no `k=1` block for it, and why `keyword_only` is the number to
   quote.
2. **The service cannot refuse.** Even when BM25 returns nothing, the dense stub supplies
   evidence, so citations always validate. `refusal_rate` and `correct_refusal_rate` are
   structurally 0.0, and the two out-of-corpus cases in each dataset are recorded as a
   known gap rather than a passing metric.
3. **`citation_corpus_coverage` is below 1.0** because the dense stub's chunk belongs to
   no corpus. That gap is the stub, made visible.

## Measured weaknesses the datasets deliberately expose

BM25 has no stemming, no stop-word list and no semantic matching. Rather than hide that,
several cases target it, and the per-case table in the baseline report shows the damage:

- `en-009` / `ar-009` — plural vs singular (`incidents` vs `Incident`) scores **0**.
- `en-011` / `ar-011` — a paraphrase sharing almost no content words ranks 4th.
- `en-012` / `ar-012` — "emergency siren" pulls the wrong entry above the right one.
- `en-014` — an out-of-corpus question still retrieves entries, matching only on `the`.

These are exactly the gaps a dense retriever exists to close, so the numbers double as
the argument for wiring BGE-M3 in.

## Baseline and thresholds

`baseline/` holds **committed** report snapshots with the timestamp stripped, so an
unchanged rerun diffs to zero lines and a real change diffs like a changelog
("`en-011` moved from rank 4 to rank 2"). Thresholds tell you *that* something moved; the
baseline tells you *what*. `reports/` is gitignored scratch output.

Floors in `thresholds.json` sit just below measured values, never at an aspirational
target — the suite exists to catch a real drop, not to fail on day one. The regeneration
procedure is written into that file. English and Arabic have wholly separate blocks: their
token statistics differ, so a shared threshold would be too loose for one and permanently
failing for the other.

`precision_at_k` is reported but never thresholded — with one or two relevant chunks per
question its ceiling at k=5 is 0.2–0.4, so a floor would penalise a perfect run.

## Enabling RAGAS later

CLAUDE.md names RAGAS for evaluation. It is deliberately not a dependency yet, because it
drives an LLM judge over the network and the deployment target is air-gapped. Three things
must be true before it can be switched on, all listed on `AnswerQualityMetrics`:

1. `ModelClientPort` is backed by a real model, so the answer text is the system's own.
2. A judge model is reachable **inside** the deployment boundary.
3. `ragas` is added to the dev dependencies and each case carries a ground-truth answer.

## Notes for anyone extending this

- Datasets are UTF-8 JSON validated on load, with `extra="forbid"`: a mistyped key fails
  at load rather than silently scoring the case zero and looking like a regression.
- `loader.validate_against_corpus` cross-checks every gold id against the real corpus, so
  a typo'd `log-06` is reported as a dataset bug, not a retrieval failure.
- The suite builds its own object graph in `surfaces.py`. The DI composing functions in
  `app/api/dependencies.py` return `fastapi.params.Depends` placeholders when called
  outside a request — silently, with no error. `test_surfaces_match_production_wiring`
  guards against the evaluation drifting away from production wiring.
- Metric functions live in `metrics.py` with no `app` imports and are verified against
  hand-worked examples, so a surprising number is always the pipeline's fault.
- Reports print case and chunk ids, never query text: it keeps mixed LTR/RTL diffs
  readable and avoids the cp1252 Windows console entirely.
