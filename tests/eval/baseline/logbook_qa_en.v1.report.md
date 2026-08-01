# Evaluation report — logbook_qa_en v1 (en)

Corpus: 16 chunks (ar, en) · 15 cases (13 answerable, 2 out-of-corpus) · top_k=5

## What this measures

Retrieval accuracy against known-correct chunk ids, and the pipeline's
structural contract. **Answer quality is not measured** — the LLM, embeddings,
dense vector search and the reranker are stubs.

## Retrieval — keyword_only

_BM25 keyword search -- the one real retrieval component_

| k | hit@k | recall@k | precision@k | MRR@k | nDCG@k |
|---|---|---|---|---|---|
| 1 | 0.6923 | 0.6154 | 0.6923 | 0.6923 | 0.6923 |
| 3 | 0.8462 | 0.8462 | 0.3333 | 0.7692 | 0.7832 |
| 5 | 0.9231 | 0.9231 | 0.2154 | 0.7885 | 0.8163 |

## Retrieval — retrieval_raw

_RetrievalService: RRF over dense + keyword, then reranking_

| k | hit@k | recall@k | precision@k | MRR@k | nDCG@k |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3 | 0.8462 | 0.8077 | 0.3077 | 0.3974 | 0.4998 |
| 5 | 0.9231 | 0.9231 | 0.2154 | 0.4128 | 0.5498 |

> MRR on this surface is capped and precision@1 pinned at 0: the dense
> stub returns one fixed chunk that ties the top keyword hit on RRF score
> and wins the tie by insertion order, so rank 1 is always the stub.
> Compare against `keyword_only` for the un-polluted retrieval number.

## Pipeline contract

| metric | value | note |
|---|---|---|
| citation_validity_rate | 1.0000 | every citation points at a retrieved chunk |
| citation_corpus_coverage | 0.7581 | below 1.0 because the dense stub's chunk is in no corpus |
| cited_relevant_rate | 0.9231 | answers citing at least one known-correct source |
| grounded_rate | 1.0000 | |
| refusal_rate | 0.0000 | |
| correct_refusal_rate | 0.0000 | **known gap** — the service cannot refuse today |
| false_refusal_rate | 0.0000 | must stay at 0 |
| empty_retrieval_rate | 0.0000 | |

## Answer quality

Computed: **False**. Requires a real model client, a
judge reachable inside the deployment boundary, and per-case ground truth.
See `AnswerQualityMetrics` in `tests/eval/schema.py`.

## Thresholds

22 of 22 passed.

## Per-case detail (keyword_only)

| case | gold | retrieved (top 5) | hit@5 | RR |
|---|---|---|---|---|
| en-001 | log-001 | log-001, log-006, log-002, log-005, log-003 | 1 | 1.000 |
| en-002 | log-002 | log-002, log-006, log-008 | 1 | 1.000 |
| en-003 | log-003 | log-003, log-006, log-002, log-005, log-007 | 1 | 1.000 |
| en-004 | log-004 | log-004, log-006, log-008, log-001, log-002 | 1 | 1.000 |
| en-005 | log-005 | log-005, log-001 | 1 | 1.000 |
| en-006 | log-006 | log-006, log-002, log-003, log-004, log-001 | 1 | 1.000 |
| en-007 | log-007 | log-006, log-007, log-002 | 1 | 0.500 |
| en-008 | log-008 | log-008, log-006, log-002, log-001 | 1 | 1.000 |
| en-009 | log-002, log-008 | log-004, log-001 | 0 | 0.000 |
| en-010 | log-003, log-005 | log-003, log-005, log-004, log-008, log-001 | 1 | 1.000 |
| en-011 | log-002 | log-006, log-008, log-007, log-002 | 1 | 0.250 |
| en-012 | log-006 | log-004, log-006, log-002 | 1 | 0.500 |
| en-013 | log-003, log-006 | log-006, log-002, log-003, log-008, log-001 | 1 | 1.000 |
| en-014 | (none — expects refusal) | log-006, log-002 | — | — |
| en-015 | (none — expects refusal) | (nothing) | — | — |

## Notes

- Answer quality is not measured: the LLM, embeddings, dense search and the reranker are stubs, so a score would measure the stub rather than the system.
- rank 1 of retrieval_raw is always the dense stub's chunk, which caps MRR at 0.5 and pins precision@1 at 0.0 there. Use keyword_only for the retrieval number.
- The service cannot currently refuse: the dense stub always supplies evidence, so citations always validate and correct_refusal_rate is structurally 0.0.
