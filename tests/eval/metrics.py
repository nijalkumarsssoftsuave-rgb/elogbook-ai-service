"""Pure retrieval and contract metrics. No imports from `app`, deliberately.

The maths here is verified against hand-worked examples in test_metrics.py, so a
surprising evaluation number can always be attributed to the pipeline rather than to the
scoring. Relevance is binary: a chunk id is either known-correct or it is not.
"""

import math


def _validate(relevant: set[str], k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not relevant:
        # Retrieval accuracy against an empty gold set is undefined. Cases with no
        # relevant chunks are scored on refusal behaviour instead; silently returning
        # 0.0 here would deflate every aggregate and look like a retrieval failure.
        raise ValueError("relevant set is empty; this case is not retrieval-scorable")


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Did the user see any correct answer at all in the top k."""
    _validate(relevant, k)
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    _validate(relevant, k)
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Reported but never thresholded: with one or two relevant chunks the ceiling is
    1/k or 2/k, so a floor on it would penalise a perfect run.
    """
    _validate(relevant, k)
    return len(set(retrieved[:k]) & relevant) / k


def reciprocal_rank_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1/rank of the first correct chunk. Aggregates into MRR."""
    _validate(relevant, k)
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Binary-gain nDCG.

    The only metric here sensitive to where the *second* and later correct chunks landed:
    hit-rate and reciprocal rank both stop at the first hit, and recall ignores order
    entirely. That is what makes it worth its eight lines on the multi-source cases.
    """
    _validate(relevant, k)
    discounted_gain = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved[:k], start=1)
        if chunk_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    ideal_gain = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return discounted_gain / ideal_gain if ideal_gain > 0 else 0.0


def mean(values: list[float]) -> float:
    """Macro-average: every case counts once regardless of how many relevant chunks it
    has, so a two-source question cannot outvote a one-source question.
    """
    return sum(values) / len(values) if values else 0.0


def rate(matching: int, total: int) -> float:
    """Contract metrics are all "fraction of cases that did X". An empty denominator is
    0.0 rather than an error: a dataset with no out-of-corpus cases legitimately has no
    correct-refusal rate, and that should not stop a run.
    """
    return matching / total if total else 0.0
