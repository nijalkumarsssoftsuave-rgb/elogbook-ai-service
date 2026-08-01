"""Hand-worked metric tests with no pipeline involved.

The point is that any surprising evaluation number is unambiguously the pipeline's
fault, never the scoring's.
"""

import math

import pytest

from tests.eval.metrics import (
    hit_rate_at_k,
    mean,
    ndcg_at_k,
    precision_at_k,
    rate,
    recall_at_k,
    reciprocal_rank_at_k,
)

_ALL_RETRIEVAL_METRICS = [
    hit_rate_at_k,
    recall_at_k,
    precision_at_k,
    reciprocal_rank_at_k,
    ndcg_at_k,
]


@pytest.mark.parametrize(
    ("retrieved", "expected"),
    [
        (["a", "x", "y"], 1.0),  # relevant at rank 1
        (["x", "y", "a"], 1.0),  # relevant at rank k
        (["x", "y", "z", "a"], 0.0),  # relevant just beyond k
        (["x", "y", "z"], 0.0),  # not retrieved at all
    ],
)
def test_hit_rate_sees_only_the_top_k(retrieved: list[str], expected: float) -> None:
    assert hit_rate_at_k(retrieved, {"a"}, 3) == expected


def test_recall_counts_the_fraction_of_relevant_chunks_found() -> None:
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, 3) == 0.5
    assert recall_at_k(["a", "b", "y"], {"a", "b"}, 3) == 1.0
    assert recall_at_k(["x", "y", "z"], {"a", "b"}, 3) == 0.0


def test_recall_handles_k_longer_than_the_retrieved_list() -> None:
    assert recall_at_k(["a"], {"a"}, 5) == 1.0


def test_precision_ceiling_shows_why_it_is_not_thresholded() -> None:
    """One relevant chunk in a top-5 can never exceed 0.2, however good retrieval is."""
    assert precision_at_k(["a", "x", "y", "z", "w"], {"a"}, 5) == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("retrieved", "expected"),
    [(["a"], 1.0), (["x", "a"], 0.5), (["x", "y", "a"], 1 / 3)],
)
def test_reciprocal_rank_is_the_inverse_of_the_first_hit_position(
    retrieved: list[str], expected: float
) -> None:
    assert reciprocal_rank_at_k(retrieved, {"a"}, 5) == pytest.approx(expected)


def test_reciprocal_rank_takes_the_first_hit_not_the_first_gold_listed() -> None:
    """Two relevant chunks at ranks 3 and 1: the score is 1.0, from the rank-1 hit."""
    assert reciprocal_rank_at_k(["b", "x", "a"], {"a", "b"}, 5) == 1.0


def test_reciprocal_rank_is_zero_when_the_first_hit_falls_beyond_k() -> None:
    assert reciprocal_rank_at_k(["x", "y", "a"], {"a"}, 2) == 0.0


def test_ndcg_is_one_when_the_only_relevant_chunk_ranks_first() -> None:
    assert ndcg_at_k(["a", "x", "y"], {"a"}, 3) == 1.0


def test_ndcg_discounts_a_relevant_chunk_at_rank_two() -> None:
    assert ndcg_at_k(["x", "a"], {"a"}, 3) == pytest.approx(1 / math.log2(3))


def test_ndcg_is_one_when_both_relevant_chunks_take_the_top_two_places() -> None:
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, 3) == 1.0


def test_ndcg_penalises_a_gap_between_the_two_relevant_chunks() -> None:
    """Ranks 1 and 3 rather than 1 and 2. Hand-computed: (1 + 1/2) / (1 + 1/log2(3))."""
    expected = (1 + 1 / 2) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["a", "x", "b"], {"a", "b"}, 3) == pytest.approx(expected)


def test_ndcg_distinguishes_orderings_that_recall_and_mrr_cannot() -> None:
    """Both orderings have recall 1.0 and reciprocal rank 1.0; only nDCG separates them.
    This is the justification for including it.
    """
    tight = ["a", "b", "x", "y"]
    spread = ["a", "x", "y", "b"]

    assert recall_at_k(tight, {"a", "b"}, 4) == recall_at_k(spread, {"a", "b"}, 4)
    assert reciprocal_rank_at_k(tight, {"a", "b"}, 4) == reciprocal_rank_at_k(
        spread, {"a", "b"}, 4
    )
    assert ndcg_at_k(tight, {"a", "b"}, 4) > ndcg_at_k(spread, {"a", "b"}, 4)


def test_duplicate_retrieved_ids_do_not_inflate_recall() -> None:
    assert recall_at_k(["a", "a", "a"], {"a", "b"}, 3) == 0.5


@pytest.mark.parametrize("metric", _ALL_RETRIEVAL_METRICS)
def test_every_metric_refuses_an_empty_relevant_set(metric) -> None:
    """Scoring retrieval against no gold set is undefined. Returning 0.0 instead would
    quietly deflate aggregates with the out-of-corpus cases.
    """
    with pytest.raises(ValueError, match="not retrieval-scorable"):
        metric(["a"], set(), 3)


@pytest.mark.parametrize("metric", _ALL_RETRIEVAL_METRICS)
@pytest.mark.parametrize("k", [0, -1])
def test_every_metric_refuses_a_non_positive_k(metric, k: int) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        metric(["a"], {"a"}, k)


def test_mean_of_no_values_is_zero() -> None:
    assert mean([]) == 0.0


def test_mean_is_a_macro_average() -> None:
    assert mean([1.0, 0.0]) == 0.5


def test_rate_with_an_empty_denominator_is_zero() -> None:
    assert rate(0, 0) == 0.0
