import pytest
from pydantic import ValidationError

from app.application.confidence.confidence_scoring_service import (
    ConfidencePolicy,
    ConfidenceScoringService,
)
from app.domain.citation import (
    Citation,
    CitationFailureReason,
    CitationValidationResult,
)
from app.domain.confidence import ConfidenceBand, ConfidenceSignalName
from app.domain.models import GeneratedAnswer, RetrievedChunk
from tests.conftest import DEFAULT_CONFIDENCE_POLICY

# The policy the service actually ships, not a copy of it. Restating the weights here would
# let the tested behaviour drift away from the configured behaviour silently -- and the
# weighting is exactly the part a deployment is invited to change.
POLICY = DEFAULT_CONFIDENCE_POLICY


def _chunks(*scores: float) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=f"log-{index:03d}",
            document_id=f"doc-{index:03d}",
            text="evidence",
            score=score,
        )
        for index, score in enumerate(scores, start=1)
    ]


def _generated(citation_count: int) -> GeneratedAnswer:
    return GeneratedAnswer(
        answer_text="an answer",
        citations=[
            Citation(citation_id=f"c{n}", chunk_id=f"log-{n:03d}", order=n)
            for n in range(1, citation_count + 1)
        ],
    )


def _score(
    chunks: list[RetrievedChunk],
    requested_top_k: int = 5,
    citation_count: int = 2,
    validation: CitationValidationResult | None = None,
    policy: ConfidencePolicy = POLICY,
) -> float:
    return (
        ConfidenceScoringService(policy)
        .score(
            chunks,
            requested_top_k,
            _generated(citation_count),
            validation or CitationValidationResult.valid(),
        )
        .score
    )


def _signal(result, name: ConfidenceSignalName):
    return next((s for s in result.signals if s.name is name), None)


# --- retrieval quality moves the score ----------------------------------------------------


def test_full_retrieval_scores_higher_than_partial_retrieval() -> None:
    """The headline case: everything else held fixed, finding what was asked for beats
    finding a fraction of it.
    """
    full = _score(_chunks(0.9, 0.5, 0.4, 0.3, 0.2))
    partial = _score(_chunks(0.9, 0.5))

    assert full > partial


def test_retrieving_nothing_scores_zero_and_lands_in_the_low_band() -> None:
    result = ConfidenceScoringService(POLICY).score(
        [], 5, _generated(0), CitationValidationResult.invalid(CitationFailureReason.NO_CITATIONS)
    )

    assert result.score == 0.0
    assert result.band is ConfidenceBand.LOW


def test_the_retrieval_signal_reports_what_it_measured() -> None:
    """A bare number is not actionable; the detail is what tells an operator which half of
    the pipeline to look at.
    """
    result = ConfidenceScoringService(POLICY).score(
        _chunks(0.9, 0.5), 5, _generated(2), CitationValidationResult.valid()
    )

    assert _signal(result, ConfidenceSignalName.RETRIEVAL).value == pytest.approx(0.4)
    assert "2 of 5" in _signal(result, ConfidenceSignalName.RETRIEVAL).detail


def test_retrieving_more_than_was_asked_for_does_not_score_above_full() -> None:
    """Coverage is a proportion of what was requested, and a proportion above 1 would let a
    generous retriever buy confidence it has not earned.
    """
    assert _score(_chunks(*[0.5] * 8)) == _score(_chunks(*[0.5] * 5))


# --- ranking separation -------------------------------------------------------------------


def test_a_clear_top_result_scores_higher_than_a_flat_ranking() -> None:
    """Same number of chunks either way. A ranking whose top result barely beats its
    runners-up is one where the evidence handed to the model could easily have differed.
    """
    decisive = _score(_chunks(1.0, 0.1, 0.1, 0.1, 0.1))
    flat = _score(_chunks(1.0, 1.0, 1.0, 1.0, 1.0))

    assert decisive > flat


def test_the_ranking_signal_is_omitted_when_there_is_nothing_to_compare() -> None:
    """A single chunk has no runner-up to stand out from. Inventing a value for that would
    be putting an opinion into a measurement, so the signal is left out and its weight
    stops counting.
    """
    result = ConfidenceScoringService(POLICY).score(
        _chunks(0.9), 1, _generated(1), CitationValidationResult.valid()
    )

    assert _signal(result, ConfidenceSignalName.RERANKER) is None
    # Retrieval is complete and every citation resolved, so the two remaining signals are
    # both 1.0 -- and the score is 1.0, not dragged down by a missing third.
    assert result.score == 1.0


def test_tied_top_scores_count_as_no_separation() -> None:
    """Two equally strong chunks means the ranking did not choose between them, which is
    the opposite of a decisive result -- so the duplicate belongs in the comparison, not
    discarded as a tie.
    """
    result = ConfidenceScoringService(POLICY).score(
        _chunks(1.0, 1.0), 2, _generated(2), CitationValidationResult.valid()
    )

    assert _signal(result, ConfidenceSignalName.RERANKER).value == 0.0


def test_the_ranking_signal_is_scale_free() -> None:
    """Scores arrive as BM25 values, cosine similarities or reciprocal ranks depending on
    what ran. Multiplying every score by a constant is the same ranking, and must be the
    same signal.
    """
    small = ConfidenceScoringService(POLICY).score(
        _chunks(0.03, 0.01, 0.01), 3, _generated(2), CitationValidationResult.valid()
    )
    large = ConfidenceScoringService(POLICY).score(
        _chunks(3000.0, 1000.0, 1000.0), 3, _generated(2), CitationValidationResult.valid()
    )

    assert small.score == large.score


def test_a_non_positive_top_score_omits_the_ranking_signal_rather_than_dividing_by_it() -> None:
    result = ConfidenceScoringService(POLICY).score(
        _chunks(0.0, 0.0), 2, _generated(2), CitationValidationResult.valid()
    )

    assert _signal(result, ConfidenceSignalName.RERANKER) is None


# --- invalid citations reduce confidence ---------------------------------------------------


def test_an_unresolved_citation_lowers_the_score() -> None:
    chunks = _chunks(0.9, 0.5, 0.4, 0.3, 0.2)
    clean = _score(chunks, citation_count=2)
    partly_unresolved = _score(
        chunks,
        citation_count=2,
        validation=CitationValidationResult.invalid(
            CitationFailureReason.UNRESOLVED_CITATION, ["c2"]
        ),
    )

    assert partly_unresolved < clean


def test_more_unresolved_citations_lower_the_score_further() -> None:
    chunks = _chunks(0.9, 0.5, 0.4, 0.3, 0.2)
    one_bad = _score(
        chunks,
        citation_count=4,
        validation=CitationValidationResult.invalid(
            CitationFailureReason.UNRESOLVED_CITATION, ["c4"]
        ),
    )
    three_bad = _score(
        chunks,
        citation_count=4,
        validation=CitationValidationResult.invalid(
            CitationFailureReason.UNRESOLVED_CITATION, ["c2", "c3", "c4"]
        ),
    )

    assert three_bad < one_bad


def test_an_answer_that_cites_nothing_scores_zero_on_citations() -> None:
    """Fluent and untraceable is the failure the whole pipeline exists to prevent, so it
    earns no credit at all rather than partial credit.
    """
    result = ConfidenceScoringService(POLICY).score(
        _chunks(0.9, 0.5, 0.4, 0.3, 0.2),
        5,
        _generated(0),
        CitationValidationResult.invalid(CitationFailureReason.NO_CITATIONS),
    )

    assert _signal(result, ConfidenceSignalName.CITATION).value == 0.0


def test_perfect_retrieval_cannot_rescue_an_uncited_answer_past_the_review_band() -> None:
    """The weighting made concrete: citations carry enough that losing them entirely keeps
    an otherwise flawless answer out of the HIGH band.
    """
    result = ConfidenceScoringService(POLICY).score(
        _chunks(1.0, 0.0, 0.0, 0.0, 0.0),
        5,
        _generated(0),
        CitationValidationResult.invalid(CitationFailureReason.NO_CITATIONS),
    )

    assert result.band is not ConfidenceBand.HIGH


# --- determinism ---------------------------------------------------------------------------


def test_the_same_inputs_always_produce_the_same_score() -> None:
    chunks = _chunks(0.83, 0.51, 0.37, 0.22, 0.19)
    service = ConfidenceScoringService(POLICY)

    results = [
        service.score(chunks, 5, _generated(3), CitationValidationResult.valid())
        for _ in range(25)
    ]

    assert len({result.score for result in results}) == 1
    assert len({result.band for result in results}) == 1


def test_two_separately_built_services_agree_on_the_same_inputs() -> None:
    """Determinism across instances too -- nothing is accumulated between calls."""
    chunks = _chunks(0.83, 0.51, 0.37)

    first = ConfidenceScoringService(POLICY).score(
        chunks, 5, _generated(3), CitationValidationResult.valid()
    )
    second = ConfidenceScoringService(POLICY).score(
        chunks, 5, _generated(3), CitationValidationResult.valid()
    )

    assert first == second


def test_the_score_does_not_depend_on_the_order_chunks_arrive_in() -> None:
    """Chunks arrive ranked, but the signals are properties of the set, not of its order.
    A signal that quietly changed with the ordering would make the score untestable.
    """
    ascending = ConfidenceScoringService(POLICY).score(
        _chunks(0.2, 0.5, 0.9), 3, _generated(2), CitationValidationResult.valid()
    )
    descending = ConfidenceScoringService(POLICY).score(
        _chunks(0.9, 0.5, 0.2), 3, _generated(2), CitationValidationResult.valid()
    )

    assert ascending.score == descending.score


# --- the policy is configuration, not a constant --------------------------------------------


def test_reweighting_changes_the_score() -> None:
    """The point of Phase 3: the scoring policy is tunable, so a deployment whose retrieval
    is weak can stop letting retrieval dominate the number.
    """
    chunks = _chunks(0.9, 0.5)
    citations_only = ConfidencePolicy(
        retrieval_weight=0.0,
        reranker_weight=0.0,
        citation_weight=1.0,
        refusal_threshold=0.3,
        review_threshold=0.6,
    )

    assert _score(chunks, policy=citations_only) == 1.0
    assert _score(chunks) < 1.0


@pytest.mark.parametrize("switched_off", list(ConfidenceSignalName))
def test_a_zero_weighted_signal_is_not_measured_at_all(
    switched_off: ConfidenceSignalName,
) -> None:
    """Weighting a signal to zero removes it rather than measuring it and multiplying the
    result away. The difference shows in `signals`, which is what an operator reads to see
    what the score was built from.
    """
    weights = dict.fromkeys(ConfidenceSignalName, 1.0)
    weights[switched_off] = 0.0
    policy = ConfidencePolicy(
        retrieval_weight=weights[ConfidenceSignalName.RETRIEVAL],
        reranker_weight=weights[ConfidenceSignalName.RERANKER],
        citation_weight=weights[ConfidenceSignalName.CITATION],
        refusal_threshold=0.3,
        review_threshold=0.6,
    )

    result = ConfidenceScoringService(policy).score(
        _chunks(0.9, 0.5), 5, _generated(2), CitationValidationResult.valid()
    )

    assert _signal(result, switched_off) is None
    assert {signal.name for signal in result.signals} == set(ConfidenceSignalName) - {
        switched_off
    }


@pytest.mark.parametrize(
    ("score_inputs", "expected"),
    [
        ((1.0, 0.1, 0.1, 0.1, 0.1), ConfidenceBand.HIGH),
        ((0.5, 0.5), ConfidenceBand.NEEDS_REVIEW),
        ((), ConfidenceBand.LOW),
    ],
)
def test_the_band_follows_the_configured_thresholds(score_inputs, expected) -> None:
    citation_count = 2 if score_inputs else 0
    validation = (
        CitationValidationResult.valid()
        if score_inputs
        else CitationValidationResult.invalid(CitationFailureReason.NO_CITATIONS)
    )

    result = ConfidenceScoringService(POLICY).score(
        _chunks(*score_inputs), 5, _generated(citation_count), validation
    )

    assert result.band is expected


def test_a_policy_whose_thresholds_are_inverted_is_rejected_on_construction() -> None:
    """Caught at startup rather than producing a band nobody can reach: with refusal above
    review, NEEDS_REVIEW would be unreachable and every marginal answer would read LOW.
    """
    with pytest.raises(ValidationError):
        ConfidencePolicy(
            retrieval_weight=0.4,
            reranker_weight=0.2,
            citation_weight=0.4,
            refusal_threshold=0.8,
            review_threshold=0.5,
        )


def test_a_policy_with_no_weight_anywhere_is_rejected() -> None:
    """Every signal switched off would score every answer 0.0 and refuse the lot, which is
    a misconfiguration rather than a policy.
    """
    with pytest.raises(ValidationError):
        ConfidencePolicy(
            retrieval_weight=0.0,
            reranker_weight=0.0,
            citation_weight=0.0,
            refusal_threshold=0.3,
            review_threshold=0.6,
        )
