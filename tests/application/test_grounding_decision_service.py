import pytest

from app.application.confidence.grounding_decision_service import GroundingDecisionService
from app.domain.citation import ResolvedCitation
from app.domain.confidence import (
    ConfidenceBand,
    ConfidenceResult,
    ConfidenceSignal,
    ConfidenceSignalName,
)
from app.domain.models import GroundedAnswer


def _answer(confidence: float) -> GroundedAnswer:
    return GroundedAnswer(
        answer_text="The alarm was caused by dust near the server room. [c1]",
        citations=[
            ResolvedCitation(
                citation_id="c1",
                order=1,
                chunk_id="log-006",
                source_id="incidents",
                document_id="doc-log-006",
                source_title="Night Shift Alarm Report",
                score=1.0,
            )
        ],
        confidence=confidence,
    )


def _confidence(score: float, band: ConfidenceBand) -> ConfidenceResult:
    return ConfidenceResult(
        score=score,
        band=band,
        signals=[
            ConfidenceSignal(
                name=ConfidenceSignalName.RETRIEVAL, value=score, weight=1.0, detail="stub"
            )
        ],
    )


# --- high confidence returns the answer ----------------------------------------------------


def test_a_high_confidence_answer_is_returned_untouched() -> None:
    answer = _answer(0.9)

    decided = GroundingDecisionService().decide(
        answer, _confidence(0.9, ConfidenceBand.HIGH)
    )

    assert decided == answer
    assert decided.refused is False
    assert decided.citations


def test_a_needs_review_answer_is_still_returned() -> None:
    """The middle band means "usable, but worth a human look". Withholding it would make
    NEEDS_REVIEW indistinguishable from LOW, and the review threshold would then be doing
    nothing the refusal threshold does not already do.
    """
    answer = _answer(0.55)

    decided = GroundingDecisionService().decide(
        answer, _confidence(0.55, ConfidenceBand.NEEDS_REVIEW)
    )

    assert decided == answer
    assert decided.refused is False


# --- low confidence returns a refusal ------------------------------------------------------


def test_a_low_confidence_answer_is_replaced_by_a_grounded_refusal() -> None:
    decided = GroundingDecisionService().decide(
        _answer(0.12), _confidence(0.12, ConfidenceBand.LOW)
    )

    assert decided.refused is True
    assert decided.is_grounded is False
    assert decided.answer_text == GroundedAnswer.REFUSAL_TEXT
    assert decided.citations == []


def test_the_withheld_answer_does_not_survive_in_any_form() -> None:
    """Not "the answer with a warning attached" -- the text the model produced must not
    reach the caller at all, or the refusal is advisory rather than a refusal.
    """
    decided = GroundingDecisionService().decide(
        _answer(0.12), _confidence(0.12, ConfidenceBand.LOW)
    )

    assert "dust near the server room" not in decided.answer_text


def test_a_low_confidence_refusal_carries_the_score_that_caused_it() -> None:
    """So the audit trail can tell the two kinds of refusal apart: this one had an answer
    and judged it too weakly supported, rather than having nothing to answer from.
    """
    decided = GroundingDecisionService().decide(
        _answer(0.12), _confidence(0.12, ConfidenceBand.LOW)
    )

    assert decided.confidence == 0.12


# --- the decision reads the band, not the score --------------------------------------------


def test_the_same_score_decides_differently_when_the_band_differs() -> None:
    """The proof that this service holds no thresholds of its own. Thresholds are
    configuration and live on ConfidencePolicy; a second comparison here would be free to
    drift from the first, and a deployment that moved its thresholds would find the
    decision quietly ignoring them.
    """
    answer = _answer(0.5)

    as_low = GroundingDecisionService().decide(answer, _confidence(0.5, ConfidenceBand.LOW))
    as_high = GroundingDecisionService().decide(answer, _confidence(0.5, ConfidenceBand.HIGH))

    assert as_low.refused is True
    assert as_high.refused is False


@pytest.mark.parametrize("band", list(ConfidenceBand))
def test_every_band_is_decided_rather_than_falling_through(band: ConfidenceBand) -> None:
    """A band added later without a rule here would silently be treated as acceptable, so
    this walks the enum rather than the three cases someone remembered.
    """
    decided = GroundingDecisionService().decide(_answer(0.5), _confidence(0.5, band))

    assert decided.refused is (band is ConfidenceBand.LOW)


# --- determinism ---------------------------------------------------------------------------


@pytest.mark.parametrize("band", list(ConfidenceBand))
def test_the_same_inputs_always_decide_the_same_way(band: ConfidenceBand) -> None:
    service = GroundingDecisionService()
    answer, confidence = _answer(0.5), _confidence(0.5, band)

    decisions = [service.decide(answer, confidence) for _ in range(25)]

    assert all(decision == decisions[0] for decision in decisions)


def test_deciding_does_not_mutate_the_answer_it_was_given() -> None:
    """It returns a verdict, it does not edit the answer in place -- otherwise a refused
    answer could not be audited as what the model actually produced.
    """
    answer = _answer(0.12)

    GroundingDecisionService().decide(answer, _confidence(0.12, ConfidenceBand.LOW))

    assert answer.refused is False
    assert answer.citations
    assert "dust near the server room" in answer.answer_text
