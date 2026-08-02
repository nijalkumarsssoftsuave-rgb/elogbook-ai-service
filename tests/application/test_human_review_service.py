import pytest

from app.application.review.human_review_service import (
    HumanReviewService,
    ReviewRoutingOutcome,
)
from app.domain.citation import ResolvedCitation
from app.domain.confidence import (
    ConfidenceBand,
    ConfidenceResult,
    ConfidenceSignal,
    ConfidenceSignalName,
)
from app.domain.models import GroundedAnswer, Question, RetrievedChunk
from app.domain.review import ReviewRequest
from app.infrastructure.review.in_memory_review_queue import InMemoryReviewQueue

QUESTION = Question(text="What caused the fire alarm?", user_id="tech-42", language="en")

EVIDENCE = [
    RetrievedChunk(
        chunk_id="log-006",
        document_id="doc-log-006",
        text="Fire alarm triggered at 02:47 due to dust near the server room.",
        score=1.4,
        metadata={"source_title": "Night Shift Alarm Report", "source_id": "incidents"},
    )
]


class BrokenQueue:
    """Fails the way a real broker fails: at the moment of the call, with its own error."""

    class Unreachable(RuntimeError):
        pass

    def __init__(self) -> None:
        self.attempts = 0

    async def enqueue(self, request: ReviewRequest) -> None:
        self.attempts += 1
        raise self.Unreachable("no route to the review queue")


def _answer() -> GroundedAnswer:
    return GroundedAnswer(
        answer_text="Dust near the server room set it off. [c1]",
        citations=[
            ResolvedCitation(
                citation_id="c1",
                order=1,
                chunk_id="log-006",
                source_id="incidents",
                document_id="doc-log-006",
                source_title="Night Shift Alarm Report",
                score=1.4,
            )
        ],
        confidence=0.42,
    )


def _confidence(band: ConfidenceBand, score: float = 0.42) -> ConfidenceResult:
    return ConfidenceResult(
        score=score,
        band=band,
        signals=[
            ConfidenceSignal(
                name=ConfidenceSignalName.RETRIEVAL,
                value=score,
                weight=1.0,
                detail="1 of 5 requested chunks retrieved",
            )
        ],
    )


async def _route(band: ConfidenceBand, queue=None, score: float = 0.42):
    queue = queue if queue is not None else InMemoryReviewQueue()
    outcome = await HumanReviewService(queue).route(
        QUESTION, _answer(), _confidence(band, score), EVIDENCE, "cid-1"
    )
    return outcome, queue


# --- who gets reviewed --------------------------------------------------------------------


async def test_a_high_confidence_answer_creates_no_review_task() -> None:
    outcome, queue = await _route(ConfidenceBand.HIGH, score=0.91)

    assert outcome is ReviewRoutingOutcome.NOT_REQUIRED
    assert queue.pending == []


async def test_a_low_confidence_answer_creates_exactly_one_review_task() -> None:
    outcome, queue = await _route(ConfidenceBand.LOW, score=0.12)

    assert outcome is ReviewRoutingOutcome.QUEUED
    assert len(queue.pending) == 1


async def test_a_needs_review_answer_creates_a_review_task() -> None:
    """The band exists to name exactly this: delivered to the caller, and still worth a
    human look.
    """
    outcome, queue = await _route(ConfidenceBand.NEEDS_REVIEW, score=0.55)

    assert outcome is ReviewRoutingOutcome.QUEUED
    assert len(queue.pending) == 1


@pytest.mark.parametrize("band", list(ConfidenceBand))
async def test_only_a_high_band_escapes_review(band: ConfidenceBand) -> None:
    """Walks the enum rather than the bands someone remembered, so a band added later routes
    to review by default. Erring that way costs a reviewer's time; erring the other way
    means an answer nobody checks.
    """
    _, queue = await _route(band)

    assert bool(queue.pending) is (band is not ConfidenceBand.HIGH)


# --- what the review task contains ----------------------------------------------------------


async def test_the_review_task_carries_the_question_answer_confidence_and_citations() -> None:
    _, queue = await _route(ConfidenceBand.LOW, score=0.12)

    request = queue.pending[0]
    assert request.correlation_id == "cid-1"
    assert request.question == QUESTION
    assert request.answer_text == "Dust near the server room set it off. [c1]"
    assert request.confidence_score == 0.12
    assert request.confidence_band is ConfidenceBand.LOW
    assert [citation.chunk_id for citation in request.citations] == ["log-006"]


async def test_the_review_task_carries_the_evidence_the_answer_was_built_from() -> None:
    """A reviewer opens this hours later, when the retrieved chunks are long gone. The
    evidence travels with the request rather than being something the queue must fetch back.
    """
    _, queue = await _route(ConfidenceBand.LOW)

    request = queue.pending[0]
    assert [chunk.chunk_id for chunk in request.evidence] == ["log-006"]
    assert "dust near the server room" in request.evidence[0].text.lower()


async def test_the_review_task_records_the_answer_that_was_withheld() -> None:
    """A LOW answer never reaches the caller, so a task holding the canned refusal would
    tell a reviewer nothing about what was withheld or why it scored badly.
    """
    _, queue = await _route(ConfidenceBand.LOW)

    request = queue.pending[0]
    assert request.answer_text != GroundedAnswer.REFUSAL_TEXT
    assert request.was_withheld is True


async def test_a_reviewed_but_delivered_answer_is_not_marked_as_withheld() -> None:
    _, queue = await _route(ConfidenceBand.NEEDS_REVIEW, score=0.55)

    assert queue.pending[0].was_withheld is False


async def test_the_queued_evidence_does_not_alias_the_callers_list() -> None:
    """The request is a snapshot. A list the pipeline goes on to mutate would let a review
    task quietly change after it was created.
    """
    evidence = list(EVIDENCE)
    queue = InMemoryReviewQueue()

    await HumanReviewService(queue).route(
        QUESTION, _answer(), _confidence(ConfidenceBand.LOW), evidence, "cid-1"
    )
    evidence.clear()

    assert len(queue.pending[0].evidence) == 1


# --- routing changes nothing -----------------------------------------------------------------


async def test_routing_returns_no_answer_and_leaves_the_score_alone() -> None:
    """The ticket's constraint made checkable: routing reads the answer and creates work.
    There is no return path by which queueing a review could alter what a caller receives.
    """
    answer, confidence = _answer(), _confidence(ConfidenceBand.LOW)

    outcome = await HumanReviewService(InMemoryReviewQueue()).route(
        QUESTION, answer, confidence, EVIDENCE, "cid-1"
    )

    assert isinstance(outcome, ReviewRoutingOutcome)
    assert not isinstance(outcome, GroundedAnswer)
    assert answer.confidence == 0.42
    assert confidence.score == 0.42
    assert confidence.band is ConfidenceBand.LOW


# --- queue failures ---------------------------------------------------------------------------


async def test_a_queue_failure_is_reported_rather_than_raised() -> None:
    """A queue that is down must not take the caller's answer with it: the answer was
    already decided on its own merits, and failing the request would trade availability for
    nothing.
    """
    outcome, queue = await _route(ConfidenceBand.LOW, queue=BrokenQueue())

    assert outcome is ReviewRoutingOutcome.QUEUE_UNAVAILABLE
    assert queue.attempts == 1


async def test_a_lost_review_task_is_distinguishable_from_one_that_was_never_needed() -> None:
    """Both leave no task in the queue, and they mean opposite things -- one is the system
    working, the other is a gap somebody has to know about. A boolean return would collapse
    them.
    """
    unavailable, _ = await _route(ConfidenceBand.LOW, queue=BrokenQueue())
    not_required, _ = await _route(ConfidenceBand.HIGH, score=0.91)

    assert unavailable is not not_required


async def test_a_queue_that_is_never_called_cannot_fail_the_request() -> None:
    """A HIGH answer does not touch the queue at all, so a broken queue is invisible to it."""
    outcome, queue = await _route(ConfidenceBand.HIGH, queue=BrokenQueue(), score=0.91)

    assert outcome is ReviewRoutingOutcome.NOT_REQUIRED
    assert queue.attempts == 0
