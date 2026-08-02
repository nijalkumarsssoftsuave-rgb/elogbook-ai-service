from enum import StrEnum

from app.application.review.ports.review_queue_port import ReviewQueuePort
from app.domain.confidence import ConfidenceBand, ConfidenceResult
from app.domain.models import GroundedAnswer, Question, RetrievedChunk
from app.domain.review import ReviewRequest


class ReviewRoutingOutcome(StrEnum):
    """What routing did about one answer.

    Three outcomes rather than a boolean, because "no task was created" covers two very
    different situations. An answer nobody needs to look at is the system working; an
    answer that needed a reviewer and could not reach the queue is a gap someone has to
    know about.
    """

    NOT_REQUIRED = "NOT_REQUIRED"
    QUEUED = "QUEUED"
    QUEUE_UNAVAILABLE = "QUEUE_UNAVAILABLE"


class HumanReviewService:
    """Routes answers that are not confidently supported to a human.

    Separate from the grounding decision, and the separation is the point: that service
    decides what the *caller* receives, this one decides what a *reviewer* receives. They
    happen to read the same band, but they answer to different people and will diverge --
    a deployment that wants every answer sampled for review should be able to say so
    without also changing what it sends back.

    **Routing never changes the answer or the confidence score.** It reads them and creates
    work. Nothing here returns an answer, so there is no path by which queueing a review
    could alter what a caller receives.
    """

    def __init__(self, review_queue: ReviewQueuePort) -> None:
        self._review_queue = review_queue

    async def route(
        self,
        question: Question,
        answer: GroundedAnswer,
        confidence: ConfidenceResult,
        evidence: list[RetrievedChunk],
        correlation_id: str,
    ) -> ReviewRoutingOutcome:
        """Creates a review task when the answer was not confidently supported.

        `answer` must be the answer as **generated**, not the one the caller received: a
        LOW-band answer has already been replaced by a refusal downstream, and a review task
        holding that refusal would tell a reviewer nothing about what was withheld.
        """
        if not self._needs_review(confidence):
            return ReviewRoutingOutcome.NOT_REQUIRED

        request = ReviewRequest(
            correlation_id=correlation_id,
            question=question,
            answer_text=answer.answer_text,
            confidence_score=confidence.score,
            confidence_band=confidence.band,
            citations=list(answer.citations),
            evidence=list(evidence),
        )

        try:
            await self._review_queue.enqueue(request)
        except Exception:  # noqa: BLE001 -- deliberately blind; see below
            # A queue that is down must not take the caller's answer with it. The answer was
            # already decided, sent or withheld on its own merits, and turning an internal
            # delivery problem into a failed request would make the system less available
            # for no gain in correctness.
            #
            # Swallowed here rather than at the orchestrator so the failure has a name: the
            # caller of this method learns the task was lost, instead of an exception
            # escaping into a pipeline that has no idea what to do with it.
            #
            # Blind on purpose. The point is not to handle a known set of queue errors but
            # to guarantee that *nothing* an adapter raises can reach the caller's request
            # -- a driver-specific exception from an adapter written next year is exactly
            # the case a narrower clause would miss.
            return ReviewRoutingOutcome.QUEUE_UNAVAILABLE

        return ReviewRoutingOutcome.QUEUED

    @staticmethod
    def _needs_review(confidence: ConfidenceResult) -> bool:
        """Anything the scoring did not call HIGH goes to a human.

        Stated as "not HIGH" rather than as a list of the other bands so that a band added
        later routes to review by default. Getting that wrong in the safe direction costs a
        reviewer's time; getting it wrong the other way means an answer nobody checks.
        """
        return confidence.band is not ConfidenceBand.HIGH
