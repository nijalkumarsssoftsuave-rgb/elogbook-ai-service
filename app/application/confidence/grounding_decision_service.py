from app.domain.confidence import ConfidenceBand, ConfidenceResult
from app.domain.models import GroundedAnswer


class GroundingDecisionService:
    """Decides whether a scored answer is well enough supported to send.

    The counterpart to ConfidenceScoringService, and deliberately not part of it. Scoring
    answers "how well supported is this?"; this answers "is that enough?". They are
    different kinds of question -- one is a measurement, the other is a policy -- and a
    component that did both could not be re-tuned without re-testing the arithmetic, nor
    re-measured without re-arguing the policy.

    **It contains no scoring logic and no thresholds of its own.** The thresholds are
    configuration, they live on ConfidencePolicy, and the band on a ConfidenceResult is
    already the outcome of applying them. Comparing the raw score again here would be a
    second implementation of the same boundary, free to drift from the first -- so this
    service reads the band and decides what to do about it, which is the only part that is
    genuinely its own.

    A refusal here is not the same event as a refusal for want of evidence. An answer got as
    far as being generated, resolved and validated; it is being withheld because the
    evidence behind it is too thin to stand on. The score travels with the refusal so that
    distinction survives into the audit trail.
    """

    def decide(self, answer: GroundedAnswer, confidence: ConfidenceResult) -> GroundedAnswer:
        """Returns the answer, or a grounded refusal in its place.

        NEEDS_REVIEW returns the answer. The middle band means "usable, but worth a human
        look", and withholding it would make the band indistinguishable from LOW -- the
        review threshold would then be doing nothing that the refusal threshold does not
        already do.
        """
        if confidence.band is ConfidenceBand.LOW:
            return GroundedAnswer.refusal(confidence=confidence.score)
        return answer
