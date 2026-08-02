from enum import StrEnum

from pydantic import BaseModel, Field

from app.domain.confidence.confidence_signal import ConfidenceSignal


class ConfidenceBand(StrEnum):
    """What the score means in operational terms.

    A band rather than a raw number because the thresholds are the part a deployment tunes,
    and a caller deciding "is 0.58 good?" for itself would be reimplementing that policy
    outside the place that owns it.
    """

    # Below the refusal threshold: the answer is not adequately supported.
    LOW = "LOW"
    # Between the two thresholds: usable, but worth a human look.
    NEEDS_REVIEW = "NEEDS_REVIEW"
    # At or above the review threshold.
    HIGH = "HIGH"


class ConfidenceResult(BaseModel):
    """How well supported an answer is, and what that verdict was built from.

    The signals travel with the score deliberately. A bare 0.42 tells an operator nothing
    they can act on; the same number alongside "2 of 5 requested chunks retrieved" tells
    them the retrieval side is the problem, not the model.

    Like CitationValidationResult, this is inert -- it reports and decides nothing. Acting
    on a LOW band is the orchestrator's business, which is what keeps scoring a measurement
    rather than a second place that can refuse.
    """

    score: float = Field(ge=0.0, le=1.0)
    band: ConfidenceBand
    signals: list[ConfidenceSignal] = Field(default_factory=list)
