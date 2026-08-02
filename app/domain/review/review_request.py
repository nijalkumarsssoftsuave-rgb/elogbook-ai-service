from pydantic import BaseModel, Field

from app.domain.citation import ResolvedCitation
from app.domain.confidence import ConfidenceBand
from app.domain.models import Question, RetrievedChunk


class ReviewRequest(BaseModel):
    """Everything a human needs to judge one answer, captured at the moment it was judged.

    Self-contained on purpose. A reviewer opens this hours or days later, by which time the
    retrieved chunks are long gone -- so the evidence travels with the request rather than
    being something the queue is expected to fetch back. That is the same reasoning that
    made resolution happen before caching: whoever reads this later cannot re-run the
    pipeline that produced it.

    The answer recorded here is **what the model produced**, not what the caller received.
    A low-confidence answer is withheld and replaced by a refusal, and a review task holding
    only that refusal would tell a reviewer nothing about what was withheld or why it
    scored badly.
    """

    correlation_id: str
    question: Question
    # The answer as generated, before any grounding decision replaced it.
    answer_text: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    confidence_band: ConfidenceBand
    citations: list[ResolvedCitation] = Field(default_factory=list)
    # The chunks the answer was generated from, so a reviewer can check the claims against
    # the same evidence the model had rather than against a fresh search.
    evidence: list[RetrievedChunk] = Field(default_factory=list)

    @property
    def was_withheld(self) -> bool:
        """True when the caller never saw this answer, because it was refused for weakness.

        The first thing a reviewer needs to know: whether they are checking something that
        went out, or something that did not.
        """
        return self.confidence_band is ConfidenceBand.LOW
