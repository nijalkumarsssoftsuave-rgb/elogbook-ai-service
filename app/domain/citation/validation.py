from enum import StrEnum

from pydantic import BaseModel, Field


class CitationFailureReason(StrEnum):
    """Why an answer's citations were rejected.

    An enum rather than free text because the reason is a decision the pipeline routes on
    and the audit trail records. A caller reading `UNRESOLVED_CITATION` back out of an audit
    row six months from now should find the same spelling the code branches on.
    """

    # The answer cited nothing at all. An uncited claim is ungrounded by definition, so
    # this is a rejection rather than a merely unhelpful answer.
    NO_CITATIONS = "NO_CITATIONS"
    # The answer cited something that does not map to the evidence it was generated from.
    UNRESOLVED_CITATION = "UNRESOLVED_CITATION"


class CitationValidationResult(BaseModel):
    """The verdict on one answer's citations: valid, or why not.

    Deliberately inert -- it carries no message and raises nothing. The validator's job ends
    at "these citations do not hold up"; deciding what the caller sees is the orchestrator's,
    and keeping the two apart is what lets the same verdict drive a retry today and a graph
    edge later without the validator knowing which.
    """

    is_valid: bool
    reason: CitationFailureReason | None = None
    # Which citations failed, by the id the answer used. Empty on success, and empty for
    # NO_CITATIONS -- there were no ids to name.
    unresolved_citation_ids: list[str] = Field(default_factory=list)

    @classmethod
    def valid(cls) -> "CitationValidationResult":
        return cls(is_valid=True)

    @classmethod
    def invalid(
        cls, reason: CitationFailureReason, unresolved_citation_ids: list[str] | None = None
    ) -> "CitationValidationResult":
        return cls(
            is_valid=False,
            reason=reason,
            unresolved_citation_ids=unresolved_citation_ids or [],
        )
