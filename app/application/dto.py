from pydantic import BaseModel, Field

from app.domain.citation import ResolvedCitation
from app.domain.models import GroundedAnswer, QueryProvenance, Transcript
from app.domain.permission import PermissionScope
from app.domain.retrieval import RetrievalFilter


class QueryRequestDTO(BaseModel):
    query: str
    user_id: str
    roles: list[str] = Field(default_factory=list)
    correlation_id: str
    top_k: int = 5
    # Carried to the audit trail, never acted on. Defaults to text, so the typed path
    # constructs this exactly as it always did.
    provenance: QueryProvenance = Field(default_factory=QueryProvenance)
    # What the caller may read. Defaults to an empty scope, which resolves to no sources
    # and therefore no evidence -- a request that forgets to populate it fails closed.
    permission_scope: PermissionScope = Field(default_factory=PermissionScope)
    # What the caller asked to narrow the search to. Deliberately separate from
    # permission_scope: one is an entitlement, the other a preference, and a filter can
    # only ever shrink what the scope already permits.
    filters: RetrievalFilter = Field(default_factory=RetrievalFilter)


class QueryResultDTO(BaseModel):
    answer_text: str
    citations: list[ResolvedCitation] = Field(default_factory=list)
    confidence: float | None = None
    is_grounded: bool = True
    refused: bool = False
    cache_hit: bool = False

    @classmethod
    def from_domain(cls, answer: GroundedAnswer, cache_hit: bool) -> "QueryResultDTO":
        return cls(
            answer_text=answer.answer_text,
            citations=answer.citations,
            confidence=answer.confidence,
            is_grounded=answer.is_grounded,
            refused=answer.refused,
            cache_hit=cache_hit,
        )


class TranscribeRequestDTO(BaseModel):
    """A voice question as it arrives from the transport: raw upload facts plus the same
    caller context a typed query carries.
    """

    audio_bytes: bytes = Field(repr=False)
    filename: str
    content_type: str
    language_hint: str | None = None
    user_id: str
    roles: list[str] = Field(default_factory=list)
    correlation_id: str
    top_k: int = 5
    permission_scope: PermissionScope = Field(default_factory=PermissionScope)


class TranscribeResultDTO(BaseModel):
    """What we heard, and the grounded answer to it.

    The transcript is returned alongside the answer rather than discarded: a caller needs
    to show the user what was understood, and a wrong answer to a misheard question is
    otherwise impossible to diagnose.
    """

    transcript: Transcript
    answer: QueryResultDTO
