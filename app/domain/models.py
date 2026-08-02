from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, Field, computed_field, field_validator

from app.domain.citation import Citation, ResolvedCitation


class DetectedLanguage(BaseModel):
    code: str
    confidence: float


class Embedding(BaseModel):
    vector: list[float]
    model: str


class Question(BaseModel):
    text: str
    user_id: str
    roles: list[str] = Field(default_factory=list)
    language: str
    asked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    """One piece of evidence as the model sees it.

    Carries the citation id it was labelled with and the text -- and deliberately **not**
    the chunk id. The model has no use for an internal identifier, and withholding it means
    the only thing it can cite is a label we issued: it cannot invent a plausible-looking
    chunk id, and the worst it can do is name a label that was never offered.
    """

    citation_id: str
    text: str


class GenerationRequest(BaseModel):
    """What the model client is asked to answer.

    Structured rather than a finished prompt string, so how that prompt is worded stays
    with the adapter that talks to the model -- phrasing is model-specific, and the layer
    above should not have an opinion about it.
    """

    question_text: str
    language: str
    evidence: list[EvidenceItem] = Field(default_factory=list)


class GeneratedAnswer(BaseModel):
    """What the model said, before its citations have been resolved.

    Distinct from GroundedAnswer, and the distinction is not cosmetic: GroundedAnswer is
    cached and audited, so by the time one is read back the retrieved chunks are long gone
    and there is nothing left to resolve references against. Resolution therefore has to
    happen before an answer becomes groundable, which means the two stages need two types.
    """

    answer_text: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float | None = None


class GroundedAnswer(BaseModel):
    """The finished answer: citations resolved, safe to cache, audit and return."""

    # What the service says when it cannot ground an answer. Lives here because it is a
    # policy the whole system shares: the prompt instructs the model to use it, the
    # orchestrator falls back to it, and the model stub returns it.
    REFUSAL_TEXT: ClassVar[str] = (
        "I don't have enough information in the available sources to answer that."
    )

    answer_text: str
    citations: list[ResolvedCitation] = Field(default_factory=list)
    confidence: float | None = None
    is_grounded: bool = True
    # Kept distinct from is_grounded: a future guardrail could flag an answer as
    # ungrounded without it being the canned refusal, and only refusals must never
    # be cached.
    refused: bool = False

    @classmethod
    def refusal(cls) -> "GroundedAnswer":
        """The safe answer returned when no grounded answer could be produced."""
        return cls(
            answer_text=cls.REFUSAL_TEXT,
            citations=[],
            confidence=None,
            is_grounded=False,
            refused=True,
        )


class CitationValidationResult(BaseModel):
    is_valid: bool
    reason: str | None = None


class PermissionScope(BaseModel):
    """What the caller is entitled to read.

    Deliberately separate from `Question.roles`, which records *who asked*. Both derive
    from the same token claim today, but they answer different questions and will diverge:
    a scope is where site restrictions and classification levels belong, and adding them
    should not change what a Question is.
    """

    roles: list[str] = Field(default_factory=list)

    @classmethod
    def from_roles(cls, roles: list[str]) -> "PermissionScope":
        return cls(roles=list(roles))


class Source(BaseModel):
    """One logical corpus that retrieval can be pointed at."""

    source_id: str
    display_name: str


class SearchScope(BaseModel):
    """The concrete search set a caller is entitled to: which sources, narrowed by which
    organisational filters.

    The output of permission resolution and the only thing retrieval consults about access.
    Where `PermissionScope` says *who the caller is*, this says *what that entitles them to
    search* -- and separating the two is what lets entitlements grow richer without the
    caller's identity changing shape.

    An empty filter list means **no constraint**, not "nothing allowed": the default scope
    over a source is everything in it. A non-empty one is enforced strictly, so a document
    that does not declare the attribute cannot satisfy it.
    """

    sources: list[Source] = Field(default_factory=list)
    area_ids: list[str] = Field(default_factory=list)
    department_ids: list[str] = Field(default_factory=list)
    company_ids: list[str] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """No sources means nothing to search, whatever the filters say."""
        return not self.sources

    @property
    def source_ids(self) -> list[str]:
        return [source.source_id for source in self.sources]


class SourceSearchRequest(BaseModel):
    """What the permitted sources are asked for.

    The embedding travels with the request because the query is embedded once, upstream,
    however many sources end up being searched.
    """

    query_text: str
    query_embedding: Embedding
    language: str | None = None
    search_scope: SearchScope = Field(default_factory=SearchScope)
    limit_per_source: int = 20


class RetrievalCandidates(BaseModel):
    """Everything the permitted sources returned, ready for fusion.

    The two methods stay in separate lists because Reciprocal Rank Fusion is what
    reconciles them, and that step is deliberately left untouched. `searched_source_ids`
    records which sources were actually queried, which is not the same as which ones
    returned a hit -- a permitted source that matched nothing still needs to be visible.
    """

    searched_source_ids: list[str] = Field(default_factory=list)
    dense: list[RetrievedChunk] = Field(default_factory=list)
    sparse: list[RetrievedChunk] = Field(default_factory=list)


class QueryOrigin(StrEnum):
    TEXT = "text"
    VOICE = "voice"


class QueryProvenance(BaseModel):
    """How a question reached the service.

    Audit metadata only. Nothing in the QA pipeline reads it and nothing may start to --
    a spoken question and a typed one must behave identically, which is the guarantee
    ES-325 established and the voice/text parity tests enforce.

    The two durations are kept separate because they answer different questions:
    `audio_duration_seconds` is how long the recording is, a property of the input;
    `transcription_duration_seconds` is how long we took to process it, a measure of our
    own performance. Collapsing them into one number would make the audit useless for
    either. Both are None for a typed query, and also for the stub speech backend, which
    cannot know a duration without decoding the audio.
    """

    origin: QueryOrigin = QueryOrigin.TEXT
    audio_duration_seconds: float | None = None
    transcription_duration_seconds: float | None = None


class AuditRecord(BaseModel):
    """One row of the audit trail: what was asked, what came back, and how it arrived."""

    correlation_id: str
    question: Question
    answer: GroundedAnswer
    provenance: QueryProvenance = Field(default_factory=QueryProvenance)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def language(self) -> str:
        # Denormalized onto the record because an audit row wants a language column of its
        # own, but computed rather than stored so it can never disagree with the question
        # it describes.
        return self.question.language


class LanguageHint(BaseModel):
    """The language a caller says the audio is in, used to steer the speech model.

    A hint is not a detected language: it is what the caller *claims*. The language the
    QA pipeline acts on is still detected from the resulting transcript.
    """

    code: str

    @field_validator("code")
    @classmethod
    def _normalize(cls, value: str) -> str:
        return value.strip().lower()

    @classmethod
    def from_optional(cls, raw: str | None) -> "LanguageHint | None":
        """Builds a hint from raw form input, treating blank and whitespace as absent.

        A single place to normalize means `" AR "`, `"ar"` and `"AR"` cannot diverge
        between validation and the speech adapter.
        """
        if raw is None or not raw.strip():
            return None
        return cls(code=raw)


class AudioMetadata(BaseModel):
    """What is known about an upload without decoding it.

    Duration is deliberately absent: it cannot be known until something actually decodes
    the audio, so it belongs on the Transcript the speech model returns, not here.
    """

    filename: str
    content_type: str
    size_bytes: int


class AudioRequest(BaseModel):
    # repr suppressed so a multi-megabyte payload never lands in a log line or a
    # validation error message.
    content: bytes = Field(repr=False)
    metadata: AudioMetadata
    language_hint: LanguageHint | None = None


class Transcript(BaseModel):
    text: str
    language: str
    confidence: float | None = None
    # None until a real decoder reports it; the stub adapter cannot know it.
    duration_seconds: float | None = None
    model: str
