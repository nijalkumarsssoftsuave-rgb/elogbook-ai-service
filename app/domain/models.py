from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, Field, computed_field, field_validator


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


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    source_title: str
    page_number: int | None = None
    score: float


class GroundedAnswer(BaseModel):
    # What the service says when it cannot ground an answer. Lives here because it is a
    # policy the whole system shares: the prompt instructs the model to use it, the
    # orchestrator falls back to it, and the model stub returns it.
    REFUSAL_TEXT: ClassVar[str] = (
        "I don't have enough information in the available sources to answer that."
    )

    answer_text: str
    citations: list[Citation] = Field(default_factory=list)
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
