from datetime import UTC, datetime
from typing import Any, ClassVar

from pydantic import BaseModel, Field, field_validator


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
