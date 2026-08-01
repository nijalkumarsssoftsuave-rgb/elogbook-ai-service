from datetime import UTC, datetime
from typing import Any, ClassVar

from pydantic import BaseModel, Field


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
