from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class DetectedLanguage(BaseModel):
    code: str
    confidence: float


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
    answer_text: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float | None = None
    is_grounded: bool = True
