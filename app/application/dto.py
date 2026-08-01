from pydantic import BaseModel, Field

from app.domain.models import Citation, GroundedAnswer


class QueryRequestDTO(BaseModel):
    query: str
    user_id: str
    roles: list[str] = Field(default_factory=list)
    correlation_id: str
    top_k: int = 5


class QueryResultDTO(BaseModel):
    answer_text: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float | None = None
    is_grounded: bool = True
    cache_hit: bool = False

    @classmethod
    def from_domain(cls, answer: GroundedAnswer, cache_hit: bool) -> "QueryResultDTO":
        return cls(
            answer_text=answer.answer_text,
            citations=answer.citations,
            confidence=answer.confidence,
            is_grounded=answer.is_grounded,
            cache_hit=cache_hit,
        )
