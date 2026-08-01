from pydantic import BaseModel, Field

from app.application.dto import QueryResultDTO


class QAQueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class CitationResponse(BaseModel):
    chunk_id: str
    document_id: str
    source_title: str
    page_number: int | None = None
    score: float


class QAQueryResponseData(BaseModel):
    answer_text: str
    citations: list[CitationResponse]
    confidence: float | None
    is_grounded: bool
    cache_hit: bool

    @classmethod
    def from_dto(cls, result: QueryResultDTO) -> "QAQueryResponseData":
        return cls(
            answer_text=result.answer_text,
            citations=[
                CitationResponse(
                    chunk_id=citation.chunk_id,
                    document_id=citation.document_id,
                    source_title=citation.source_title,
                    page_number=citation.page_number,
                    score=citation.score,
                )
                for citation in result.citations
            ],
            confidence=result.confidence,
            is_grounded=result.is_grounded,
            cache_hit=result.cache_hit,
        )
