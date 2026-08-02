from pydantic import BaseModel, Field

from app.application.dto import QueryResultDTO
from app.domain.query import QueryFilters


class QAQueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    # Optional, and validated by QueryFilters itself rather than restated here: the
    # rules for what a filter may say belong with the type, not with each transport
    # that happens to accept one. An invalid filter fails request validation and comes
    # back as a 422 with the offending field named.
    filters: QueryFilters = Field(default_factory=QueryFilters)


class CitationResponse(BaseModel):
    # citation_id, order and source_id are additions, not replacements: every field a
    # client already reads is still here, so nothing downstream breaks.
    citation_id: str
    order: int
    chunk_id: str
    source_id: str
    document_id: str
    source_title: str
    page_number: int | None = None
    score: float


class QAQueryResponseData(BaseModel):
    answer_text: str
    citations: list[CitationResponse]
    confidence: float | None
    is_grounded: bool
    refused: bool
    cache_hit: bool

    @classmethod
    def from_dto(cls, result: QueryResultDTO) -> "QAQueryResponseData":
        return cls(
            answer_text=result.answer_text,
            citations=[
                CitationResponse(
                    citation_id=citation.citation_id,
                    order=citation.order,
                    chunk_id=citation.chunk_id,
                    source_id=citation.source_id,
                    document_id=citation.document_id,
                    source_title=citation.source_title,
                    page_number=citation.page_number,
                    score=citation.score,
                )
                for citation in result.citations
            ],
            confidence=result.confidence,
            is_grounded=result.is_grounded,
            refused=result.refused,
            cache_hit=result.cache_hit,
        )
