from app.application.citation.citation_validation_service import CitationValidationService
from app.domain.citation import Citation
from app.domain.models import GeneratedAnswer, RetrievedChunk


def _chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk_id="log-001", document_id="doc-1", text="text", score=1.0),
        RetrievedChunk(chunk_id="log-002", document_id="doc-2", text="text", score=0.9),
    ]


def _citation(chunk_id: str, order: int = 1) -> Citation:
    return Citation(citation_id=f"c{order}", chunk_id=chunk_id, order=order)


def test_answer_citing_only_retrieved_chunks_is_valid() -> None:
    answer = GeneratedAnswer(answer_text="a", citations=[_citation("log-001")])

    result = CitationValidationService().validate(answer, _chunks())

    assert result.is_valid is True
    assert result.reason is None


def test_answer_with_no_citations_is_rejected() -> None:
    answer = GeneratedAnswer(answer_text="an uncited claim", citations=[])

    result = CitationValidationService().validate(answer, _chunks())

    assert result.is_valid is False
    assert result.reason == "no_citations"


def test_answer_citing_a_chunk_that_was_never_retrieved_is_rejected() -> None:
    answer = GeneratedAnswer(
        answer_text="a", citations=[_citation("log-001"), _citation("log-999")]
    )

    result = CitationValidationService().validate(answer, _chunks())

    assert result.is_valid is False
    assert result.reason == "unknown_chunk_ids:log-999"


def test_rejection_reason_lists_every_unknown_chunk_id() -> None:
    answer = GeneratedAnswer(
        answer_text="a", citations=[_citation("log-998"), _citation("log-999")]
    )

    result = CitationValidationService().validate(answer, _chunks())

    assert result.reason == "unknown_chunk_ids:log-998,log-999"


def test_answer_is_rejected_when_no_evidence_was_retrieved_at_all() -> None:
    answer = GeneratedAnswer(answer_text="a", citations=[_citation("log-001")])

    result = CitationValidationService().validate(answer, [])

    assert result.is_valid is False
