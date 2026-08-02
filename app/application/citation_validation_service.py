from app.domain.models import CitationValidationResult, GeneratedAnswer, RetrievedChunk


class CitationValidationService:
    """Checks that an answer's citations are actually grounded in the retrieved evidence.

    Runs on the model's *references*, before resolution: an answer that cites something we
    never retrieved should be rejected without paying to resolve it, and rejecting it is
    cheaper than unpicking a finished citation.

    Structural validation only — no model, no I/O, so it is synchronous and real rather
    than stubbed. It answers one question: does every citation point at a chunk we really
    retrieved? A future semantic check (does the cited chunk actually support the claim?)
    would need the citation-resolver model and belongs behind its own port.
    """

    def validate(
        self, answer: GeneratedAnswer, retrieved_chunks: list[RetrievedChunk]
    ) -> CitationValidationResult:
        if not answer.citations:
            return CitationValidationResult(is_valid=False, reason="no_citations")

        retrieved_ids = {chunk.chunk_id for chunk in retrieved_chunks}
        unknown_ids = sorted(
            {
                citation.chunk_id
                for citation in answer.citations
                if citation.chunk_id not in retrieved_ids
            }
        )
        if unknown_ids:
            return CitationValidationResult(
                is_valid=False, reason=f"unknown_chunk_ids:{','.join(unknown_ids)}"
            )

        return CitationValidationResult(is_valid=True, reason=None)
