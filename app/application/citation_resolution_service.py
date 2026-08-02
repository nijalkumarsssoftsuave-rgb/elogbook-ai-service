from app.domain.citation import ResolvedCitation
from app.domain.models import GeneratedAnswer, GroundedAnswer, RetrievedChunk


class CitationResolutionService:
    """Turns the citation references an answer makes into finished citations.

    The "Citation Resolver" stage of the request flow in CLAUDE.md. It is the last point
    where both halves are in hand -- the model's references and the chunks they point at --
    which is exactly why resolution has to happen here rather than later: a GroundedAnswer
    is cached and audited, so when one is read back the evidence is long gone.

    Structural resolution only. Checking whether a cited chunk actually supports the claim
    is a semantic question needing a model, and belongs behind its own port.
    """

    def resolve(
        self, generated: GeneratedAnswer, retrieved_chunks: list[RetrievedChunk]
    ) -> GroundedAnswer:
        chunk_index = {chunk.chunk_id: chunk for chunk in retrieved_chunks}

        resolved: list[ResolvedCitation] = []
        for citation in generated.citations:
            chunk = chunk_index.get(citation.chunk_id)
            # A reference with no matching chunk is dropped rather than resolved into a
            # fabricated citation. Generation already filters these out, so reaching this
            # branch means the chunk list changed underneath the answer -- in which case
            # inventing a source would be the worst possible response.
            if chunk is None:
                continue
            resolved.append(
                ResolvedCitation(
                    citation_id=citation.citation_id,
                    order=citation.order,
                    chunk_id=chunk.chunk_id,
                    source_id=chunk.metadata.get("source_id", "unknown"),
                    document_id=chunk.document_id,
                    source_title=chunk.metadata.get("source_title", "Untitled"),
                    page_number=chunk.metadata.get("page_number"),
                    score=chunk.score,
                    metadata=dict(chunk.metadata),
                )
            )

        return GroundedAnswer(
            answer_text=generated.answer_text,
            citations=resolved,
            confidence=generated.confidence,
            is_grounded=True,
        )
