from app.domain.citation import ResolvedCitation
from app.domain.models import GeneratedAnswer, GroundedAnswer, RetrievedChunk


class CitationResolver:
    """Turns the citation references an answer makes into finished citations.

    The "Citation Resolver" stage of the request flow in CLAUDE.md:

        generated citation ids -> retrieved evidence -> resolved citation objects

    It is the last point where both halves are in hand -- the model's references and the
    chunks they point at -- which is exactly why resolution has to happen here rather than
    later: a GroundedAnswer is cached and audited, so when one is read back the evidence is
    long gone.

    **Resolution never searches the corpus.** The only place it looks is the list of chunks
    this answer was generated from, which is why it takes no vector store, no repository and
    no ports at all. A chunk_id the answer cites but retrieval did not return is not
    something to go and fetch: it is a reference to evidence the model was never shown.
    Widening the lookup would also let an answer cite a document the asker is not permitted
    to see, because the permission scope is applied during retrieval and nowhere after it.

    Structural resolution only. Checking whether a cited chunk actually *supports* the claim
    is a semantic question needing a model, and belongs behind its own port.
    """

    def resolve(
        self, generated: GeneratedAnswer, retrieved_chunks: list[RetrievedChunk]
    ) -> GroundedAnswer:
        chunk_index = self._index(retrieved_chunks)

        resolved: list[ResolvedCitation] = []
        for citation in generated.citations:
            chunk = chunk_index.get(citation.chunk_id)
            # A reference with no matching chunk is dropped rather than resolved into a
            # fabricated citation. Generation already filters these out, so reaching this
            # branch means the chunk list changed underneath the answer -- in which case
            # inventing a source would be the worst possible response.
            if chunk is None:
                continue
            resolved.append(self._to_citation(citation.citation_id, citation.order, chunk))

        return GroundedAnswer(
            answer_text=generated.answer_text,
            citations=resolved,
            confidence=generated.confidence,
            is_grounded=True,
        )

    @staticmethod
    def _index(retrieved_chunks: list[RetrievedChunk]) -> dict[str, RetrievedChunk]:
        """Indexes the evidence by chunk id, keeping the first copy of a repeated id.

        Chunks arrive ranked, and the same chunk can appear twice -- returned by two sources
        that share a document, or surviving fusion in duplicate. First wins, so a citation
        reports the score and metadata of the best-ranked copy rather than of whichever one
        happened to come last.
        """
        index: dict[str, RetrievedChunk] = {}
        for chunk in retrieved_chunks:
            index.setdefault(chunk.chunk_id, chunk)
        return index

    @staticmethod
    def _to_citation(citation_id: str, order: int, chunk: RetrievedChunk) -> ResolvedCitation:
        return ResolvedCitation(
            citation_id=citation_id,
            order=order,
            chunk_id=chunk.chunk_id,
            source_id=chunk.metadata.get("source_id", "unknown"),
            document_id=chunk.document_id,
            source_title=chunk.metadata.get("source_title", "Untitled"),
            page_number=chunk.metadata.get("page_number"),
            score=chunk.score,
            # Copied, not shared: a citation outlives the chunk it came from -- it is cached
            # and audited -- so it must not hold a dict someone can mutate underneath it.
            metadata=dict(chunk.metadata),
        )
