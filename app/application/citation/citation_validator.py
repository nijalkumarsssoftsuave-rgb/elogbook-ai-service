from app.domain.citation import CitationFailureReason, CitationValidationResult
from app.domain.models import GeneratedAnswer, GroundedAnswer, RetrievedChunk


class CitationValidator:
    """Checks that every citation an answer makes resolved to evidence that was retrieved.

    Runs *after* resolution, which is the change ES-331 makes. Validating references before
    they are resolved could only ask whether a chunk id looked retrievable; validating after
    asks the question that actually matters -- did each one become a real citation, pointing
    at a real chunk this answer was generated from?

    It only reports. It does not strip the offending citation, does not invent a replacement
    and does not build a refusal: an answer whose citations do not hold up is rejected whole
    by the orchestrator, because an answer with a citation quietly removed still contains the
    claim that citation was supporting.

    Structural validation only -- no model, no I/O, so it is synchronous and real rather than
    stubbed. Whether a cited chunk genuinely *supports* the claim is a semantic question
    needing a model, and belongs behind its own port.
    """

    def validate(
        self,
        generated: GeneratedAnswer,
        grounded: GroundedAnswer,
        retrieved_chunks: list[RetrievedChunk],
    ) -> CitationValidationResult:
        """Compares what the answer cited against what resolution could account for.

        Three arguments because the check spans both sides of resolution: `generated` is
        what the model claimed, `grounded` is what survived, and `retrieved_chunks` is the
        evidence both must agree with. Resolution drops a reference it cannot match, so
        comparing only the resolved list would make a dropped citation invisible -- which is
        exactly the failure this validator exists to catch.
        """
        if not generated.citations:
            return CitationValidationResult.invalid(CitationFailureReason.NO_CITATIONS)

        resolved_ids = {citation.citation_id for citation in grounded.citations}
        dropped = [
            citation.citation_id
            for citation in generated.citations
            if citation.citation_id not in resolved_ids
        ]
        if dropped:
            return CitationValidationResult.invalid(
                CitationFailureReason.UNRESOLVED_CITATION, dropped
            )

        # Belt and braces: today the resolver builds citations only from the chunks it was
        # handed, so this cannot fire. It is checked anyway because this is the component
        # whose whole job is not to take that on trust -- if resolution ever gains a lookup,
        # this is the guard that notices.
        retrieved_ids = {chunk.chunk_id for chunk in retrieved_chunks}
        unretrieved = [
            citation.citation_id
            for citation in grounded.citations
            if citation.chunk_id not in retrieved_ids
        ]
        if unretrieved:
            return CitationValidationResult.invalid(
                CitationFailureReason.UNRESOLVED_CITATION, unretrieved
            )

        return CitationValidationResult.valid()
