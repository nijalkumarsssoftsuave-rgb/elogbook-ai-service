from app.application.audit_service import AuditService
from app.application.citation.citation_resolver import CitationResolver
from app.application.citation.citation_validator import CitationValidator
from app.application.confidence.confidence_scoring_service import (
    ConfidenceScoringService,
)
from app.application.dto import QueryRequestDTO, QueryResultDTO
from app.application.guardrail_service import GuardrailService
from app.application.language_detection_service import LanguageDetectionService
from app.application.qa.nodes.generation import GenerationNode
from app.application.qa.nodes.retrieval import RetrievalNode
from app.domain.models import GroundedAnswer, Question, RetrievedChunk


class QAApplicationService:
    """Orchestrates a QA query end-to-end, one step per business capability:

    language detection -> cache lookup -> input guardrail -> retrieval ->
    evidence guardrail -> generate, validate & resolve citations (one retry, else refuse)
    -> output guardrail -> audit & cache.

    Each collaborator owns its own details; this service only sequences them.
    """

    # One initial attempt plus exactly one retry, then a safe refusal.
    _MAX_GENERATION_ATTEMPTS = 2

    def __init__(
        self,
        language_detection: LanguageDetectionService,
        guardrail_service: GuardrailService,
        retrieval_node: RetrievalNode,
        generation_node: GenerationNode,
        citation_resolver: CitationResolver,
        citation_validator: CitationValidator,
        confidence_scoring_service: ConfidenceScoringService,
        audit_service: AuditService,
    ) -> None:
        self._language_detection = language_detection
        self._guardrail_service = guardrail_service
        self._retrieval_node = retrieval_node
        self._generation_node = generation_node
        self._citation_resolver = citation_resolver
        self._citation_validator = citation_validator
        self._confidence_scoring_service = confidence_scoring_service
        self._audit_service = audit_service

    async def execute(self, request: QueryRequestDTO) -> QueryResultDTO:
        detected = await self._language_detection.detect_and_validate(request.query)

        question = Question(
            text=request.query,
            user_id=request.user_id,
            roles=request.roles,
            language=detected.code,
        )

        cached = await self._audit_service.get_cached(question)
        if cached is not None:
            await self._audit_service.record_cache_hit(
                question, cached, request.correlation_id, request.provenance
            )
            return QueryResultDTO.from_domain(cached, cache_hit=True)

        await self._guardrail_service.check_question(question)

        chunks = await self._retrieval_node.run(
            question, request.permission_scope, top_k=request.top_k
        )
        chunks = await self._guardrail_service.screen_retrieved_chunks(chunks)

        answer = await self._generate_and_validate(question, chunks, request.top_k)
        answer = await self._guardrail_service.check_answer(answer)

        # Provenance is handed to the audit trail and nowhere else -- this service does not
        # branch on how the question arrived, and must not start to.
        await self._audit_service.finalize(
            question, answer, request.correlation_id, request.provenance
        )
        return QueryResultDTO.from_domain(answer, cache_hit=False)

    async def _generate_and_validate(
        self, question: Question, chunks: list[RetrievedChunk], requested_top_k: int
    ) -> GroundedAnswer:
        """Generates an answer, resolves its citations, and keeps it only if they hold up.

                generate -> resolve -> validate -+- valid   -> return the answer
                                                 |
                                                 +- invalid -> retry, then refuse

        Acting on the verdict is this service's job, not the validator's: the validator
        reports, and the orchestrator decides. That split is what lets the same verdict drive
        a retry today and a conditional edge once the flow becomes a LangGraph graph, without
        the validator learning anything about either.

        A failed attempt is discarded whole. Dropping the offending citation and returning
        the rest would leave the claim it was supporting standing with nothing behind it, and
        substituting another citation would be fabricating a source -- so the only honest
        outcomes are a fresh attempt or a grounded refusal.

        The loop lives here rather than inside any one collaborator because it is the only
        place holding all three: having the generation node call the validator (or vice
        versa) would couple collaborators that are otherwise siblings.
        """
        for _ in range(self._MAX_GENERATION_ATTEMPTS):
            generated = await self._generation_node.generate(question, chunks)
            grounded = self._citation_resolver.resolve(generated, chunks)
            validation = self._citation_validator.validate(generated, grounded, chunks)
            if validation.is_valid:
                confidence = self._confidence_scoring_service.score(
                    chunks, requested_top_k, generated, validation
                )
                # Scored, recorded, and not acted on. A LOW band does not turn into a
                # refusal here: the citations validated, so the answer is grounded, and
                # deciding that a weakly supported grounded answer is worse than none is
                # a policy call this ticket does not make.
                return grounded.model_copy(update={"confidence": confidence.score})
        return GroundedAnswer.refusal()
