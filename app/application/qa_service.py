from app.application.audit_service import AuditService
from app.application.citation_validation_service import CitationValidationService
from app.application.dto import QueryRequestDTO, QueryResultDTO
from app.application.generation_service import GenerationService
from app.application.guardrail_service import GuardrailService
from app.application.language_detection_service import LanguageDetectionService
from app.application.retrieval_service import RetrievalService
from app.domain.models import GroundedAnswer, Question, RetrievedChunk


class QAApplicationService:
    """Orchestrates a QA query end-to-end, one step per business capability:

    language detection -> cache lookup -> input guardrail -> retrieval ->
    evidence guardrail -> generate & validate citations (one retry, else refuse) ->
    output guardrail -> audit & cache.

    Each collaborator owns its own details; this service only sequences them.
    """

    # One initial attempt plus exactly one retry, then a safe refusal.
    _MAX_GENERATION_ATTEMPTS = 2

    def __init__(
        self,
        language_detection: LanguageDetectionService,
        guardrail_service: GuardrailService,
        retrieval_service: RetrievalService,
        generation_service: GenerationService,
        citation_validation_service: CitationValidationService,
        audit_service: AuditService,
    ) -> None:
        self._language_detection = language_detection
        self._guardrail_service = guardrail_service
        self._retrieval_service = retrieval_service
        self._generation_service = generation_service
        self._citation_validation_service = citation_validation_service
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
            await self._audit_service.record_cache_hit(question, cached, request.correlation_id)
            return QueryResultDTO.from_domain(cached, cache_hit=True)

        await self._guardrail_service.check_question(question)

        chunks = await self._retrieval_service.retrieve(question, top_k=request.top_k)
        chunks = await self._guardrail_service.screen_retrieved_chunks(chunks)

        answer = await self._generate_and_validate(question, chunks)
        answer = await self._guardrail_service.check_answer(answer)

        await self._audit_service.finalize(question, answer, request.correlation_id)
        return QueryResultDTO.from_domain(answer, cache_hit=False)

    async def _generate_and_validate(
        self, question: Question, chunks: list[RetrievedChunk]
    ) -> GroundedAnswer:
        """Generates an answer and keeps it only if its citations check out.

        The loop lives here rather than inside either service because it is the only
        place holding both: having GenerationService call CitationValidationService (or
        vice versa) would couple two collaborators that are otherwise siblings.
        """
        for _ in range(self._MAX_GENERATION_ATTEMPTS):
            answer = await self._generation_service.generate(question, chunks)
            if self._citation_validation_service.validate(answer, chunks).is_valid:
                return answer
        return GroundedAnswer.refusal()
