import hashlib

from app.application.dto import QueryRequestDTO, QueryResultDTO
from app.application.language_detection_service import LanguageDetectionService
from app.application.ports import (
    AuditPort,
    CachePort,
    GuardrailPort,
    ModelClientPort,
    RetrieverPort,
)
from app.domain.models import Question


class QAApplicationService:
    """Orchestrates a QA query end-to-end, mirroring the request-flow in CLAUDE.md:
    language detection -> guardrail -> retrieval -> LLM -> citation/grounding -> cache + audit.
    """

    def __init__(
        self,
        retriever: RetrieverPort,
        model_client: ModelClientPort,
        guardrail: GuardrailPort,
        cache: CachePort,
        audit: AuditPort,
        language_detection: LanguageDetectionService,
    ) -> None:
        self._retriever = retriever
        self._model_client = model_client
        self._guardrail = guardrail
        self._cache = cache
        self._audit = audit
        self._language_detection = language_detection

    async def execute(self, request: QueryRequestDTO) -> QueryResultDTO:
        detected = await self._language_detection.detect_and_validate(request.query)

        question = Question(
            text=request.query,
            user_id=request.user_id,
            roles=request.roles,
            language=detected.code,
        )
        cache_key = self._build_cache_key(question)

        cached = await self._cache.get(cache_key)
        if cached is not None:
            await self._audit.record_query(question, cached, request.correlation_id)
            return QueryResultDTO.from_domain(cached, cache_hit=True)

        await self._guardrail.check_input(question)
        chunks = await self._retriever.retrieve(question, top_k=request.top_k)
        answer = await self._model_client.generate(question, chunks)
        answer = await self._guardrail.check_output(answer)

        await self._cache.set(cache_key, answer)
        await self._audit.record_query(question, answer, request.correlation_id)
        return QueryResultDTO.from_domain(answer, cache_hit=False)

    @staticmethod
    def _build_cache_key(question: Question) -> str:
        return f"qa:{hashlib.sha256(question.text.encode()).hexdigest()}"
