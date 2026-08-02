import hashlib

from app.application.ports import AuditPort, CachePort
from app.domain.models import AuditRecord, GroundedAnswer, QueryProvenance, Question


class AuditService:
    """Owns the read-through cache and the audit trail, which move together: every query
    is audited, but only genuine answers are worth caching.
    """

    def __init__(self, cache: CachePort, audit: AuditPort) -> None:
        self._cache = cache
        self._audit = audit

    async def get_cached(self, question: Question) -> GroundedAnswer | None:
        return await self._cache.get(self._cache_key(question))

    async def record_cache_hit(
        self,
        question: Question,
        answer: GroundedAnswer,
        correlation_id: str,
        provenance: QueryProvenance,
    ) -> None:
        """A cache hit still needs an audit record; there is nothing new to cache."""
        await self._record(question, answer, correlation_id, provenance)

    async def finalize(
        self,
        question: Question,
        answer: GroundedAnswer,
        correlation_id: str,
        provenance: QueryProvenance,
    ) -> None:
        await self._record(question, answer, correlation_id, provenance)
        # Refusals are audited but never cached: a later attempt with better evidence
        # could succeed, and a cached refusal would suppress that indefinitely.
        if not answer.refused:
            await self._cache.set(self._cache_key(question), answer)

    async def _record(
        self,
        question: Question,
        answer: GroundedAnswer,
        correlation_id: str,
        provenance: QueryProvenance,
    ) -> None:
        """The one place that knows how an audit record is assembled."""
        await self._audit.record(
            AuditRecord(
                correlation_id=correlation_id,
                question=question,
                answer=answer,
                provenance=provenance,
            )
        )

    @staticmethod
    def _cache_key(question: Question) -> str:
        # Deliberately derived from the question alone. Provenance must never enter this:
        # the moment it does, a spoken question stops sharing cache entries with the
        # identical typed one and ES-325's guarantee quietly rots.
        return f"qa:{hashlib.sha256(question.text.encode()).hexdigest()}"
