from typing import Protocol

from app.domain.models import GroundedAnswer, Question, RetrievedChunk


class RetrieverPort(Protocol):
    async def retrieve(self, question: Question, top_k: int = 5) -> list[RetrievedChunk]: ...


class ModelClientPort(Protocol):
    async def generate(
        self, question: Question, context_chunks: list[RetrievedChunk]
    ) -> GroundedAnswer: ...


class GuardrailPort(Protocol):
    async def check_input(self, question: Question) -> None: ...

    async def check_output(self, answer: GroundedAnswer) -> GroundedAnswer: ...


class CachePort(Protocol):
    async def get(self, key: str) -> GroundedAnswer | None: ...

    async def set(self, key: str, value: GroundedAnswer, ttl_seconds: int = 300) -> None: ...


class AuditPort(Protocol):
    async def record_query(
        self, question: Question, answer: GroundedAnswer, correlation_id: str
    ) -> None: ...
