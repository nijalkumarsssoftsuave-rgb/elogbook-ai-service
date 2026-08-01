from typing import Protocol

from app.domain.models import DetectedLanguage, Embedding, GroundedAnswer, Question, RetrievedChunk


class LanguageDetectorPort(Protocol):
    async def detect(self, text: str) -> DetectedLanguage: ...


class EmbeddingPort(Protocol):
    async def embed(self, text: str) -> Embedding: ...


class VectorStorePort(Protocol):
    """Dense (vector-similarity) retrieval. `search` rather than `retrieve` — the latter
    is reserved for RetrievalService's business-capability method.
    """

    async def search(self, embedding: Embedding, top_k: int = 5) -> list[RetrievedChunk]: ...


class KeywordRetrieverPort(Protocol):
    """Sparse (keyword) retrieval, e.g. BM25.

    `language` restricts the search to documents in that language. It exists because a
    mixed-language keyword index genuinely matches across languages on shared tokens --
    ASCII digits in timestamps, Latin acronyms like HVAC -- which would otherwise let
    Arabic documents surface for English questions. Passing it down rather than
    filtering results afterwards is what a real backend (OpenSearch, Milvus) needs in
    order to apply the restriction server-side instead of over-fetching and discarding.
    """

    async def search(
        self, query_text: str, top_k: int = 5, language: str | None = None
    ) -> list[RetrievedChunk]: ...


class RerankerPort(Protocol):
    async def rerank(
        self, question: Question, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]: ...


class ModelClientPort(Protocol):
    """Sends a fully-built prompt to the LLM and returns the raw completion text.
    Prompt construction and citation parsing live in the application layer, not here.
    """

    async def generate(self, prompt: str) -> str: ...


class GuardrailPort(Protocol):
    async def check_input(self, question: Question) -> None: ...

    async def check_retrieved_chunks(
        self, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]: ...

    async def check_output(self, answer: GroundedAnswer) -> GroundedAnswer: ...


class CachePort(Protocol):
    async def get(self, key: str) -> GroundedAnswer | None: ...

    async def set(self, key: str, value: GroundedAnswer, ttl_seconds: int = 300) -> None: ...


class AuditPort(Protocol):
    async def record_query(
        self, question: Question, answer: GroundedAnswer, correlation_id: str
    ) -> None: ...
