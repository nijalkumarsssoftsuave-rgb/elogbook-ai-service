from typing import Protocol

from app.domain.models import (
    AudioRequest,
    AuditRecord,
    DetectedLanguage,
    Embedding,
    GenerationRequest,
    GroundedAnswer,
    Question,
    RetrievalCandidates,
    RetrievedChunk,
    SourceSearchRequest,
    Transcript,
)
from app.domain.permission import PermissionScope, SearchScope


class LanguageDetectorPort(Protocol):
    async def detect(self, text: str) -> DetectedLanguage: ...


class SpeechToTextPort(Protocol):
    """Turns recorded audio into text.

    The whole `AudioRequest` is passed rather than raw bytes because a real engine needs
    the surrounding facts too: the declared format to pick a decoder, and the language
    hint to skip its own language-identification pass.
    """

    async def transcribe(self, audio: AudioRequest) -> Transcript: ...


class EmbeddingPort(Protocol):
    async def embed(self, text: str) -> Embedding: ...


class VectorStorePort(Protocol):
    """Dense (vector-similarity) retrieval. `search` rather than `retrieve` — the latter
    is reserved for RetrievalService's business-capability method.
    """

    async def search(
        self, embedding: Embedding, top_k: int = 5, source_id: str | None = None
    ) -> list[RetrievedChunk]: ...


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
        self,
        query_text: str,
        top_k: int = 5,
        language: str | None = None,
        source_id: str | None = None,
    ) -> list[RetrievedChunk]: ...


class PermissionResolverPort(Protocol):
    """Turns a caller's permission scope into the search set they are entitled to.

    Resolution only -- it performs no retrieval and reads no documents. Keeping the
    authorization decision in its own port is what lets the retrieval below it trust the
    scope it is handed instead of re-deciding access per result.
    """

    async def resolve(self, scope: PermissionScope) -> SearchScope: ...


class MultiSourceRetrieverPort(Protocol):
    """Searches every source it is given, with both retrieval methods, and returns the
    merged candidates.

    It performs no authorization -- the sources on the request are already the permitted
    ones -- and no reranking, which stays downstream in the untouched fusion tail.
    """

    async def search(self, request: SourceSearchRequest) -> RetrievalCandidates: ...


class RerankerPort(Protocol):
    async def rerank(
        self, question: Question, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]: ...


class ModelClientPort(Protocol):
    """Answers a question over labelled evidence and returns the raw completion text.

    Takes a structured request rather than a finished prompt: how the instructions are
    worded is model-specific, so the template belongs with the adapter that speaks to that
    model. Parsing the citation markers back out stays in the application layer.
    """

    async def generate(self, request: GenerationRequest) -> str: ...


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
    """Writes the audit trail.

    Takes one assembled record rather than loose arguments, so the next audit field to be
    added changes the record and not this signature.
    """

    async def record(self, record: AuditRecord) -> None: ...
