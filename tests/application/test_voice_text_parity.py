import inspect

from app.application.audit_service import AuditService
from app.application.citation.citation_resolver import CitationResolver
from app.application.citation.citation_validator import CitationValidator
from app.application.confidence.confidence_scoring_service import (
    ConfidenceScoringService,
)
from app.application.dto import QueryRequestDTO, TranscribeRequestDTO
from app.application.guardrail_service import GuardrailService
from app.application.language_detection_service import LanguageDetectionService
from app.application.permission.permission_resolver import PermissionResolver
from app.application.qa.nodes.generation import GenerationNode
from app.application.qa.nodes.retrieval import RetrievalNode
from app.application.qa_service import QAApplicationService
from app.application.retrieval_service import RetrievalService
from app.application.stt_service import STTApplicationService
from app.application.transcription_service import TranscriptionService
from app.domain.models import GroundedAnswer
from app.domain.permission import PermissionScope
from app.infrastructure.language.script_language_detector import ScriptLanguageDetector
from app.infrastructure.retrieval.bm25_keyword_retriever import BM25KeywordRetriever
from app.infrastructure.retrieval.multi_source_retriever import MultiSourceRetriever
from app.infrastructure.retrieval.permission_catalogue import ROLE_GRANTS, SOURCE_CATALOGUE
from app.infrastructure.stubs.audit_stub import AuditStub
from app.infrastructure.stubs.embedding_stub import EmbeddingStub
from app.infrastructure.stubs.guardrail_stub import GuardrailStub
from app.infrastructure.stubs.model_client_stub import ModelClientStub
from app.infrastructure.stubs.reranker_stub import RerankerStub
from app.infrastructure.stubs.speech_to_text_stub import SpeechToTextStub
from app.infrastructure.stubs.vector_store_stub import VectorStoreStub
from tests.conftest import DEFAULT_CONFIDENCE_POLICY

SUPPORTED_LANGUAGES = ["en", "ar"]

# Retrieval fails closed since ES-327; both paths carry the same scope so the comparison
# is between two real answers rather than two refusals.
VIEWER = PermissionScope.from_roles(["viewer"])
ALLOWED_CONTENT_TYPES = ["audio/wav"]
MAX_AUDIO_BYTES = 1_000_000


class RecordingCache:
    """A real read-through cache, unlike CacheStub which never hits.

    Needed because the cache key is derived from the question text
    (`AuditService._cache_key`), which makes a cache hit the sharpest available evidence
    that the voice and text paths built the *same* question -- not merely questions that
    happened to produce the same answer.
    """

    def __init__(self) -> None:
        self.entries: dict[str, GroundedAnswer] = {}

    async def get(self, key: str) -> GroundedAnswer | None:
        return self.entries.get(key)

    async def set(self, key: str, value: GroundedAnswer, ttl_seconds: int = 300) -> None:
        self.entries[key] = value


def _build_qa_service(cache: RecordingCache) -> QAApplicationService:
    """The production object graph, with stubs only where a real component does not exist
    yet. Built by hand rather than through `dependencies.py` because those factories return
    `fastapi.params.Depends` objects outside a request, silently.
    """
    return QAApplicationService(
        LanguageDetectionService(ScriptLanguageDetector(), SUPPORTED_LANGUAGES),
        GuardrailService(GuardrailStub()),
        RetrievalNode(
            PermissionResolver(SOURCE_CATALOGUE, ROLE_GRANTS),
            RetrievalService(
                EmbeddingStub(),
                MultiSourceRetriever(VectorStoreStub(), BM25KeywordRetriever()),
                RerankerStub(),
            ),
        ),
        GenerationNode(ModelClientStub()),
        CitationResolver(),
        CitationValidator(),
        ConfidenceScoringService(DEFAULT_CONFIDENCE_POLICY),
        AuditService(cache, AuditStub()),
    )


def _build_stt_service(qa_service: QAApplicationService) -> STTApplicationService:
    return STTApplicationService(
        TranscriptionService(
            SpeechToTextStub(), ALLOWED_CONTENT_TYPES, MAX_AUDIO_BYTES, SUPPORTED_LANGUAGES
        ),
        qa_service,
    )


def _voice_request(language_hint: str | None = None) -> TranscribeRequestDTO:
    return TranscribeRequestDTO(
        audio_bytes=b"pretend-this-is-a-wav-file",
        filename="question.wav",
        content_type="audio/wav",
        language_hint=language_hint,
        user_id="u1",
        roles=["viewer"],
        correlation_id="cid-voice",
        top_k=5,
        permission_scope=VIEWER,
    )


def _text_request(query: str) -> QueryRequestDTO:
    return QueryRequestDTO(
        query=query,
        user_id="u1",
        roles=["viewer"],
        correlation_id="cid-text",
        top_k=5,
        permission_scope=VIEWER,
    )


async def test_a_spoken_question_hits_the_cache_a_typed_one_warmed() -> None:
    """The strongest available proof that both paths feed the same pipeline.

    Matching answers could be a coincidence -- two different questions can retrieve the
    same evidence. A cache hit cannot: the key is a hash of the question text, so this can
    only pass if the voice path produced byte-identical input to the text path.
    """
    cache = RecordingCache()
    qa_service = _build_qa_service(cache)
    stt_service = _build_stt_service(qa_service)

    # Ask by voice once, only to learn what the engine heard.
    heard = (await stt_service.execute(_voice_request())).transcript.text
    cache.entries.clear()

    typed = await qa_service.execute(_text_request(heard))
    spoken = await stt_service.execute(_voice_request())

    assert typed.cache_hit is False
    assert spoken.answer.cache_hit is True
    assert len(cache.entries) == 1


async def test_voice_and_text_produce_the_same_answer() -> None:
    cache = RecordingCache()
    qa_service = _build_qa_service(cache)
    stt_service = _build_stt_service(qa_service)

    heard = (await stt_service.execute(_voice_request())).transcript.text
    cache.entries.clear()

    spoken = await stt_service.execute(_voice_request())
    cache.entries.clear()
    typed = await qa_service.execute(_text_request(heard))

    # Compared whole rather than field by field: a field added to one path and not the
    # other is exactly the drift this test exists to catch.
    assert spoken.answer == typed


async def test_the_speech_orchestrator_holds_no_pipeline_internals() -> None:
    """ES-325's boundary, made executable: retrieval, generation and reranking stay inside
    the QA capability. This fails the day someone injects RetrievalService here for
    convenience -- which would work, and would quietly give voice queries their own
    pipeline to drift away in.
    """
    collaborators = [
        parameter.annotation
        for name, parameter in inspect.signature(STTApplicationService).parameters.items()
        if name != "self"
    ]

    assert collaborators == [TranscriptionService, QAApplicationService]
