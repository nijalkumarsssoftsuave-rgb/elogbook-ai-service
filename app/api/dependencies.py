from functools import lru_cache

from fastapi import Depends

from app.application.audit_service import AuditService
from app.application.citation.citation_resolver import CitationResolver
from app.application.citation.citation_validator import CitationValidator
from app.application.confidence.confidence_scoring_service import (
    ConfidencePolicy,
    ConfidenceScoringService,
)
from app.application.confidence.grounding_decision_service import (
    GroundingDecisionService,
)
from app.application.guardrail_service import GuardrailService
from app.application.language_detection_service import LanguageDetectionService
from app.application.permission.permission_resolver import PermissionResolver
from app.application.ports import (
    AuditPort,
    CachePort,
    EmbeddingPort,
    GuardrailPort,
    KeywordRetrieverPort,
    LanguageDetectorPort,
    ModelClientPort,
    MultiSourceRetrieverPort,
    PermissionResolverPort,
    RerankerPort,
    SpeechToTextPort,
    VectorStorePort,
)
from app.application.qa.nodes.generation import GenerationNode
from app.application.qa.nodes.retrieval import RetrievalNode
from app.application.qa_service import QAApplicationService
from app.application.retrieval_service import RetrievalService
from app.application.review.human_review_service import HumanReviewService
from app.application.review.ports.review_queue_port import ReviewQueuePort
from app.application.stt_service import STTApplicationService
from app.application.transcription_service import TranscriptionService
from app.core.config import Settings, get_settings
from app.infrastructure.language.script_language_detector import ScriptLanguageDetector
from app.infrastructure.model_serving.speech.faster_whisper_adapter import FasterWhisperAdapter
from app.infrastructure.retrieval.bm25_keyword_retriever import BM25KeywordRetriever
from app.infrastructure.retrieval.multi_source_retriever import MultiSourceRetriever
from app.infrastructure.retrieval.permission_catalogue import ROLE_GRANTS, SOURCE_CATALOGUE
from app.infrastructure.review.in_memory_review_queue import InMemoryReviewQueue
from app.infrastructure.stubs.audit_stub import AuditStub
from app.infrastructure.stubs.cache_stub import CacheStub
from app.infrastructure.stubs.embedding_stub import EmbeddingStub
from app.infrastructure.stubs.guardrail_stub import GuardrailStub
from app.infrastructure.stubs.model_client_stub import ModelClientStub
from app.infrastructure.stubs.reranker_stub import RerankerStub
from app.infrastructure.stubs.speech_to_text_stub import SpeechToTextStub
from app.infrastructure.stubs.vector_store_stub import VectorStoreStub

# Ports: one cached factory each. Swapping a stub for a real adapter is a one-line
# change here and nothing downstream moves.

_STUB_BACKEND = "stub"
_FASTER_WHISPER_BACKEND = "faster_whisper"


@lru_cache
def get_language_detector_port() -> LanguageDetectorPort:
    return ScriptLanguageDetector()


@lru_cache
def get_embedding_port() -> EmbeddingPort:
    return EmbeddingStub()


@lru_cache
def get_vector_store_port() -> VectorStorePort:
    return VectorStoreStub()


@lru_cache
def get_keyword_retriever_port() -> KeywordRetrieverPort:
    return BM25KeywordRetriever()


@lru_cache
def get_reranker_port() -> RerankerPort:
    return RerankerStub()


@lru_cache
def get_model_client_port() -> ModelClientPort:
    return ModelClientStub()


@lru_cache
def get_guardrail_port() -> GuardrailPort:
    return GuardrailStub()


@lru_cache
def get_cache_port() -> CachePort:
    return CacheStub()


@lru_cache
def get_audit_port() -> AuditPort:
    return AuditStub()


def build_speech_to_text_port(settings: Settings) -> SpeechToTextPort:
    """Picks the speech adapter named by STT_BACKEND.

    Kept as a plain function taking explicit Settings rather than folded into the cached
    factory below: the factory reads get_settings() internally, so exercising the other
    branch would mean clearing an lru_cache, which leaks across a test session and makes
    test order matter.

    An unknown value raises instead of falling back to the stub. Silently serving canned
    text because someone typoed the backend name is the kind of failure that gets noticed
    in a demo rather than in a log.
    """
    if settings.stt_backend == _STUB_BACKEND:
        return SpeechToTextStub()
    if settings.stt_backend == _FASTER_WHISPER_BACKEND:
        # Constructing this does not load the model -- that happens lazily on the first
        # transcription, so a missing model cannot stop the service from starting.
        return FasterWhisperAdapter(
            model_name=settings.stt_model,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
            timeout_seconds=settings.stt_timeout_seconds,
            local_files_only=settings.stt_local_files_only,
            download_root=settings.stt_download_root,
        )
    raise ValueError(
        f"Unknown STT_BACKEND '{settings.stt_backend}'. "
        f"Expected one of: {_STUB_BACKEND}, {_FASTER_WHISPER_BACKEND}."
    )


@lru_cache
def get_speech_to_text_port() -> SpeechToTextPort:
    return build_speech_to_text_port(get_settings())


@lru_cache
def get_permission_resolver_port() -> PermissionResolverPort:
    return PermissionResolver(SOURCE_CATALOGUE, ROLE_GRANTS)


@lru_cache
def get_multi_source_retriever_port() -> MultiSourceRetrieverPort:
    # Composed from the two single-method retrievers rather than replacing them: dense and
    # keyword search are still distinct capabilities, this adapter just fans them out
    # across the permitted sources.
    return MultiSourceRetriever(get_vector_store_port(), get_keyword_retriever_port())


# Services: composed per request from the cached ports above.


def get_language_detection_service(
    detector: LanguageDetectorPort = Depends(get_language_detector_port),
    settings: Settings = Depends(get_settings),
) -> LanguageDetectionService:
    return LanguageDetectionService(
        detector=detector, supported_languages=settings.supported_languages
    )


def get_retrieval_service(
    embedding: EmbeddingPort = Depends(get_embedding_port),
    multi_source_retriever: MultiSourceRetrieverPort = Depends(get_multi_source_retriever_port),
    reranker: RerankerPort = Depends(get_reranker_port),
) -> RetrievalService:
    return RetrievalService(embedding, multi_source_retriever, reranker)


def get_retrieval_node(
    permission_resolver: PermissionResolverPort = Depends(get_permission_resolver_port),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> RetrievalNode:
    return RetrievalNode(permission_resolver, retrieval_service)


def get_guardrail_service(
    guardrail: GuardrailPort = Depends(get_guardrail_port),
) -> GuardrailService:
    return GuardrailService(guardrail=guardrail)


def get_generation_node(
    model_client: ModelClientPort = Depends(get_model_client_port),
) -> GenerationNode:
    return GenerationNode(model_client=model_client)


def get_citation_resolver() -> CitationResolver:
    return CitationResolver()


def get_citation_validator() -> CitationValidator:
    return CitationValidator()


@lru_cache
def get_confidence_policy() -> ConfidencePolicy:
    """Reads the scoring policy off Settings once.

    The translation lives here because the application layer must not import configuration
    -- the service takes a policy object, and this is the seam that turns env vars into one.
    """
    settings = get_settings()
    return ConfidencePolicy(
        retrieval_weight=settings.confidence_retrieval_weight,
        reranker_weight=settings.confidence_reranker_weight,
        citation_weight=settings.confidence_citation_weight,
        refusal_threshold=settings.confidence_refusal_threshold,
        review_threshold=settings.confidence_review_threshold,
    )


def get_confidence_scoring_service(
    policy: ConfidencePolicy = Depends(get_confidence_policy),
) -> ConfidenceScoringService:
    return ConfidenceScoringService(policy=policy)


def get_grounding_decision_service() -> GroundingDecisionService:
    return GroundingDecisionService()


@lru_cache
def get_review_queue_port() -> ReviewQueuePort:
    # Cached, because an in-memory queue that was rebuilt per request would drop every
    # task it was handed. The durable adapters will not care either way.
    return InMemoryReviewQueue()


def get_human_review_service(
    review_queue: ReviewQueuePort = Depends(get_review_queue_port),
) -> HumanReviewService:
    return HumanReviewService(review_queue=review_queue)


def get_audit_service(
    cache: CachePort = Depends(get_cache_port),
    audit: AuditPort = Depends(get_audit_port),
) -> AuditService:
    return AuditService(cache=cache, audit=audit)


def get_qa_service(
    language_detection: LanguageDetectionService = Depends(get_language_detection_service),
    guardrail_service: GuardrailService = Depends(get_guardrail_service),
    retrieval_node: RetrievalNode = Depends(get_retrieval_node),
    generation_node: GenerationNode = Depends(get_generation_node),
    citation_resolver: CitationResolver = Depends(get_citation_resolver),
    citation_validator: CitationValidator = Depends(get_citation_validator),
    confidence_scoring_service: ConfidenceScoringService = Depends(
        get_confidence_scoring_service
    ),
    grounding_decision_service: GroundingDecisionService = Depends(
        get_grounding_decision_service
    ),
    human_review_service: HumanReviewService = Depends(get_human_review_service),
    audit_service: AuditService = Depends(get_audit_service),
) -> QAApplicationService:
    return QAApplicationService(
        language_detection,
        guardrail_service,
        retrieval_node,
        generation_node,
        citation_resolver,
        citation_validator,
        confidence_scoring_service,
        grounding_decision_service,
        human_review_service,
        audit_service,
    )


def get_transcription_service(
    speech_to_text: SpeechToTextPort = Depends(get_speech_to_text_port),
    settings: Settings = Depends(get_settings),
) -> TranscriptionService:
    return TranscriptionService(
        speech_to_text=speech_to_text,
        allowed_content_types=settings.stt_allowed_content_types,
        max_audio_bytes=settings.stt_max_audio_bytes,
        supported_languages=settings.supported_languages,
    )


def get_stt_service(
    transcription_service: TranscriptionService = Depends(get_transcription_service),
    qa_service: QAApplicationService = Depends(get_qa_service),
) -> STTApplicationService:
    return STTApplicationService(transcription_service, qa_service)
