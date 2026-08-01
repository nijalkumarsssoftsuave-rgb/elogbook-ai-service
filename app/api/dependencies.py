from functools import lru_cache

from fastapi import Depends

from app.application.audit_service import AuditService
from app.application.citation_validation_service import CitationValidationService
from app.application.generation_service import GenerationService
from app.application.guardrail_service import GuardrailService
from app.application.language_detection_service import LanguageDetectionService
from app.application.ports import (
    AuditPort,
    CachePort,
    EmbeddingPort,
    GuardrailPort,
    KeywordRetrieverPort,
    LanguageDetectorPort,
    ModelClientPort,
    RerankerPort,
    VectorStorePort,
)
from app.application.qa_service import QAApplicationService
from app.application.retrieval_service import RetrievalService
from app.core.config import Settings, get_settings
from app.infrastructure.language.script_language_detector import ScriptLanguageDetector
from app.infrastructure.retrieval.bm25_keyword_retriever import BM25KeywordRetriever
from app.infrastructure.stubs.audit_stub import AuditStub
from app.infrastructure.stubs.cache_stub import CacheStub
from app.infrastructure.stubs.embedding_stub import EmbeddingStub
from app.infrastructure.stubs.guardrail_stub import GuardrailStub
from app.infrastructure.stubs.model_client_stub import ModelClientStub
from app.infrastructure.stubs.reranker_stub import RerankerStub
from app.infrastructure.stubs.vector_store_stub import VectorStoreStub

# Ports: one cached factory each. Swapping a stub for a real adapter is a one-line
# change here and nothing downstream moves.


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
    vector_store: VectorStorePort = Depends(get_vector_store_port),
    keyword_retriever: KeywordRetrieverPort = Depends(get_keyword_retriever_port),
    reranker: RerankerPort = Depends(get_reranker_port),
) -> RetrievalService:
    return RetrievalService(embedding, vector_store, keyword_retriever, reranker)


def get_guardrail_service(
    guardrail: GuardrailPort = Depends(get_guardrail_port),
) -> GuardrailService:
    return GuardrailService(guardrail=guardrail)


def get_generation_service(
    model_client: ModelClientPort = Depends(get_model_client_port),
) -> GenerationService:
    return GenerationService(model_client=model_client)


def get_citation_validation_service() -> CitationValidationService:
    return CitationValidationService()


def get_audit_service(
    cache: CachePort = Depends(get_cache_port),
    audit: AuditPort = Depends(get_audit_port),
) -> AuditService:
    return AuditService(cache=cache, audit=audit)


def get_qa_service(
    language_detection: LanguageDetectionService = Depends(get_language_detection_service),
    guardrail_service: GuardrailService = Depends(get_guardrail_service),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    generation_service: GenerationService = Depends(get_generation_service),
    citation_validation_service: CitationValidationService = Depends(
        get_citation_validation_service
    ),
    audit_service: AuditService = Depends(get_audit_service),
) -> QAApplicationService:
    return QAApplicationService(
        language_detection,
        guardrail_service,
        retrieval_service,
        generation_service,
        citation_validation_service,
        audit_service,
    )
