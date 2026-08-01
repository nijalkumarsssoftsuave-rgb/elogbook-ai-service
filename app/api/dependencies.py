from functools import lru_cache

from fastapi import Depends

from app.application.language_detection_service import LanguageDetectionService
from app.application.ports import (
    AuditPort,
    CachePort,
    GuardrailPort,
    LanguageDetectorPort,
    ModelClientPort,
    RetrieverPort,
)
from app.application.qa_service import QAApplicationService
from app.core.config import Settings, get_settings
from app.infrastructure.stubs.audit_stub import AuditStub
from app.infrastructure.stubs.cache_stub import CacheStub
from app.infrastructure.stubs.guardrail_stub import GuardrailStub
from app.infrastructure.stubs.language_detector_stub import LanguageDetectorStub
from app.infrastructure.stubs.model_client_stub import ModelClientStub
from app.infrastructure.stubs.retriever_stub import RetrieverStub


@lru_cache
def get_retriever_port() -> RetrieverPort:
    return RetrieverStub()


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


@lru_cache
def get_language_detector_port() -> LanguageDetectorPort:
    return LanguageDetectorStub()


def get_language_detection_service(
    detector: LanguageDetectorPort = Depends(get_language_detector_port),
    settings: Settings = Depends(get_settings),
) -> LanguageDetectionService:
    return LanguageDetectionService(detector=detector, supported_languages=settings.supported_languages)


def get_qa_service(
    retriever: RetrieverPort = Depends(get_retriever_port),
    model_client: ModelClientPort = Depends(get_model_client_port),
    guardrail: GuardrailPort = Depends(get_guardrail_port),
    cache: CachePort = Depends(get_cache_port),
    audit: AuditPort = Depends(get_audit_port),
    language_detection: LanguageDetectionService = Depends(get_language_detection_service),
) -> QAApplicationService:
    return QAApplicationService(retriever, model_client, guardrail, cache, audit, language_detection)
