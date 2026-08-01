from functools import lru_cache

from fastapi import Depends

from app.application.ports import (
    AuditPort,
    CachePort,
    GuardrailPort,
    ModelClientPort,
    RetrieverPort,
)
from app.application.qa_service import QAApplicationService
from app.infrastructure.stubs.audit_stub import AuditStub
from app.infrastructure.stubs.cache_stub import CacheStub
from app.infrastructure.stubs.guardrail_stub import GuardrailStub
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


def get_qa_service(
    retriever: RetrieverPort = Depends(get_retriever_port),
    model_client: ModelClientPort = Depends(get_model_client_port),
    guardrail: GuardrailPort = Depends(get_guardrail_port),
    cache: CachePort = Depends(get_cache_port),
    audit: AuditPort = Depends(get_audit_port),
) -> QAApplicationService:
    return QAApplicationService(retriever, model_client, guardrail, cache, audit)
