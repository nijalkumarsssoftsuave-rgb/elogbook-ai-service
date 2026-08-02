"""Builds the evaluation surfaces.

The object graph is assembled here by hand rather than through `app.api.dependencies`,
for two reasons. First, the composing functions there take their collaborators as
`Depends(...)` defaults, so calling them outside a FastAPI request returns a service
whose ports are `fastapi.params.Depends` objects -- with no error raised. Second, the
port factories are `@lru_cache`d, so importing them would share instances with the rest
of the test suite and make this wiring depend on import order.

`test_surfaces_match_production_wiring` guards the cost of that choice: it asserts each
port built here is the same concrete class production uses, so the evaluation cannot
silently drift into scoring a different pipeline.
"""

from dataclasses import dataclass

from app.application.audit_service import AuditService
from app.application.citation_validation_service import CitationValidationService
from app.application.generation_service import GenerationService
from app.application.guardrail_service import GuardrailService
from app.application.language_detection_service import LanguageDetectionService
from app.application.qa.nodes.retrieval import RetrievalNode
from app.application.qa_service import QAApplicationService
from app.application.retrieval_service import RetrievalService
from app.infrastructure.language.script_language_detector import ScriptLanguageDetector
from app.infrastructure.retrieval.bm25_keyword_retriever import BM25KeywordRetriever
from app.infrastructure.retrieval.fixture_corpus import DEFAULT_FIXTURE_CORPUS, FixtureDocument
from app.infrastructure.retrieval.multi_source_retriever import MultiSourceRetriever
from app.infrastructure.retrieval.source_resolver import RoleBasedSourceResolver
from app.infrastructure.stubs.audit_stub import AuditStub
from app.infrastructure.stubs.cache_stub import CacheStub
from app.infrastructure.stubs.embedding_stub import EmbeddingStub
from app.infrastructure.stubs.guardrail_stub import GuardrailStub
from app.infrastructure.stubs.model_client_stub import ModelClientStub
from app.infrastructure.stubs.reranker_stub import RerankerStub
from app.infrastructure.stubs.vector_store_stub import VectorStoreStub

# Every language the evaluation may ask about, passed explicitly. Deliberately NOT read
# from Settings: `get_settings()` is lru_cached and reads the developer's .env, so an
# evaluation that consulted it would score differently on different machines and would
# tempt someone into calling cache_clear(), which leaks across the pytest session.
EVALUATION_LANGUAGES = ["en", "ar"]


@dataclass(frozen=True)
class EvaluationSurfaces:
    keyword_retriever: BM25KeywordRetriever
    retrieval_service: RetrievalService
    qa_service: QAApplicationService
    corpus: tuple[FixtureDocument, ...]

    @property
    def corpus_chunk_ids(self) -> frozenset[str]:
        return frozenset(document.chunk_id for document in self.corpus)

    @property
    def corpus_languages(self) -> list[str]:
        return sorted({document.language for document in self.corpus})


def build_surfaces(corpus: list[FixtureDocument] | None = None) -> EvaluationSurfaces:
    """A fresh, self-contained pipeline. Nothing here is cached or shared, so two runs in
    one process cannot influence each other. BM25KeywordRetriever indexes the corpus in
    its constructor, so this is the expensive call -- build once per run, not per case.
    """
    documents = list(DEFAULT_FIXTURE_CORPUS if corpus is None else corpus)
    keyword_retriever = BM25KeywordRetriever(corpus=documents)
    retrieval_service = RetrievalService(
        EmbeddingStub(),
        RoleBasedSourceResolver(),
        MultiSourceRetriever(VectorStoreStub(), keyword_retriever),
        RerankerStub(),
    )
    qa_service = QAApplicationService(
        LanguageDetectionService(ScriptLanguageDetector(), list(EVALUATION_LANGUAGES)),
        GuardrailService(GuardrailStub()),
        RetrievalNode(retrieval_service),
        GenerationService(ModelClientStub()),
        CitationValidationService(),
        AuditService(CacheStub(), AuditStub()),
    )
    return EvaluationSurfaces(
        keyword_retriever=keyword_retriever,
        retrieval_service=retrieval_service,
        qa_service=qa_service,
        corpus=tuple(documents),
    )
