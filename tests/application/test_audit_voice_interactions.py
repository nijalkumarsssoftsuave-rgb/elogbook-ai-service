from app.application.audit_service import AuditService
from app.application.citation.citation_resolver import CitationResolver
from app.application.citation.citation_validation_service import CitationValidationService
from app.application.dto import QueryRequestDTO, TranscribeRequestDTO
from app.application.guardrail_service import GuardrailService
from app.application.language_detection_service import LanguageDetectionService
from app.application.qa.nodes.generation import GenerationNode
from app.application.qa.nodes.retrieval import RetrievalNode
from app.application.qa_service import QAApplicationService
from app.application.retrieval_service import RetrievalService
from app.application.stt_service import STTApplicationService
from app.application.transcription_service import TranscriptionService
from app.domain.models import (
    AudioRequest,
    AuditRecord,
    PermissionScope,
    QueryOrigin,
    Transcript,
)
from app.infrastructure.language.script_language_detector import ScriptLanguageDetector
from app.infrastructure.retrieval.bm25_keyword_retriever import BM25KeywordRetriever
from app.infrastructure.retrieval.multi_source_retriever import MultiSourceRetriever
from app.infrastructure.retrieval.permission_resolver import PermissionResolver
from app.infrastructure.stubs.cache_stub import CacheStub
from app.infrastructure.stubs.embedding_stub import EmbeddingStub
from app.infrastructure.stubs.guardrail_stub import GuardrailStub
from app.infrastructure.stubs.model_client_stub import ModelClientStub
from app.infrastructure.stubs.reranker_stub import RerankerStub
from app.infrastructure.stubs.speech_to_text_stub import SpeechToTextStub
from app.infrastructure.stubs.vector_store_stub import VectorStoreStub

SUPPORTED_LANGUAGES = ["en", "ar"]

# Retrieval fails closed since ES-327, so a request without a scope reaches no sources and
# the pipeline refuses. These tests are about the audit record, not authorization, so they
# carry a scope that resolves to everything -- otherwise every assertion below would pass
# for the wrong reason.
VIEWER = PermissionScope.from_roles(["viewer"])

# The stub speech backend reports no duration -- honestly, since it never decodes the
# audio. A fake engine is layered over it here so `audio_duration_seconds` has a real value
# to travel with; the real adapter is exercised against genuine audio separately.
KNOWN_AUDIO_SECONDS = 11.0


class RecordingAudit:
    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    async def record(self, record: AuditRecord) -> None:
        self.records.append(record)


class TimedSpeechToText:
    """Wraps the stub so the transcript carries a duration a decoder would have reported."""

    def __init__(self, duration_seconds: float | None = KNOWN_AUDIO_SECONDS) -> None:
        self._inner = SpeechToTextStub()
        self._duration = duration_seconds

    async def transcribe(self, audio: AudioRequest) -> Transcript:
        transcript = await self._inner.transcribe(audio)
        return transcript.model_copy(update={"duration_seconds": self._duration})


def _build(audit: RecordingAudit) -> tuple[QAApplicationService, STTApplicationService]:
    qa_service = QAApplicationService(
        LanguageDetectionService(ScriptLanguageDetector(), SUPPORTED_LANGUAGES),
        GuardrailService(GuardrailStub()),
        RetrievalNode(
            PermissionResolver(),
            RetrievalService(
                EmbeddingStub(),
                MultiSourceRetriever(VectorStoreStub(), BM25KeywordRetriever()),
                RerankerStub(),
            ),
        ),
        GenerationNode(ModelClientStub()),
        CitationValidationService(),
        CitationResolver(),
        AuditService(CacheStub(), audit),
    )
    stt_service = STTApplicationService(
        TranscriptionService(
            TimedSpeechToText(), ["audio/wav"], 1_000_000, SUPPORTED_LANGUAGES
        ),
        qa_service,
    )
    return qa_service, stt_service


def _voice_request(
    correlation_id: str = "cid-voice", language_hint: str | None = None
) -> TranscribeRequestDTO:
    return TranscribeRequestDTO(
        audio_bytes=b"pretend-this-is-a-wav-file",
        filename="question.wav",
        content_type="audio/wav",
        language_hint=language_hint,
        user_id="tech-42",
        roles=["viewer"],
        correlation_id=correlation_id,
        top_k=5,
        permission_scope=VIEWER,
    )


async def test_a_voice_query_is_audited_with_origin_language_duration_and_correlation_id() -> (
    None
):
    """ES-326's checklist, on one record."""
    audit = RecordingAudit()
    _, stt_service = _build(audit)

    await stt_service.execute(_voice_request(correlation_id="cid-abc"))

    assert len(audit.records) == 1
    record = audit.records[0]

    assert record.provenance.origin is QueryOrigin.VOICE
    assert record.language == "en"
    assert record.provenance.audio_duration_seconds == KNOWN_AUDIO_SECONDS
    # Wall-clock, so this is asserted on presence and sign rather than a magic number.
    assert record.provenance.transcription_duration_seconds is not None
    assert record.provenance.transcription_duration_seconds >= 0
    assert record.correlation_id == "cid-abc"


async def test_an_arabic_voice_query_audits_the_language_that_was_detected() -> None:
    """The language on the record comes from detection over the transcript, not from the
    caller's hint -- so it reflects what was actually asked.
    """
    audit = RecordingAudit()
    _, stt_service = _build(audit)

    await stt_service.execute(_voice_request(language_hint="ar"))

    assert audit.records[0].language == "ar"


async def test_a_typed_query_is_audited_as_text_with_no_speech_timings() -> None:
    """Without this, `origin` could be a field that is always "voice" and the distinction
    would be worthless. The absent durations are equally the point: a typed query has no
    audio to measure, so inventing zeros would be a lie.
    """
    audit = RecordingAudit()
    qa_service, _ = _build(audit)

    await qa_service.execute(
        QueryRequestDTO(
            query="What caused the fire alarm during the night shift?",
            user_id="tech-42",
            roles=["viewer"],
            correlation_id="cid-typed",
            permission_scope=VIEWER,
        )
    )

    record = audit.records[0]
    assert record.provenance.origin is QueryOrigin.TEXT
    assert record.provenance.audio_duration_seconds is None
    assert record.provenance.transcription_duration_seconds is None
    assert record.language == "en"
    assert record.correlation_id == "cid-typed"


async def test_provenance_changes_the_audit_record_and_nothing_else() -> None:
    """ES-325 said the pipeline cannot tell how a question arrived; ES-326 needs the audit
    trail to know. Both hold only because provenance is *carried* and never *acted on* --
    so the answer to a spoken question must still equal the answer to the typed one.
    """
    audit = RecordingAudit()
    qa_service, stt_service = _build(audit)

    spoken = await stt_service.execute(_voice_request())
    typed = await qa_service.execute(
        QueryRequestDTO(
            query=spoken.transcript.text,
            user_id="tech-42",
            roles=["viewer"],
            correlation_id="cid-voice",
            permission_scope=VIEWER,
        )
    )

    assert spoken.answer == typed

    voice_record, text_record = audit.records
    assert voice_record.provenance.origin is QueryOrigin.VOICE
    assert text_record.provenance.origin is QueryOrigin.TEXT
    # Same question, same answer -- only the provenance differs.
    assert voice_record.question.text == text_record.question.text
    assert voice_record.answer == text_record.answer
