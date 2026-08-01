from app.application.dto import QueryRequestDTO, TranscribeRequestDTO, TranscribeResultDTO
from app.application.qa_service import QAApplicationService
from app.application.transcription_service import TranscriptionService
from app.domain.models import AudioMetadata, AudioRequest, LanguageHint


class STTApplicationService:
    """Orchestrates a spoken question end-to-end:

    transcribe the audio -> ask the transcript as a normal question -> return both.

    It depends on QAApplicationService as a whole rather than on the six services that
    make it up. The QA pipeline is already one composed capability; re-composing it here
    would duplicate its wiring and let the two entry points drift apart. A change to the
    QA pipeline therefore reaches voice queries for free.
    """

    def __init__(
        self,
        transcription_service: TranscriptionService,
        qa_service: QAApplicationService,
    ) -> None:
        self._transcription_service = transcription_service
        self._qa_service = qa_service

    async def execute(self, request: TranscribeRequestDTO) -> TranscribeResultDTO:
        audio = AudioRequest(
            content=request.audio_bytes,
            metadata=AudioMetadata(
                filename=request.filename,
                content_type=request.content_type,
                size_bytes=len(request.audio_bytes),
            ),
            language_hint=LanguageHint.from_optional(request.language_hint),
        )

        transcript = await self._transcription_service.transcribe(audio)

        # From here the transcript is just a question. It goes through the same language
        # gate, guardrails, retrieval and citation validation as a typed one -- including
        # language detection, which reads the transcript itself rather than trusting the
        # caller's hint.
        answer = await self._qa_service.execute(
            QueryRequestDTO(
                query=transcript.text,
                user_id=request.user_id,
                roles=request.roles,
                correlation_id=request.correlation_id,
                top_k=request.top_k,
            )
        )

        return TranscribeResultDTO(transcript=transcript, answer=answer)
