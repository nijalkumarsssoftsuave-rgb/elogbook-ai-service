from app.application.dto import QueryRequestDTO, TranscribeRequestDTO, TranscribeResultDTO
from app.application.qa_service import QAApplicationService
from app.application.transcription_service import TranscriptionService
from app.domain.models import AudioMetadata, AudioRequest, LanguageHint, Transcript


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

        # Normalized once, here, before the transcript is either reported or asked.
        #
        # The guarantee that a spoken question behaves exactly like a typed one belongs to
        # this service, not to every speech engine remembering to tidy its output. Stray
        # whitespace would not change the answer -- BM25 ignores it -- but it changes the
        # cache key, so a voice question would silently miss the entry its identical typed
        # twin just warmed. Copying rather than trimming only on the way into the query
        # keeps the transcript we return and the question we answered the same string.
        transcript = transcript.model_copy(update={"text": transcript.text.strip()})

        answer = await self._qa_service.execute(self._to_query_request(transcript, request))
        return TranscribeResultDTO(transcript=transcript, answer=answer)

    @staticmethod
    def _to_query_request(
        transcript: Transcript, request: TranscribeRequestDTO
    ) -> QueryRequestDTO:
        """The seam where a transcript stops being audio and becomes an ordinary question.

        Past this line nothing about speech travels any further: the standard QA pipeline
        owns the language gate, guardrails, retrieval, generation and citation validation,
        and it cannot tell how the question arrived. Note that the caller's language hint
        is deliberately *not* forwarded -- the pipeline detects language from the
        transcript itself, exactly as it does for typed input.
        """
        return QueryRequestDTO(
            query=transcript.text,
            user_id=request.user_id,
            roles=request.roles,
            correlation_id=request.correlation_id,
            top_k=request.top_k,
        )
