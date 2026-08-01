from pydantic import BaseModel

from app.api.schemas.qa import QAQueryResponseData
from app.application.dto import TranscribeResultDTO


class TranscriptResponse(BaseModel):
    text: str
    language: str
    confidence: float | None
    duration_seconds: float | None
    model: str


class STTTranscribeResponseData(BaseModel):
    """Both halves of a voice query: what we heard, and the answer to it.

    The answer reuses the QA endpoint's response shape unchanged, so a client that
    already renders a typed answer needs no new rendering code for a spoken one.
    """

    transcript: TranscriptResponse
    answer: QAQueryResponseData

    @classmethod
    def from_dto(cls, result: TranscribeResultDTO) -> "STTTranscribeResponseData":
        return cls(
            transcript=TranscriptResponse(
                text=result.transcript.text,
                language=result.transcript.language,
                confidence=result.transcript.confidence,
                duration_seconds=result.transcript.duration_seconds,
                model=result.transcript.model,
            ),
            answer=QAQueryResponseData.from_dto(result.answer),
        )
