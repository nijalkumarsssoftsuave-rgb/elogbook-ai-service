from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.dependencies import get_stt_service
from app.api.schemas.envelope import ApiResponse
from app.api.schemas.stt import STTTranscribeResponseData
from app.application.dto import TranscribeRequestDTO
from app.application.stt_service import STTApplicationService
from app.application.transcription_service import enforce_size_limit
from app.core.config import Settings, get_settings
from app.core.correlation import get_correlation_id
from app.core.security import RbacContext, get_rbac_context

router = APIRouter(prefix="/api/v1/stt", tags=["stt"])

# What we call an upload that arrives without a filename. Only used for error messages
# and the extension fallback, so a placeholder is harmless.
_UNNAMED_UPLOAD = "upload"


@router.post("/transcribe", response_model=ApiResponse[STTTranscribeResponseData])
async def transcribe(
    file: Annotated[UploadFile, File(description="The recorded question.")],
    language_hint: Annotated[
        str | None, Form(description="Language of the audio, e.g. 'en' or 'ar'.")
    ] = None,
    top_k: Annotated[int, Form(ge=1, le=20)] = 5,
    rbac: RbacContext = Depends(get_rbac_context),
    correlation_id: str = Depends(get_correlation_id),
    settings: Settings = Depends(get_settings),
    stt_service: STTApplicationService = Depends(get_stt_service),
) -> ApiResponse[STTTranscribeResponseData]:
    # Checked from the size the multipart parser already knows, before reading anything,
    # so an oversized upload is refused without being pulled into memory. The same rule
    # runs again inside TranscriptionService against the bytes we actually hold, which is
    # what covers the case where the parser reports no size at all.
    if file.size is not None:
        enforce_size_limit(file.size, settings.stt_max_audio_bytes)

    audio_bytes = await file.read()

    request_dto = TranscribeRequestDTO(
        audio_bytes=audio_bytes,
        filename=file.filename or _UNNAMED_UPLOAD,
        content_type=file.content_type or "",
        language_hint=language_hint,
        user_id=rbac.user_id,
        roles=rbac.roles,
        correlation_id=correlation_id,
        top_k=top_k,
    )
    result = await stt_service.execute(request_dto)
    return ApiResponse.ok(
        data=STTTranscribeResponseData.from_dto(result), correlation_id=correlation_id
    )
