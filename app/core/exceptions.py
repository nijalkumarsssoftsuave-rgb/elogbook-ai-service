from typing import Any

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

from app.api.schemas.envelope import ApiResponse, ErrorDetail
from app.core.correlation import get_correlation_id
from app.core.feature_flags import Feature
from app.domain.exceptions import (
    AudioRejectionReason,
    FeatureDisabledError,
    InvalidAudioError,
    TranscriptionFailedError,
    TranscriptionFailureReason,
    UnsupportedLanguageError,
)

# An oversized upload gets 413 rather than a plain 400 so a client can tell "this file is
# too big" from "this file is the wrong kind" by status alone, and knows not to retry the
# same bytes. Every other audio rejection is an ordinary bad request.
_AUDIO_REJECTION_STATUS: dict[AudioRejectionReason, int] = {
    AudioRejectionReason.FILE_TOO_LARGE: 413,
}
_DEFAULT_AUDIO_REJECTION_STATUS = 400

# Both mean "the audio was fine, we were not" -- so both are retryable, and the split
# tells a client whether the engine is down or merely slow.
_TRANSCRIPTION_FAILURE_STATUS: dict[TranscriptionFailureReason, int] = {
    TranscriptionFailureReason.MODEL_UNAVAILABLE: 503,
    TranscriptionFailureReason.TIMEOUT: 504,
}


# A switched-off endpoint is *absent*: a caller should not be able to tell a capability
# this deployment does not run from one that was never built. A switched-off request option
# is different -- the endpoint is there, and the caller sent something it will not honour,
# so they are told to change the request rather than left to assume it was applied.
_FEATURE_DISABLED_STATUS: dict[str, int] = {
    Feature.VOICE: 404,
    Feature.ADVANCED_FILTERS: 422,
}
_DEFAULT_FEATURE_DISABLED_STATUS = 404


def error_response(
    status_code: int, code: str, message: str, details: dict[str, Any] | None = None
) -> JSONResponse:
    envelope = ApiResponse[None].fail(
        error=ErrorDetail(code=code, message=message, details=details),
        correlation_id=get_correlation_id(),
    )
    return JSONResponse(status_code=status_code, content=jsonable_encoder(envelope.model_dump()))


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return error_response(
        status_code=422,
        code="VALIDATION_ERROR",
        message="Request validation failed",
        details={"errors": jsonable_encoder(exc.errors())},
    )


async def unsupported_language_exception_handler(
    request: Request, exc: UnsupportedLanguageError
) -> JSONResponse:
    return error_response(
        status_code=400,
        code="UNSUPPORTED_LANGUAGE",
        message=str(exc),
        details={"language_code": exc.language_code, "supported_languages": exc.supported_languages},
    )


async def invalid_audio_exception_handler(
    request: Request, exc: InvalidAudioError
) -> JSONResponse:
    return error_response(
        status_code=_AUDIO_REJECTION_STATUS.get(exc.reason, _DEFAULT_AUDIO_REJECTION_STATUS),
        code="INVALID_AUDIO",
        message=str(exc),
        details={"reason": exc.reason.value, **exc.details},
    )


async def transcription_failed_exception_handler(
    request: Request, exc: TranscriptionFailedError
) -> JSONResponse:
    return error_response(
        status_code=_TRANSCRIPTION_FAILURE_STATUS[exc.reason],
        code="TRANSCRIPTION_FAILED",
        message=str(exc),
        details={"reason": exc.reason.value, **exc.details},
    )


async def feature_disabled_exception_handler(
    request: Request, exc: FeatureDisabledError
) -> JSONResponse:
    return error_response(
        status_code=_FEATURE_DISABLED_STATUS.get(
            exc.feature, _DEFAULT_FEATURE_DISABLED_STATUS
        ),
        code="FEATURE_DISABLED",
        message=str(exc),
        details={"feature": exc.feature},
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return error_response(
        status_code=500,
        code="INTERNAL_ERROR",
        message="An internal error occurred",
    )
