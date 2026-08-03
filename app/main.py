from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app.api.v1.qa import router as qa_router
from app.api.v1.stt import router as stt_router
from app.core.correlation import CorrelationIdMiddleware
from app.core.exceptions import (
    feature_disabled_exception_handler,
    invalid_audio_exception_handler,
    transcription_failed_exception_handler,
    unhandled_exception_handler,
    unsupported_language_exception_handler,
    validation_exception_handler,
)
from app.core.security import AuthenticationMiddleware
from app.domain.exceptions import (
    FeatureDisabledError,
    InvalidAudioError,
    TranscriptionFailedError,
    UnsupportedLanguageError,
)


def create_app() -> FastAPI:
    app = FastAPI(title="elogbook-ai-service")

    # Order matters: correlation id must be set before auth so a 401 can carry it.
    app.add_middleware(AuthenticationMiddleware)
    app.add_middleware(CorrelationIdMiddleware)

    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(UnsupportedLanguageError, unsupported_language_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(FeatureDisabledError, feature_disabled_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(InvalidAudioError, invalid_audio_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(TranscriptionFailedError, transcription_failed_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)

    app.include_router(qa_router)
    app.include_router(stt_router)

    return app


app = create_app()
