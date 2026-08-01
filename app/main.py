from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app.api.v1.qa import router as qa_router
from app.core.correlation import CorrelationIdMiddleware
from app.core.exceptions import unhandled_exception_handler, validation_exception_handler
from app.core.security import AuthenticationMiddleware


def create_app() -> FastAPI:
    app = FastAPI(title="elogbook-ai-service")

    # Order matters: correlation id must be set before auth so a 401 can carry it.
    app.add_middleware(AuthenticationMiddleware)
    app.add_middleware(CorrelationIdMiddleware)

    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)

    app.include_router(qa_router)

    return app


app = create_app()
