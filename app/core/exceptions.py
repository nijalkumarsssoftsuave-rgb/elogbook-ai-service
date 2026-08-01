from typing import Any

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

from app.api.schemas.envelope import ApiResponse, ErrorDetail
from app.core.correlation import get_correlation_id


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


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return error_response(
        status_code=500,
        code="INTERNAL_ERROR",
        message="An internal error occurred",
    )
