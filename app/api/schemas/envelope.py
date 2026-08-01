from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class ApiResponse(BaseModel, Generic[T]):
    success: bool
    data: T | None = None
    error: ErrorDetail | None = None
    correlation_id: str

    @classmethod
    def ok(cls, data: T, correlation_id: str) -> "ApiResponse[T]":
        return cls(success=True, data=data, error=None, correlation_id=correlation_id)

    @classmethod
    def fail(cls, error: ErrorDetail, correlation_id: str) -> "ApiResponse[T]":
        return cls(success=False, data=None, error=error, correlation_id=correlation_id)
