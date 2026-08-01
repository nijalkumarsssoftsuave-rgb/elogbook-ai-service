from typing import cast

import jwt
from fastapi import Request
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.core.config import get_settings
from app.core.exceptions import error_response


class RbacContext(BaseModel):
    user_id: str
    roles: list[str] = []


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Verifies the upstream backend's service JWT and attaches RBAC context to request.state."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        auth_header = request.headers.get("Authorization", "")
        scheme, _, token = auth_header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return error_response(401, "UNAUTHORIZED", "Missing or malformed bearer token")

        settings = get_settings()
        try:
            payload = jwt.decode(token, settings.service_jwt_secret, algorithms=[settings.service_jwt_algorithm])
        except jwt.ExpiredSignatureError:
            return error_response(401, "UNAUTHORIZED", "Service token expired")
        except jwt.InvalidTokenError:
            return error_response(401, "UNAUTHORIZED", "Invalid service token")

        user_id = payload.get("sub")
        if not user_id:
            return error_response(401, "UNAUTHORIZED", "Token missing subject claim")

        request.state.rbac_context = RbacContext(user_id=user_id, roles=payload.get("roles", []))
        return await call_next(request)


def get_rbac_context(request: Request) -> RbacContext:
    return cast(RbacContext, request.state.rbac_context)
