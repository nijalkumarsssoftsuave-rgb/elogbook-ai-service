import os
from datetime import UTC, datetime, timedelta

os.environ.setdefault("SERVICE_JWT_SECRET", "test-secret-key-for-elogbook-ai-service-unit-tests")
os.environ.setdefault("SERVICE_JWT_ALGORITHM", "HS256")

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import create_app

TEST_SECRET = os.environ["SERVICE_JWT_SECRET"]
TEST_ALGORITHM = os.environ["SERVICE_JWT_ALGORITHM"]


def signed_jwt(
    sub: str | None = "test-user",
    roles: list[str] | None = None,
    expires_in_seconds: int = 300,
    secret: str = TEST_SECRET,
    algorithm: str = TEST_ALGORITHM,
) -> str:
    payload: dict = {
        "exp": datetime.now(UTC) + timedelta(seconds=expires_in_seconds),
    }
    if sub is not None:
        payload["sub"] = sub
    if roles is not None:
        payload["roles"] = roles
    return jwt.encode(payload, secret, algorithm=algorithm)


@pytest.fixture
def app() -> FastAPI:
    return create_app()


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture
def auth_headers() -> dict[str, str]:
    token = signed_jwt(sub="test-user", roles=["viewer"])
    return {"Authorization": f"Bearer {token}"}
