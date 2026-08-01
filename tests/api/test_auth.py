from fastapi.testclient import TestClient

from tests.conftest import signed_jwt


def _assert_unauthorized_envelope(response) -> None:
    assert response.status_code == 401
    body = response.json()
    assert body["success"] is False
    assert body["data"] is None
    assert body["error"]["code"] == "UNAUTHORIZED"
    assert body["correlation_id"]
    assert response.headers["X-Correlation-ID"] == body["correlation_id"]


def test_missing_authorization_header_is_rejected(client: TestClient) -> None:
    response = client.post("/api/v1/qa/query", json={"query": "hello"})
    _assert_unauthorized_envelope(response)


def test_malformed_scheme_is_rejected(client: TestClient) -> None:
    token = signed_jwt()
    response = client.post(
        "/api/v1/qa/query", json={"query": "hello"}, headers={"Authorization": f"Basic {token}"}
    )
    _assert_unauthorized_envelope(response)


def test_bad_signature_is_rejected(client: TestClient) -> None:
    token = signed_jwt(secret="a-completely-different-signing-secret-value")
    response = client.post(
        "/api/v1/qa/query", json={"query": "hello"}, headers={"Authorization": f"Bearer {token}"}
    )
    _assert_unauthorized_envelope(response)


def test_expired_token_is_rejected(client: TestClient) -> None:
    token = signed_jwt(expires_in_seconds=-60)
    response = client.post(
        "/api/v1/qa/query", json={"query": "hello"}, headers={"Authorization": f"Bearer {token}"}
    )
    _assert_unauthorized_envelope(response)


def test_token_missing_subject_claim_is_rejected(client: TestClient) -> None:
    token = signed_jwt(sub=None)
    response = client.post(
        "/api/v1/qa/query", json={"query": "hello"}, headers={"Authorization": f"Bearer {token}"}
    )
    _assert_unauthorized_envelope(response)
