from fastapi.testclient import TestClient


def _assert_validation_error_envelope(response) -> None:
    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["data"] is None
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert body["correlation_id"]


def test_empty_query_is_rejected(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post("/api/v1/qa/query", json={"query": ""}, headers=auth_headers)
    _assert_validation_error_envelope(response)


def test_missing_query_field_is_rejected(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post("/api/v1/qa/query", json={}, headers=auth_headers)
    _assert_validation_error_envelope(response)


def test_wrong_query_type_is_rejected(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post("/api/v1/qa/query", json={"query": 12345}, headers=auth_headers)
    _assert_validation_error_envelope(response)


def test_top_k_out_of_range_is_rejected(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post(
        "/api/v1/qa/query", json={"query": "hello", "top_k": 999}, headers=auth_headers
    )
    _assert_validation_error_envelope(response)
