from fastapi.testclient import TestClient


def test_query_happy_path_returns_dummy_answer(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post(
        "/api/v1/qa/query",
        json={"query": "What happened on the last shift?"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["error"] is None
    assert body["data"]["answer_text"]
    assert isinstance(body["data"]["citations"], list)
    assert body["correlation_id"]
    assert response.headers["X-Correlation-ID"] == body["correlation_id"]


def test_query_echoes_client_supplied_correlation_id(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    headers = {**auth_headers, "X-Correlation-ID": "fixed-correlation-id"}

    response = client.post("/api/v1/qa/query", json={"query": "Any pending actions?"}, headers=headers)

    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] == "fixed-correlation-id"
    assert response.json()["correlation_id"] == "fixed-correlation-id"
