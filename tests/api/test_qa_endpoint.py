from fastapi.testclient import TestClient

from tests.conftest import signed_jwt


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


def test_citations_are_returned_as_structured_references(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    """ES-329 end to end: a caller receives resolved citations carrying the reference the
    model made (citation_id, order) alongside the source detail resolution supplied.
    """
    response = client.post(
        "/api/v1/qa/query",
        json={"query": "What caused the fire alarm during the night shift?"},
        headers=auth_headers,
    )

    citations = response.json()["data"]["citations"]
    assert citations

    first = citations[0]
    assert first["citation_id"] == "c1"
    assert first["order"] == 1
    assert first["source_id"]
    assert first["source_title"]
    # Every field a client could already read is still present.
    assert {"chunk_id", "document_id", "source_title", "page_number", "score"} <= set(first)

    # Numbering is contiguous and follows the order the model cited them in.
    assert [c["order"] for c in citations] == list(range(1, len(citations) + 1))
    assert [c["citation_id"] for c in citations] == [f"c{i}" for i in range(1, len(citations) + 1)]


def test_a_grounded_answer_carries_a_confidence_score(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    """ES-334 end to end. The field has been on the response since the first ticket and has
    always been null; this is where it starts meaning something.
    """
    response = client.post(
        "/api/v1/qa/query",
        json={"query": "What caused the fire alarm during the night shift?"},
        headers=auth_headers,
    )

    data = response.json()["data"]
    assert data["refused"] is False
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0


def test_the_same_question_scores_the_same_on_every_request(
    app, auth_headers: dict[str, str]
) -> None:
    """Determinism where a caller can actually see it. Asked from two separate clients so
    the second answer is computed rather than served from the first one's cache.
    """
    body = {"query": "What maintenance was carried out on the packaging machine?"}

    scores = {
        TestClient(app).post("/api/v1/qa/query", json=body, headers=auth_headers).json()[
            "data"
        ]["confidence"]
        for _ in range(3)
    }

    assert len(scores) == 1


def test_a_refusal_carries_no_confidence_score(app) -> None:
    """A refusal is not a weakly supported answer -- it is the absence of one, so scoring it
    would invite a caller to compare the two on the same scale.
    """
    headers = {"Authorization": f"Bearer {signed_jwt(sub='u1', roles=['some-unknown-role'])}"}

    response = TestClient(app).post(
        "/api/v1/qa/query", json={"query": "anything at all"}, headers=headers
    )

    data = response.json()["data"]
    assert data["refused"] is True
    assert data["confidence"] is None
