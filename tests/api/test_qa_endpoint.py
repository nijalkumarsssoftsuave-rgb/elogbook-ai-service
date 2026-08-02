import pytest
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


# --- ES-337: filter parameters ----------------------------------------------------------------

_FILTERED_BODY = {
    "query": "Show pump failures",
    "filters": {
        "area_ids": ["north"],
        "date_from": "2026-07-01",
        "date_to": "2026-07-31",
    },
}


def test_a_query_without_filters_is_still_accepted(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    """Filters are optional, so every client written before this ticket keeps working."""
    response = client.post(
        "/api/v1/qa/query", json={"query": "What happened last shift?"}, headers=auth_headers
    )

    assert response.status_code == 200


def test_a_query_with_valid_filters_is_accepted(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.post("/api/v1/qa/query", json=_FILTERED_BODY, headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["data"]["answer_text"]


def test_filters_change_which_documents_are_cited(
    app, auth_headers: dict[str, str]
) -> None:
    """The tripwire ES-337 left behind, now firing as intended: filters used to be accepted
    and ignored, and this asserted that. ES-338 applies them, so the same question with a
    filter must reach a different, narrower set of documents.
    """
    unfiltered = TestClient(app).post(
        "/api/v1/qa/query",
        json={"query": "What caused the fire alarm during the night shift?"},
        headers=auth_headers,
    )
    filtered = TestClient(app).post(
        "/api/v1/qa/query",
        json={
            "query": "What caused the fire alarm during the night shift?",
            "filters": {"area_ids": ["south"]},
        },
        headers=auth_headers,
    )

    unfiltered_ids = {c["chunk_id"] for c in unfiltered.json()["data"]["citations"]}
    filtered_ids = {c["chunk_id"] for c in filtered.json()["data"]["citations"]}

    assert filtered_ids != unfiltered_ids
    assert filtered_ids < unfiltered_ids, "a filter narrows; it never reaches new documents"


@pytest.mark.parametrize(
    ("filters", "reason"),
    [
        ({"date_from": "2026-07-31", "date_to": "2026-07-01"}, "backwards date range"),
        ({"area_ids": ["north", ""]}, "blank filter value"),
        ({"date_from": "the first of July"}, "unparseable date"),
        ({"area_id": "north"}, "misspelled filter key"),
        ({"area_ids": "north"}, "a string where a list belongs"),
    ],
)
def test_an_invalid_filter_is_rejected_with_a_validation_error(
    client: TestClient, auth_headers: dict[str, str], filters: dict, reason: str
) -> None:
    response = client.post(
        "/api/v1/qa/query",
        json={"query": "Show pump failures", "filters": filters},
        headers=auth_headers,
    )

    assert response.status_code == 422, reason
    body = response.json()
    assert body["error"]["code"] == "VALIDATION_ERROR"
    # The offending field is named, so a client can fix it without guessing.
    assert body["error"]["details"]["errors"]


def test_an_invalid_filter_is_rejected_before_the_pipeline_runs(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    """Rejection is request validation, so nothing is retrieved, generated or audited for a
    request that was never well formed.
    """
    response = client.post(
        "/api/v1/qa/query",
        json={"query": "Show pump failures", "filters": {"area_ids": [""]}},
        headers=auth_headers,
    )

    assert response.status_code == 422
    assert "data" not in response.json() or response.json()["data"] is None


def test_a_filter_that_matches_nothing_returns_a_grounded_refusal(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    """No evidence survives the filter, so the pipeline has nothing to answer from and gives
    the ordinary refusal -- not an error, and certainly not an unfiltered answer.
    """
    response = client.post(
        "/api/v1/qa/query",
        json={
            "query": "What caused the fire alarm during the night shift?",
            "filters": {"area_ids": ["atlantis"]},
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["refused"] is True
    assert data["citations"] == []
    assert data["confidence"] is None


def test_a_date_filter_reaches_retrieval(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.post(
        "/api/v1/qa/query",
        json={
            "query": "What caused the fire alarm during the night shift?",
            "filters": {"date_from": "2026-08-01", "date_to": "2026-08-31"},
        },
        headers=auth_headers,
    )

    cited = {c["chunk_id"] for c in response.json()["data"]["citations"]}
    assert cited <= {"log-007", "log-008"}


def test_a_tag_filter_reaches_retrieval(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.post(
        "/api/v1/qa/query",
        json={
            "query": "What caused the fire alarm during the night shift?",
            "filters": {"tags": ["alarm"]},
        },
        headers=auth_headers,
    )

    cited = {c["chunk_id"] for c in response.json()["data"]["citations"]}
    assert cited <= {"log-006"}


def test_a_filter_cannot_reach_a_source_the_caller_may_not_read(
    app,
) -> None:
    """The combination rule end to end: a contractor filtering for an area still cannot see
    the incident report that happens to sit in it. A filter narrows; it never widens.
    """
    headers = {"Authorization": f"Bearer {signed_jwt(sub='u1', roles=['contractor'])}"}

    response = TestClient(app).post(
        "/api/v1/qa/query",
        json={
            "query": "What caused the fire alarm during the night shift?",
            "filters": {"area_ids": ["north"]},
        },
        headers=headers,
    )

    cited = {c["chunk_id"] for c in response.json()["data"]["citations"]}
    assert "log-006" not in cited, "log-006 is north, but lives in a source contractors lack"
