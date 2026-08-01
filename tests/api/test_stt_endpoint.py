from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from tests.conftest import TEST_ALGORITHM, TEST_SECRET

# Arabic appears only as module-level constants so assertion lines stay pure ASCII: a
# failing assertion prints its operands, and this project is developed on a cp1252 console
# where printing Arabic raises.
_ARABIC_ALARM_QUESTION = "ما سبب إنذار الحريق في الوردية الليلية؟"

_ENDPOINT = "/api/v1/stt/transcribe"

# Not real audio -- the stub adapter never decodes it. Its only job is to be a non-empty
# multipart part with a plausible filename and type.
_AUDIO_BYTES = b"RIFF....WAVEfmt fake audio payload"


def _upload(content: bytes = _AUDIO_BYTES, filename: str = "question.wav",
            content_type: str = "audio/wav") -> dict[str, tuple[str, bytes, str]]:
    return {"file": (filename, content, content_type)}


def test_transcribe_returns_the_transcript_and_the_grounded_answer(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.post(_ENDPOINT, files=_upload(), headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["error"] is None

    transcript = body["data"]["transcript"]
    assert transcript["text"] == "What caused the fire alarm during the night shift?"
    assert transcript["language"] == "en"
    assert transcript["model"] == "faster-whisper-large-v3-turbo-stub"

    answer = body["data"]["answer"]
    assert answer["answer_text"]
    assert isinstance(answer["citations"], list)

    assert body["correlation_id"]
    assert response.headers["X-Correlation-ID"] == body["correlation_id"]


def test_the_transcript_is_actually_retrieved_over(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    """The spoken question must reach the same retrieval path a typed one does, not just
    be echoed back: the night-shift alarm question should cite the English alarm entry.
    """
    response = client.post(_ENDPOINT, files=_upload(), headers=auth_headers)

    cited = {citation["chunk_id"] for citation in response.json()["data"]["answer"]["citations"]}
    assert "log-006" in cited


def test_an_arabic_hint_flows_through_to_arabic_retrieval(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.post(
        _ENDPOINT,
        files=_upload(),
        data={"language_hint": "ar"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["transcript"]["text"] == _ARABIC_ALARM_QUESTION
    assert data["transcript"]["language"] == "ar"

    # The pipeline detects Arabic from the transcript itself and retrieves the Arabic
    # corpus -- the hint only chose what the speech model returned.
    cited = {citation["chunk_id"] for citation in data["answer"]["citations"]}
    assert "log-ar-006" in cited
    assert not any(chunk_id.startswith("log-0") for chunk_id in cited)


def test_a_client_supplied_correlation_id_is_echoed(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    headers = {**auth_headers, "X-Correlation-ID": "fixed-correlation-id"}

    response = client.post(_ENDPOINT, files=_upload(), headers=headers)

    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] == "fixed-correlation-id"
    assert response.json()["correlation_id"] == "fixed-correlation-id"


def test_an_unauthenticated_upload_is_rejected(client: TestClient) -> None:
    response = client.post(_ENDPOINT, files=_upload())

    assert response.status_code == 401
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "UNAUTHORIZED"


def test_a_request_with_no_file_part_is_a_validation_error(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.post(_ENDPOINT, data={"language_hint": "en"}, headers=auth_headers)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_a_non_audio_upload_is_rejected(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.post(
        _ENDPOINT,
        files=_upload(filename="notes.txt", content_type="text/plain"),
        headers=auth_headers,
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "INVALID_AUDIO"
    assert error["details"]["reason"] == "unsupported_content_type"
    assert error["details"]["content_type"] == "text/plain"


def test_an_empty_upload_is_rejected(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.post(_ENDPOINT, files=_upload(content=b""), headers=auth_headers)

    assert response.status_code == 400
    assert response.json()["error"]["details"]["reason"] == "empty_audio"


def test_an_unsupported_language_hint_is_rejected(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    """Reuses the query endpoint's language contract rather than inventing a second one."""
    response = client.post(
        _ENDPOINT, files=_upload(), data={"language_hint": "fr"}, headers=auth_headers
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "UNSUPPORTED_LANGUAGE"
    assert error["details"]["language_code"] == "fr"


def test_an_oversized_upload_is_rejected_with_413(
    app: FastAPI, auth_headers: dict[str, str]
) -> None:
    """The limit is lowered through settings rather than by building a real 25 MB body:
    the rule under test is the ceiling, not the machine's ability to buffer megabytes.
    """
    app.dependency_overrides[get_settings] = lambda: Settings(
        service_jwt_secret=TEST_SECRET,
        service_jwt_algorithm=TEST_ALGORITHM,
        stt_max_audio_bytes=8,
    )
    client = TestClient(app)

    response = client.post(_ENDPOINT, files=_upload(), headers=auth_headers)

    assert response.status_code == 413
    details = response.json()["error"]["details"]
    assert details["reason"] == "file_too_large"
    assert details["size_bytes"] == len(_AUDIO_BYTES)
    assert details["max_bytes"] == 8


def test_top_k_out_of_range_is_rejected(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.post(
        _ENDPOINT, files=_upload(), data={"top_k": 999}, headers=auth_headers
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
