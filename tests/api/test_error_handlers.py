import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_stt_service
from app.application.dto import TranscribeRequestDTO, TranscribeResultDTO
from app.domain.exceptions import TranscriptionFailedError, TranscriptionFailureReason

# These handlers are registered in app/main.py and, until this file existed, were only ever
# proven by hand against a running server. They are the paths a caller hits when something
# has gone wrong, which is exactly when a malformed error envelope is least welcome.

_ENDPOINT = "/api/v1/stt/transcribe"
_UPLOAD = {"file": ("question.wav", b"RIFF....WAVEfmt fake audio payload", "audio/wav")}


class FailingSTTService:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def execute(self, request: TranscribeRequestDTO) -> TranscribeResultDTO:
        raise self._error


def _client_that_fails_with(app: FastAPI, error: Exception, **kwargs: bool) -> TestClient:
    app.dependency_overrides[get_stt_service] = lambda: FailingSTTService(error)
    return TestClient(app, **kwargs)


@pytest.mark.parametrize(
    ("reason", "expected_status"),
    [
        (TranscriptionFailureReason.MODEL_UNAVAILABLE, 503),
        (TranscriptionFailureReason.TIMEOUT, 504),
    ],
    ids=["model_unavailable", "timeout"],
)
def test_a_transcription_failure_maps_to_its_own_retryable_status(
    app: FastAPI,
    auth_headers: dict[str, str],
    reason: TranscriptionFailureReason,
    expected_status: int,
) -> None:
    """Both mean "the audio was fine, we were not", so both are retryable -- and the split
    tells a client whether the engine is down or merely slow.
    """
    error = TranscriptionFailedError(reason, "the engine let us down", {"model": "tiny"})
    client = _client_that_fails_with(app, error)

    response = client.post(_ENDPOINT, files=_UPLOAD, headers=auth_headers)

    assert response.status_code == expected_status
    body = response.json()
    assert body["success"] is False
    assert body["data"] is None
    assert body["error"]["code"] == "TRANSCRIPTION_FAILED"
    assert body["error"]["details"] == {"reason": reason.value, "model": "tiny"}
    assert response.headers["X-Correlation-ID"] == body["correlation_id"]


def test_an_unexpected_error_returns_the_envelope_and_leaks_nothing(
    app: FastAPI, auth_headers: dict[str, str]
) -> None:
    """raise_server_exceptions=False makes TestClient behave like a real server rather than
    re-raising, which is the only way to see what a caller would actually receive.
    """
    error = RuntimeError("connection string: postgres://user:hunter2@db/prod")
    client = _client_that_fails_with(app, error, raise_server_exceptions=False)

    response = client.post(_ENDPOINT, files=_UPLOAD, headers=auth_headers)

    assert response.status_code == 500
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "INTERNAL_ERROR"
    # The point of the generic message: an unexpected exception must not spill its innards
    # to a caller. This one carries a password, and none of it may appear in the response.
    assert "hunter2" not in response.text
    assert body["error"]["message"] == "An internal error occurred"
