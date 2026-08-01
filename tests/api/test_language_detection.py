from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_language_detector_port
from app.domain.models import DetectedLanguage


class UnsupportedLanguageDetectorStub:
    async def detect(self, text: str) -> DetectedLanguage:
        return DetectedLanguage(code="fr", confidence=0.95)


def test_query_rejects_unsupported_language(app: FastAPI, auth_headers: dict[str, str]) -> None:
    app.dependency_overrides[get_language_detector_port] = lambda: UnsupportedLanguageDetectorStub()
    client = TestClient(app)

    response = client.post(
        "/api/v1/qa/query",
        json={"query": "Bonjour, que s'est-il passé ?"},
        headers=auth_headers,
    )

    assert response.status_code == 400
    body = response.json()
    assert body["success"] is False
    assert body["data"] is None
    assert body["error"]["code"] == "UNSUPPORTED_LANGUAGE"
    assert body["error"]["details"]["language_code"] == "fr"
    assert body["error"]["details"]["supported_languages"] == ["en", "ar"]
    assert body["correlation_id"]
    assert response.headers["X-Correlation-ID"] == body["correlation_id"]
