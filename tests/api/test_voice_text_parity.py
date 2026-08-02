import pytest
from fastapi.testclient import TestClient

# Not real audio -- the default stub backend never decodes it.
_AUDIO = {"file": ("question.wav", b"RIFF....WAVEfmt fake audio payload", "audio/wav")}

_STT_ENDPOINT = "/api/v1/stt/transcribe"
_QA_ENDPOINT = "/api/v1/qa/query"


def _ask_by_voice(
    client: TestClient, headers: dict[str, str], language_hint: str | None = None
) -> dict:
    data = {"language_hint": language_hint} if language_hint else None
    response = client.post(_STT_ENDPOINT, files=_AUDIO, data=data, headers=headers)
    assert response.status_code == 200
    return response.json()["data"]


def _ask_by_text(client: TestClient, headers: dict[str, str], query: str) -> dict:
    response = client.post(_QA_ENDPOINT, json={"query": query}, headers=headers)
    assert response.status_code == 200
    return response.json()["data"]


# The transcript is read back out of the voice response rather than hard-coded, so this
# file needs no Arabic literals at all -- assertion lines stay pure ASCII, which matters on
# the cp1252 console this project is developed on.
@pytest.mark.parametrize(
    ("language_hint", "expected_language", "expected_citation_prefix"),
    [
        (None, "en", "log-0"),
        ("ar", "ar", "log-ar-"),
    ],
    ids=["english", "arabic"],
)
def test_a_spoken_question_gets_the_same_answer_as_the_same_typed_question(
    client: TestClient,
    auth_headers: dict[str, str],
    language_hint: str | None,
    expected_language: str,
    expected_citation_prefix: str,
) -> None:
    """ES-325's contract at the level a caller actually sees it.

    Both languages are covered because that is where the two paths could plausibly differ:
    the Arabic route exercises a different corpus and a real decision by the language
    detector, rather than the detector's default.
    """
    spoken = _ask_by_voice(client, auth_headers, language_hint)
    typed = _ask_by_text(client, auth_headers, spoken["transcript"]["text"])

    # Whole-object equality, not field spot-checks: a field added to one response shape and
    # not the other is exactly the drift worth catching.
    assert spoken["answer"] == typed

    # Guards against the comparison being vacuously true -- both paths really did run the
    # pipeline for the language under test.
    assert spoken["transcript"]["language"] == expected_language
    cited = [citation["chunk_id"] for citation in typed["citations"]]
    assert any(chunk_id.startswith(expected_citation_prefix) for chunk_id in cited)


def test_the_voice_answer_carries_no_extra_or_missing_fields(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    """The two responses must stay the same *shape*, not just hold the same values -- a
    client rendering a typed answer should need no new code for a spoken one.
    """
    spoken = _ask_by_voice(client, auth_headers)
    typed = _ask_by_text(client, auth_headers, spoken["transcript"]["text"])

    assert spoken["answer"].keys() == typed.keys()
