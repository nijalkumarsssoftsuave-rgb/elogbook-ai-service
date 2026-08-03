import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_feature_flag_port
from app.core.feature_flags import FeatureFlags
from app.infrastructure.configuration.feature_flag_provider import FeatureFlagProvider
from app.main import create_app
from tests.conftest import signed_jwt

QUESTION = "What caused the fire alarm during the night shift?"
AUDIO = {"file": ("question.wav", b"RIFF....WAVEfmt fake audio payload", "audio/wav")}
HEADERS = {"Authorization": f"Bearer {signed_jwt(sub='tech-42', roles=['viewer'])}"}


def _client(**flags: bool) -> TestClient:
    """An app whose feature flags are overridden, leaving everything else production-wired.

    Overriding the port rather than the environment keeps each test independent: the
    provider is cached per process, so setting env vars would leak between tests in
    whatever order they happen to run.
    """
    app: FastAPI = create_app()
    app.dependency_overrides[get_feature_flag_port] = lambda: FeatureFlagProvider(
        FeatureFlags(**flags)
    )
    return TestClient(app)


def _ask(client: TestClient, **body) -> dict:
    response = client.post(
        "/api/v1/qa/query", json={"query": QUESTION, **body}, headers=HEADERS
    )
    return {"status": response.status_code, "body": response.json()}


# --- everything enabled is today's behaviour ----------------------------------------------


def test_with_every_flag_on_the_pipeline_behaves_as_it_always_did() -> None:
    result = _ask(_client())

    assert result["status"] == 200
    data = result["body"]["data"]
    assert data["citations"]
    assert data["confidence"] is not None


# --- voice ---------------------------------------------------------------------------------


def test_voice_disabled_makes_the_speech_endpoint_absent() -> None:
    """A caller should not be able to tell a capability this deployment does not run from
    one that was never built, so a switched-off endpoint is a 404.
    """
    response = _client(voice_enabled=False).post(
        "/api/v1/stt/transcribe", files=AUDIO, headers=HEADERS
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "FEATURE_DISABLED"
    assert response.json()["error"]["details"]["feature"] == "voice"


def test_voice_disabled_leaves_the_typed_endpoint_alone() -> None:
    """The regression that matters: switching off one capability must not touch another."""
    result = _ask(_client(voice_enabled=False))

    assert result["status"] == 200
    assert result["body"]["data"]["citations"]


def test_voice_enabled_still_serves_speech() -> None:
    response = _client().post("/api/v1/stt/transcribe", files=AUDIO, headers=HEADERS)

    assert response.status_code == 200


# --- confidence scoring ----------------------------------------------------------------------


def test_confidence_disabled_returns_an_answer_with_no_score() -> None:
    """Not a zero, and not a guess: the field is null, exactly as it was before scoring
    existed. An answer is still grounded and cited -- only the measurement is gone.
    """
    result = _ask(
        _client(confidence_scoring_enabled=False, human_review_enabled=False)
    )

    assert result["status"] == 200
    data = result["body"]["data"]
    assert data["confidence"] is None
    assert data["refused"] is False
    assert data["citations"]


def test_confidence_disabled_cannot_withhold_an_answer() -> None:
    """With no score there is no grounding decision, so nothing can be refused for being
    weakly supported. A refusal here could only come from the citation checks.
    """
    result = _ask(
        _client(confidence_scoring_enabled=False, human_review_enabled=False)
    )

    assert result["body"]["data"]["refused"] is False


# --- human review ----------------------------------------------------------------------------


def test_review_disabled_still_scores_and_answers() -> None:
    """Review and scoring are separate switches in the one direction that makes sense:
    scoring without review is a measurement nobody acts on, which is a reasonable thing to
    want.
    """
    result = _ask(_client(human_review_enabled=False))

    assert result["status"] == 200
    assert result["body"]["data"]["confidence"] is not None


# --- advanced filters -------------------------------------------------------------------------


def test_filters_disabled_rejects_a_request_that_sends_them() -> None:
    """Refused rather than ignored. Silently dropping the filter would hand back a broader
    result set than was asked for, indistinguishable from a correctly filtered one.
    """
    result = _ask(_client(advanced_filters_enabled=False), filters={"area_ids": ["north"]})

    assert result["status"] == 422
    assert result["body"]["error"]["code"] == "FEATURE_DISABLED"
    assert result["body"]["error"]["details"]["feature"] == "advanced_filters"


def test_filters_disabled_leaves_unfiltered_requests_working() -> None:
    result = _ask(_client(advanced_filters_enabled=False))

    assert result["status"] == 200
    assert result["body"]["data"]["citations"]


def test_an_empty_filters_object_is_not_a_filtered_request() -> None:
    """A client that always sends the key, with nothing in it, has not asked for anything --
    so it is not refused.
    """
    result = _ask(_client(advanced_filters_enabled=False), filters={})

    assert result["status"] == 200


def test_filters_enabled_still_filters() -> None:
    result = _ask(_client(), filters={"area_ids": ["north"]})

    cited = {c["chunk_id"] for c in result["body"]["data"]["citations"]}
    assert cited <= {"log-001", "log-003", "log-005", "log-006"}


# --- everything off ----------------------------------------------------------------------------


ALL_OFF = {
    "voice_enabled": False,
    "confidence_scoring_enabled": False,
    "human_review_enabled": False,
    "advanced_filters_enabled": False,
}


def test_with_every_capability_off_the_core_pipeline_still_answers() -> None:
    """The strongest regression available: retrieval, permission scoping, generation,
    citation resolution and validation are not flagged, so they must survive every switch
    being off.
    """
    result = _ask(_client(**ALL_OFF))

    assert result["status"] == 200
    data = result["body"]["data"]
    assert data["refused"] is False
    assert data["citations"]
    assert data["answer_text"]
    assert data["is_grounded"] is True


def test_permission_scoping_is_not_a_feature_that_can_be_switched_off() -> None:
    """Access control has no flag, and this is the test that says so out loud. A contractor
    still cannot reach the incident report with every switch turned off.
    """
    client = _client(**ALL_OFF)
    headers = {"Authorization": f"Bearer {signed_jwt(sub='u1', roles=['contractor'])}"}

    response = client.post("/api/v1/qa/query", json={"query": QUESTION}, headers=headers)

    cited = {c["chunk_id"] for c in response.json()["data"]["citations"]}
    assert "log-006" not in cited


@pytest.mark.parametrize(
    "flags",
    [
        {"voice_enabled": False},
        {"confidence_scoring_enabled": False, "human_review_enabled": False},
        {"human_review_enabled": False},
        {"advanced_filters_enabled": False},
        ALL_OFF,
    ],
    ids=["voice", "confidence", "review", "filters", "all"],
)
def test_no_combination_of_flags_changes_which_documents_are_cited(flags: dict) -> None:
    """Flags decide which steps run, never what retrieval finds. If a switch moved the
    evidence, it would be changing the answer rather than the pipeline.
    """
    baseline = _ask(_client())["body"]["data"]["citations"]
    flagged = _ask(_client(**flags))["body"]["data"]["citations"]

    assert [c["chunk_id"] for c in flagged] == [c["chunk_id"] for c in baseline]
