import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.domain.permission import BaseRole, CustomRole
from app.infrastructure.retrieval.fixture_corpus import INCIDENTS, SAFETY, SHIFT_LOGS
from tests.conftest import signed_jwt

# End-to-end propagation: the roles in the verified token decide which sources retrieval
# is allowed to search, and therefore which documents can be cited. Asserting on the
# citations rather than on an internal object is deliberate -- it is the only place a
# permission failure would actually be visible to a caller.

_QUERY = "What caused the fire alarm during the night shift?"
_AUDIO = {"file": ("question.wav", b"RIFF....WAVEfmt fake audio payload", "audio/wav")}

# log-006 (the night-shift alarm) lives in `incidents`, which a contractor cannot read.
_INCIDENT_CHUNK = "log-006"
_SHIFT_LOG_CHUNKS = {"log-001", "log-003", "log-005"}
# The documents the fixture corpus places in the north area, across all three sources.
_NORTH_CHUNKS = {
    "log-001", "log-003", "log-005", "log-006",
    "log-ar-001", "log-ar-003", "log-ar-005", "log-ar-006",
}


def _headers(*roles: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {signed_jwt(sub='tech-42', roles=list(roles))}"}


def _cited(response) -> set[str]:
    assert response.status_code == 200
    body = response.json()
    # The voice endpoint nests its answer; the query endpoint returns it directly.
    answer = body["data"].get("answer", body["data"])
    return {citation["chunk_id"] for citation in answer["citations"]}


def _ask_text(client: TestClient, *roles: str):
    return client.post("/api/v1/qa/query", json={"query": _QUERY}, headers=_headers(*roles))


def _ask_voice(client: TestClient, *roles: str):
    return client.post("/api/v1/stt/transcribe", files=_AUDIO, headers=_headers(*roles))


def test_a_viewer_reaches_documents_from_every_source(client: TestClient) -> None:
    cited = _cited(_ask_text(client, "viewer"))

    assert _INCIDENT_CHUNK in cited


@pytest.mark.parametrize("role", list(BaseRole))
def test_every_base_role_reaches_documents_from_every_source(
    app: FastAPI, role: BaseRole
) -> None:
    """ES-332 end to end. Admin is the one that changes: it was not in the grant table
    before, so an admin token resolved to no sources and got a refusal -- a base
    operational role that could read less than a contractor.
    """
    cited = _cited(_ask_text(TestClient(app), role))

    assert _INCIDENT_CHUNK in cited
    assert cited & _SHIFT_LOG_CHUNKS


def test_a_contractor_cannot_cite_a_source_they_may_not_read(client: TestClient) -> None:
    """The same question, the same corpus, a narrower token -- and the incident report
    becomes unreachable. Nothing else about the request changes.
    """
    cited = _cited(_ask_text(client, "contractor"))

    assert _INCIDENT_CHUNK not in cited
    assert cited & _SHIFT_LOG_CHUNKS, "a contractor should still reach the shift logs"


def test_an_area_manager_reaches_only_documents_from_their_area(client: TestClient) -> None:
    """ES-333 end to end. A restriction that is not a source restriction: an area manager
    may read every source, so the incident report is not excluded for being an incident --
    it is reachable because it happened in the north.
    """
    cited = _cited(_ask_text(client, CustomRole.AREA_MANAGER))

    assert cited, "an area manager should still reach their own area"
    for chunk_id in cited:
        assert chunk_id in _NORTH_CHUNKS, f"{chunk_id} is outside the north area"


def test_a_base_role_held_alongside_a_restricted_one_lifts_the_restriction(
    client: TestClient,
) -> None:
    """The agreed precedence rule, visible where it actually matters: the same token plus
    `viewer` reaches a southern document the area manager alone cannot see.

    Note what is *not* asserted -- that every document the restricted token cited is still
    cited by the widened one. The entitlement never narrows, and that is pinned on the
    resolved scope in tests/application/test_permission_resolver.py. Citations are a
    different thing: they are the top-k survivors of a ranking, so widening the permitted
    set adds competitors and can push a previously cited document below the cut. Asserting
    it here would be asserting that ranking is stable, which it is not and need not be.
    """
    restricted = _cited(_ask_text(client, CustomRole.AREA_MANAGER))
    widened = _cited(_ask_text(client, BaseRole.VIEWER, CustomRole.AREA_MANAGER))

    assert restricted <= _NORTH_CHUNKS
    assert widened - _NORTH_CHUNKS, "adding a base role should reach beyond the north"


def test_no_citation_ever_comes_from_outside_the_permitted_sources(
    client: TestClient,
) -> None:
    """The strong form: not "the one document we thought of is absent", but "everything
    present is allowed".
    """
    response = _ask_text(client, "contractor")
    body = response.json()

    for citation in body["data"]["citations"]:
        assert not citation["chunk_id"].startswith(("log-002", "log-004", "log-006", "log-007"))


def test_a_token_with_no_recognised_role_gets_a_refusal_not_an_error(
    client: TestClient,
) -> None:
    """Fails closed and stays well-formed: retrieval resolves to no sources, the pipeline
    finds no evidence, and the caller gets the ordinary grounded refusal rather than a 500
    or -- far worse -- an answer built from documents they cannot read.
    """
    response = _ask_text(client, "some-unknown-role")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["refused"] is True
    assert data["citations"] == []


def test_a_token_with_no_roles_at_all_retrieves_nothing(client: TestClient) -> None:
    response = _ask_text(client)

    assert response.status_code == 200
    assert response.json()["data"]["refused"] is True


def test_the_scope_reaches_retrieval_on_the_voice_path_too(client: TestClient) -> None:
    """A spoken question is answered from exactly the sources the same caller could have
    reached by typing it -- the ES-325 guarantee has to hold for authorization as well.
    """
    viewer_cited = _cited(_ask_voice(client, "viewer"))
    contractor_cited = _cited(_ask_voice(client, "contractor"))

    assert _INCIDENT_CHUNK in viewer_cited
    assert _INCIDENT_CHUNK not in contractor_cited


@pytest.mark.parametrize("ask", [_ask_text, _ask_voice], ids=["text", "voice"])
def test_both_entry_points_agree_on_what_a_contractor_may_see(
    app: FastAPI, ask
) -> None:
    client = TestClient(app)

    cited = _cited(ask(client, "contractor"))

    assert cited
    assert _INCIDENT_CHUNK not in cited


def test_the_source_catalogue_ids_are_the_ones_the_corpus_uses() -> None:
    """Guards a silent mismatch: a typo in either place would make a source resolve to a
    set of documents that does not exist, which looks exactly like "no results".
    """
    from app.infrastructure.retrieval.fixture_corpus import DEFAULT_FIXTURE_CORPUS
    from app.infrastructure.retrieval.permission_catalogue import SOURCE_CATALOGUE

    corpus_sources = {document.source_id for document in DEFAULT_FIXTURE_CORPUS}

    assert corpus_sources == set(SOURCE_CATALOGUE) == {SHIFT_LOGS, INCIDENTS, SAFETY}
