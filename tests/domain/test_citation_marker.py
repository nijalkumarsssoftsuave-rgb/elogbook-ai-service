import pytest

from app.domain.citation import (
    CITATION_MARKER_PATTERN,
    citation_marker,
    evidence_citation_id,
)

# The prompt template (infrastructure) writes these markers and the generation node
# (application) reads them back. If the two ever described different syntax the failure
# would be silent -- the model would cite diligently and every citation would vanish. These
# tests pin the round trip so that drift is loud instead.


@pytest.mark.parametrize("position", [1, 2, 9, 10, 137])
def test_what_the_writer_emits_the_reader_finds(position: int) -> None:
    citation_id = evidence_citation_id(position)

    assert CITATION_MARKER_PATTERN.findall(citation_marker(citation_id)) == [citation_id]


def test_evidence_ids_are_numbered_from_one() -> None:
    assert [evidence_citation_id(p) for p in (1, 2, 3)] == ["c1", "c2", "c3"]


def test_markers_are_found_in_running_prose_in_order() -> None:
    text = "Overheating causes it [c2]. Replace the filter monthly [c1]."

    assert CITATION_MARKER_PATTERN.findall(text) == ["c2", "c1"]


@pytest.mark.parametrize(
    "text",
    [
        "[[log-006]]",  # the old chunk-id syntax
        "[log-006]",  # a chunk id in single brackets
        "[C1]",  # wrong case
        "[c]",  # no number
        "[1]",  # no prefix
        "c1",  # no brackets
    ],
)
def test_text_that_is_not_a_citation_marker_is_not_matched(text: str) -> None:
    """Notably the pre-ES-329 `[[chunk_id]]` form: an old cached completion or a stale
    prompt must not be silently reinterpreted as a citation.
    """
    assert CITATION_MARKER_PATTERN.findall(text) == []
