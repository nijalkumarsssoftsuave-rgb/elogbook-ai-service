from app.application.citation_resolution_service import CitationResolutionService
from app.domain.citation import Citation
from app.domain.models import GeneratedAnswer, RetrievedChunk

CHUNKS = [
    RetrievedChunk(
        chunk_id="log-001",
        document_id="doc-log-001",
        text="Morning shift equipment check completed.",
        score=1.5,
        metadata={
            "source_title": "Morning Shift Equipment Log",
            "source_id": "shift-logs",
            "page_number": 3,
            "language": "en",
        },
    ),
    RetrievedChunk(
        chunk_id="log-006",
        document_id="doc-log-006",
        text="Fire alarm triggered during the night shift.",
        score=1.1,
        metadata={"source_title": "Night Shift Alarm Report", "source_id": "incidents"},
    ),
]


def _generated(*chunk_ids: str) -> GeneratedAnswer:
    return GeneratedAnswer(
        answer_text="an answer",
        citations=[
            Citation(citation_id=f"c{index}", chunk_id=chunk_id, order=index)
            for index, chunk_id in enumerate(chunk_ids, start=1)
        ],
    )


def test_a_reference_resolves_to_the_chunk_it_points_at() -> None:
    grounded = CitationResolutionService().resolve(_generated("log-001"), CHUNKS)

    citation = grounded.citations[0]
    assert citation.citation_id == "c1"
    assert citation.order == 1
    assert citation.chunk_id == "log-001"
    assert citation.source_id == "shift-logs"
    assert citation.document_id == "doc-log-001"
    assert citation.source_title == "Morning Shift Equipment Log"
    assert citation.page_number == 3
    assert citation.score == 1.5


def test_the_chunks_own_metadata_travels_with_the_citation() -> None:
    """So anything the retriever recorded stays reachable without this model growing a
    field for each new key.
    """
    grounded = CitationResolutionService().resolve(_generated("log-001"), CHUNKS)

    assert grounded.citations[0].metadata["language"] == "en"


def test_metadata_is_copied_rather_than_shared_with_the_chunk() -> None:
    """A citation outlives the chunk it came from -- it is cached and audited -- so it must
    not hold a reference someone could mutate underneath it.
    """
    grounded = CitationResolutionService().resolve(_generated("log-001"), CHUNKS)

    grounded.citations[0].metadata["language"] = "mutated"

    assert CHUNKS[0].metadata["language"] == "en"


def test_reference_order_is_preserved() -> None:
    grounded = CitationResolutionService().resolve(_generated("log-006", "log-001"), CHUNKS)

    assert [(c.chunk_id, c.order) for c in grounded.citations] == [
        ("log-006", 1),
        ("log-001", 2),
    ]


def test_a_reference_with_no_matching_chunk_is_dropped_not_fabricated() -> None:
    """Generation already filters these, so reaching this branch means the chunk list moved
    underneath the answer -- and inventing a source would be the worst possible response.
    """
    grounded = CitationResolutionService().resolve(_generated("log-001", "log-999"), CHUNKS)

    assert [c.chunk_id for c in grounded.citations] == ["log-001"]


def test_a_chunk_missing_its_source_id_resolves_to_unknown_rather_than_raising() -> None:
    """Every chunk carries a source_id since ES-327, but a stub or a future adapter might
    not. Losing the whole answer over a missing label would be the wrong trade.
    """
    bare = [RetrievedChunk(chunk_id="x-1", document_id="doc-x", text="t", score=1.0)]

    grounded = CitationResolutionService().resolve(_generated("x-1"), bare)

    assert grounded.citations[0].source_id == "unknown"
    assert grounded.citations[0].source_title == "Untitled"
    assert grounded.citations[0].page_number is None


def test_an_answer_with_no_references_resolves_to_no_citations() -> None:
    grounded = CitationResolutionService().resolve(GeneratedAnswer(answer_text="a"), CHUNKS)

    assert grounded.citations == []
    assert grounded.answer_text == "a"


def test_the_answer_text_and_confidence_survive_resolution() -> None:
    """Resolution is about citations. Touching anything else would make it a second place
    that can change what the model said.
    """
    generated = GeneratedAnswer(
        answer_text="the alarm was a false positive", citations=[], confidence=0.8
    )

    grounded = CitationResolutionService().resolve(generated, CHUNKS)

    assert grounded.answer_text == "the alarm was a false positive"
    assert grounded.confidence == 0.8
    assert grounded.is_grounded is True
    assert grounded.refused is False
