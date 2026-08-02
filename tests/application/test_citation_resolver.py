import inspect

from app.application.citation.citation_resolver import CitationResolver
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


# --- valid chunk ------------------------------------------------------------------------


def test_a_reference_resolves_to_the_chunk_it_points_at() -> None:
    grounded = CitationResolver().resolve(_generated("log-001"), CHUNKS)

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
    grounded = CitationResolver().resolve(_generated("log-001"), CHUNKS)

    assert grounded.citations[0].metadata["language"] == "en"


def test_metadata_is_copied_rather_than_shared_with_the_chunk() -> None:
    """A citation outlives the chunk it came from -- it is cached and audited -- so it must
    not hold a reference someone could mutate underneath it.
    """
    grounded = CitationResolver().resolve(_generated("log-001"), CHUNKS)

    grounded.citations[0].metadata["language"] = "mutated"

    assert CHUNKS[0].metadata["language"] == "en"


def test_reference_order_is_preserved() -> None:
    grounded = CitationResolver().resolve(_generated("log-006", "log-001"), CHUNKS)

    assert [(c.chunk_id, c.order) for c in grounded.citations] == [
        ("log-006", 1),
        ("log-001", 2),
    ]


# --- missing chunk ----------------------------------------------------------------------


def test_a_reference_with_no_matching_chunk_is_dropped_not_fabricated() -> None:
    """Generation already filters these, so reaching this branch means the chunk list moved
    underneath the answer -- and inventing a source would be the worst possible response.
    """
    grounded = CitationResolver().resolve(_generated("log-001", "log-999"), CHUNKS)

    assert [c.chunk_id for c in grounded.citations] == ["log-001"]


def test_resolution_looks_only_at_the_evidence_it_was_handed() -> None:
    """The rule the ticket is built on: a chunk that exists in the corpus but was not
    retrieved for *this* answer stays unresolved. The permission scope is applied during
    retrieval and nowhere after it, so a resolver that could reach past the evidence list
    would be a way to cite a document the asker is not allowed to see.
    """
    grounded = CitationResolver().resolve(_generated("log-042"), CHUNKS)

    assert grounded.citations == []


def test_the_resolver_has_no_way_to_reach_a_corpus() -> None:
    """A structural tripwire for the rule above. The resolver holds no state and takes only
    the answer and the evidence, so there is no store, index or repository it *could*
    search. If a parameter is ever added here, this test is where that argument happens.
    """
    assert CitationResolver().__dict__ == {}
    assert list(inspect.signature(CitationResolver.resolve).parameters) == [
        "self",
        "generated",
        "retrieved_chunks",
    ]


def test_resolving_against_no_evidence_at_all_resolves_nothing() -> None:
    grounded = CitationResolver().resolve(_generated("log-001"), [])

    assert grounded.citations == []


# --- duplicate chunk --------------------------------------------------------------------


def test_two_references_to_the_same_chunk_both_resolve_to_it() -> None:
    """The same chunk can be offered as evidence twice and so be cited under two ids. Both
    markers are in the answer text, so both must resolve or a footnote points at nothing.
    """
    grounded = CitationResolver().resolve(_generated("log-001", "log-001"), CHUNKS)

    assert [(c.citation_id, c.chunk_id, c.order) for c in grounded.citations] == [
        ("c1", "log-001", 1),
        ("c2", "log-001", 2),
    ]
    assert grounded.citations[0].source_title == grounded.citations[1].source_title


def test_a_repeated_chunk_in_the_evidence_resolves_to_its_best_ranked_copy() -> None:
    """Chunks arrive ranked and the same one can appear twice -- returned by two sources
    that share a document. The first copy wins, so the citation reports the score the answer
    was actually ranked on rather than whichever duplicate happened to come last.
    """
    duplicated = [
        CHUNKS[0],
        RetrievedChunk(
            chunk_id="log-001",
            document_id="doc-log-001",
            text="Morning shift equipment check completed.",
            score=0.2,
            metadata={"source_title": "Stale Copy", "source_id": "shift-logs"},
        ),
    ]

    grounded = CitationResolver().resolve(_generated("log-001"), duplicated)

    assert len(grounded.citations) == 1
    assert grounded.citations[0].score == 1.5
    assert grounded.citations[0].source_title == "Morning Shift Equipment Log"


# --- invalid citation -------------------------------------------------------------------


def test_a_citation_with_an_empty_chunk_id_is_dropped() -> None:
    """No chunk has an empty id, so this can only be a malformed reference -- and the safe
    reading of a malformed reference is that it cites nothing.
    """
    answer = GeneratedAnswer(
        answer_text="an answer",
        citations=[Citation(citation_id="c1", chunk_id="", order=1)],
    )

    grounded = CitationResolver().resolve(answer, CHUNKS)

    assert grounded.citations == []


def test_a_chunk_missing_its_source_id_resolves_to_unknown_rather_than_raising() -> None:
    """Every chunk carries a source_id since ES-327, but a stub or a future adapter might
    not. Losing the whole answer over a missing label would be the wrong trade.
    """
    bare = [RetrievedChunk(chunk_id="x-1", document_id="doc-x", text="t", score=1.0)]

    grounded = CitationResolver().resolve(_generated("x-1"), bare)

    assert grounded.citations[0].source_id == "unknown"
    assert grounded.citations[0].source_title == "Untitled"
    assert grounded.citations[0].page_number is None


def test_an_answer_with_no_references_resolves_to_no_citations() -> None:
    grounded = CitationResolver().resolve(GeneratedAnswer(answer_text="a"), CHUNKS)

    assert grounded.citations == []
    assert grounded.answer_text == "a"


def test_the_answer_text_and_confidence_survive_resolution() -> None:
    """Resolution is about citations. Touching anything else would make it a second place
    that can change what the model said.
    """
    generated = GeneratedAnswer(
        answer_text="the alarm was a false positive", citations=[], confidence=0.8
    )

    grounded = CitationResolver().resolve(generated, CHUNKS)

    assert grounded.answer_text == "the alarm was a false positive"
    assert grounded.confidence == 0.8
    assert grounded.is_grounded is True
    assert grounded.refused is False
