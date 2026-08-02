from app.application.citation.citation_resolver import CitationResolver
from app.application.citation.citation_validator import CitationValidator
from app.domain.citation import (
    Citation,
    CitationFailureReason,
    CitationValidationResult,
    ResolvedCitation,
)
from app.domain.models import GeneratedAnswer, GroundedAnswer, RetrievedChunk


def _chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk_id="log-001", document_id="doc-1", text="text", score=1.0),
        RetrievedChunk(chunk_id="log-002", document_id="doc-2", text="text", score=0.9),
    ]


def _generated(*chunk_ids: str) -> GeneratedAnswer:
    return GeneratedAnswer(
        answer_text="an answer",
        citations=[
            Citation(citation_id=f"c{order}", chunk_id=chunk_id, order=order)
            for order, chunk_id in enumerate(chunk_ids, start=1)
        ],
    )


def _validate(
    generated: GeneratedAnswer, chunks: list[RetrievedChunk] | None = None
) -> CitationValidationResult:
    """Runs the real resolver first, because validation only means something downstream of
    it -- an unresolved citation is by definition one resolution could not account for.
    """
    evidence = _chunks() if chunks is None else chunks
    grounded = CitationResolver().resolve(generated, evidence)
    return CitationValidator().validate(generated, grounded, evidence)


# --- valid citations ---------------------------------------------------------------------


def test_an_answer_whose_citations_all_resolve_is_valid() -> None:
    result = _validate(_generated("log-001"))

    assert result.is_valid is True
    assert result.reason is None
    assert result.unresolved_citation_ids == []


def test_every_citation_of_a_multi_citation_answer_must_resolve() -> None:
    result = _validate(_generated("log-001", "log-002"))

    assert result.is_valid is True


# --- empty citations ---------------------------------------------------------------------


def test_an_answer_that_cites_nothing_is_rejected() -> None:
    """An uncited claim is ungrounded by definition, so this is a rejection rather than a
    merely unhelpful answer.
    """
    result = _validate(GeneratedAnswer(answer_text="an uncited claim"))

    assert result.is_valid is False
    assert result.reason is CitationFailureReason.NO_CITATIONS
    # There were no ids to name, so nothing is listed -- the reason carries the meaning.
    assert result.unresolved_citation_ids == []


# --- a citation that does not map to retrieved evidence -----------------------------------


def test_a_single_unresolved_citation_rejects_the_whole_answer() -> None:
    result = _validate(_generated("log-999"))

    assert result.is_valid is False
    assert result.reason is CitationFailureReason.UNRESOLVED_CITATION
    assert result.unresolved_citation_ids == ["c1"]


def test_every_unresolved_citation_is_named() -> None:
    result = _validate(_generated("log-998", "log-999"))

    assert result.is_valid is False
    assert result.unresolved_citation_ids == ["c1", "c2"]


def test_one_bad_citation_among_good_ones_still_rejects_the_answer() -> None:
    """The point of ES-331: a mixed answer is not partially acceptable. The claim behind the
    unresolved citation is in the text either way, so the answer is rejected whole.
    """
    result = _validate(_generated("log-001", "log-999", "log-002"))

    assert result.is_valid is False
    assert result.reason is CitationFailureReason.UNRESOLVED_CITATION
    assert result.unresolved_citation_ids == ["c2"]


def test_an_answer_is_rejected_when_no_evidence_was_retrieved_at_all() -> None:
    result = _validate(_generated("log-001"), chunks=[])

    assert result.is_valid is False
    assert result.reason is CitationFailureReason.UNRESOLVED_CITATION


def test_a_resolved_citation_pointing_outside_the_evidence_is_rejected() -> None:
    """The resolver cannot currently produce this -- it builds citations only from the
    chunks it was handed. Checked anyway, because not taking that on trust is this
    component's entire job: if resolution ever gains a lookup, this is the guard that
    notices.
    """
    generated = _generated("log-001")
    grounded = GroundedAnswer(
        answer_text="an answer",
        citations=[
            ResolvedCitation(
                citation_id="c1",
                order=1,
                chunk_id="log-777",
                source_id="incidents",
                document_id="doc-777",
                source_title="Never Retrieved",
                score=1.0,
            )
        ],
    )

    result = CitationValidator().validate(generated, grounded, _chunks())

    assert result.is_valid is False
    assert result.reason is CitationFailureReason.UNRESOLVED_CITATION
    assert result.unresolved_citation_ids == ["c1"]


# --- the validator reports, it does not act -----------------------------------------------


def test_the_validator_neither_removes_nor_replaces_a_bad_citation() -> None:
    """It returns a verdict and leaves both the answer and the resolved citations alone.
    Stripping the offending citation would leave the claim it supported standing with
    nothing behind it; substituting another would be fabricating a source.
    """
    generated = _generated("log-001", "log-999")
    grounded = CitationResolver().resolve(generated, _chunks())

    CitationValidator().validate(generated, grounded, _chunks())

    assert [c.citation_id for c in generated.citations] == ["c1", "c2"]
    assert [c.citation_id for c in grounded.citations] == ["c1"]
    assert grounded.answer_text == "an answer"
    assert grounded.refused is False


def test_the_verdict_carries_no_message_for_the_caller() -> None:
    """Building the refusal is the orchestrator's job. Keeping it out of the verdict is what
    lets the same result drive a retry today and a graph edge later.
    """
    result = _validate(_generated("log-999"))

    assert GroundedAnswer.REFUSAL_TEXT not in result.model_dump_json()
