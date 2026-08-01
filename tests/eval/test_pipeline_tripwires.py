"""Executable documentation of the stub artifacts that shape the evaluation numbers.

These tests are deliberately fragile: each one FAILS when the pipeline improves, and the
failure message says what to do about it. They are tripwires, not regressions -- a red
build here is good news that needs a follow-up action.
"""

from app.application.dto import QueryRequestDTO
from app.domain.models import Question
from tests.eval.surfaces import EvaluationSurfaces

_ENGLISH_QUESTIONS = [
    "What caused the fire alarm during the night shift?",
    "What happened near the loading dock?",
    "Which machine had its drive belt replaced?",
]


async def test_rank_one_of_retrieval_is_always_the_dense_stub(
    evaluation_surfaces: EvaluationSurfaces,
) -> None:
    for text in _ENGLISH_QUESTIONS:
        chunks = await evaluation_surfaces.retrieval_service.retrieve(
            Question(text=text, user_id="tripwire", language="en"), top_k=5
        )
        assert chunks[0].chunk_id == "dense-stub-chunk-1", (
            "Rank 1 is no longer the dense stub, so the vector store may now be real. "
            "Regenerate the baseline and re-derive the retrieval_raw thresholds: k=1 "
            "metrics on that surface are no longer structurally pinned at 0."
        )


async def test_the_service_cannot_refuse_even_with_no_keyword_matches(
    evaluation_surfaces: EvaluationSurfaces,
) -> None:
    query = "How do I reset my payroll password?"

    keyword_hits = await evaluation_surfaces.keyword_retriever.search(
        query, top_k=20, language="en"
    )
    result = await evaluation_surfaces.qa_service.execute(
        QueryRequestDTO(
            query=query, user_id="tripwire", correlation_id="tripwire", top_k=5
        )
    )

    assert keyword_hits == []
    assert result.refused is False, (
        "The service now refuses when it has no relevant evidence. Add a "
        "correct_refusal_rate floor to tests/eval/thresholds.json and regenerate the "
        "baseline."
    )


async def test_citations_do_not_all_resolve_to_the_corpus(
    evaluation_surfaces: EvaluationSurfaces,
) -> None:
    result = await evaluation_surfaces.qa_service.execute(
        QueryRequestDTO(
            query="What caused the fire alarm during the night shift?",
            user_id="tripwire",
            correlation_id="tripwire",
            top_k=5,
        )
    )
    cited = {citation.chunk_id for citation in result.citations}

    assert not cited <= evaluation_surfaces.corpus_chunk_ids, (
        "Every citation now resolves to a real corpus chunk, so the dense stub is "
        "probably gone. Raise the citation_corpus_coverage expectation to 1.0."
    )


async def test_language_detection_distinguishes_arabic_from_english(
    evaluation_surfaces: EvaluationSurfaces,
) -> None:
    """The inverse of a tripwire: this pins behaviour that ES-322 introduced and that the
    Arabic evaluation depends on. If it fails, Arabic cases are passing the language gate
    for the wrong reason.
    """
    result = await evaluation_surfaces.qa_service.execute(
        QueryRequestDTO(
            query="ما سبب إنذار الحريق أثناء الوردية الليلية؟",
            user_id="tripwire",
            correlation_id="tripwire",
            top_k=5,
        )
    )

    assert any(
        citation.chunk_id.startswith("log-ar-") for citation in result.citations
    ), "An Arabic question retrieved no Arabic evidence; language routing is broken."
