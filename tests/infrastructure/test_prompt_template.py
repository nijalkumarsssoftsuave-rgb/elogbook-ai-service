from app.domain.models import EvidenceItem, GenerationRequest, GroundedAnswer
from app.infrastructure.model_serving.llm.prompt_template import build_prompt


def _request(*, evidence: list[EvidenceItem] | None = None) -> GenerationRequest:
    return GenerationRequest(
        question_text="What caused the fire alarm?",
        language="en",
        evidence=evidence
        if evidence is not None
        else [
            EvidenceItem(citation_id="c1", text="Fire alarm triggered at 02:47."),
            EvidenceItem(citation_id="c2", text="Minor slip near the loading dock."),
        ],
    )


def test_evidence_is_labelled_with_its_citation_marker() -> None:
    prompt = build_prompt(_request())

    assert "[c1] Fire alarm triggered at 02:47." in prompt
    assert "[c2] Minor slip near the loading dock." in prompt


def test_the_prompt_contains_no_internal_identifiers() -> None:
    """Only labels we issued reach the model, so there is nothing internal for it to echo
    back or imitate -- and the worst it can cite is an id that was never offered.
    """
    prompt = build_prompt(
        GenerationRequest(
            question_text="q",
            language="en",
            evidence=[EvidenceItem(citation_id="c1", text="Some evidence text.")],
        )
    )

    assert "chunk_id" not in prompt
    assert "log-" not in prompt
    assert "source_title" not in prompt


def test_the_instructions_ask_for_citation_ids_in_the_agreed_syntax() -> None:
    prompt = build_prompt(_request())

    assert "[c1]" in prompt
    assert "ONLY" in prompt


def test_the_refusal_sentence_is_quoted_verbatim() -> None:
    """The model is told to reply with exactly this sentence, and the pipeline recognises
    it, so the two must be the same string rather than two similar ones.
    """
    prompt = build_prompt(_request())

    assert GroundedAnswer.REFUSAL_TEXT in prompt


def test_the_question_and_language_are_present() -> None:
    prompt = build_prompt(_request())

    assert "What caused the fire alarm?" in prompt
    assert "(en)" in prompt


def test_a_request_with_no_evidence_still_renders() -> None:
    """Retrieval returning nothing must not crash prompt construction -- the model is
    simply asked a question with an empty evidence block, and refuses.
    """
    prompt = build_prompt(_request(evidence=[]))

    assert "--- EVIDENCE ---" in prompt
    assert GroundedAnswer.REFUSAL_TEXT in prompt
