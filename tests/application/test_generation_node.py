from app.application.qa.nodes.generation import GenerationNode
from app.domain.models import GenerationRequest, Question, RetrievedChunk


class FakeModelClient:
    def __init__(self, completion: str) -> None:
        self._completion = completion
        self.received: GenerationRequest | None = None

    async def generate(self, request: GenerationRequest) -> str:
        self.received = request
        return self._completion


def _question(language: str = "en") -> Question:
    return Question(text="what happened on the last shift?", user_id="u1", language=language)


def _chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="log-001",
            document_id="doc-log-001",
            text="Morning shift equipment check completed.",
            score=1.5,
            metadata={"source_title": "Morning Shift Equipment Log", "page_number": 3},
        ),
        RetrievedChunk(
            chunk_id="log-002",
            document_id="doc-log-002",
            text="Minor slip near the loading dock.",
            score=1.1,
            metadata={"source_title": "Incident Report - Loading Dock"},
        ),
    ]


async def _generate(
    completion: str, language: str = "en"
) -> tuple[object, FakeModelClient]:
    client = FakeModelClient(completion)
    answer = await GenerationNode(model_client=client).generate(_question(language), _chunks())
    return answer, client


# --- what the model is given ---------------------------------------------------------------


async def test_evidence_is_labelled_with_citation_ids() -> None:
    _, client = await _generate("Nothing cited here.")

    assert client.received is not None
    assert [item.citation_id for item in client.received.evidence] == ["c1", "c2"]
    assert client.received.evidence[0].text == "Morning shift equipment check completed."


async def test_the_model_is_never_shown_an_internal_chunk_id() -> None:
    """Withholding it is what stops the model inventing a plausible-looking `log-014`: the
    only thing it can cite is a label we issued.
    """
    _, client = await _generate("Nothing cited here.")

    assert client.received is not None
    assert "chunk_id" not in client.received.evidence[0].model_dump()
    assert "log-001" not in " ".join(item.text for item in client.received.evidence)


async def test_the_question_and_language_reach_the_model() -> None:
    _, client = await _generate("Nothing cited here.", language="ar")

    assert client.received is not None
    assert client.received.question_text == "what happened on the last shift?"
    assert client.received.language == "ar"


# --- the four cases the ticket names ---------------------------------------------------------


async def test_single_citation() -> None:
    answer, _ = await _generate("Checks were done [c1].")

    assert [(c.citation_id, c.chunk_id, c.order) for c in answer.citations] == [
        ("c1", "log-001", 1)
    ]
    assert answer.answer_text == "Checks were done [c1]."


async def test_multiple_citations() -> None:
    answer, _ = await _generate("Checks were done [c1] and a slip occurred [c2].")

    assert [(c.citation_id, c.chunk_id, c.order) for c in answer.citations] == [
        ("c1", "log-001", 1),
        ("c2", "log-002", 2),
    ]


async def test_duplicate_citations_are_recorded_once_but_left_in_the_text() -> None:
    """A claim may legitimately cite the same evidence twice, so both markers stay in the
    prose -- the citation list is a set of sources, not a count of mentions.
    """
    answer, _ = await _generate("First [c1]. Second [c1]. Third [c1].")

    assert [c.citation_id for c in answer.citations] == ["c1"]
    assert answer.answer_text.count("[c1]") == 3


async def test_no_citations() -> None:
    answer, _ = await _generate("A plain answer with no markers at all.")

    assert answer.citations == []
    assert answer.answer_text == "A plain answer with no markers at all."


# --- order is not the label -------------------------------------------------------------------


async def test_order_records_first_mention_not_the_evidence_label() -> None:
    """The distinction ES-329 Phase 1 anticipated and these phases make real: the id is
    where the evidence sat, the order is where the model cited it.
    """
    answer, _ = await _generate("A slip [c2], then the checks [c1].")

    assert [(c.citation_id, c.order) for c in answer.citations] == [("c2", 1), ("c1", 2)]


# --- markers we never issued -----------------------------------------------------------------


async def test_an_unissued_citation_id_produces_no_citation() -> None:
    answer, _ = await _generate("An invented source says so [c9].")

    assert answer.citations == []


async def test_an_unissued_marker_is_removed_from_the_answer_text() -> None:
    """Leaving [c9] in the prose would show the reader a footnote pointing at nothing --
    the text equivalent of a fabricated citation.
    """
    answer, _ = await _generate("Checks were done [c1] and something else [c9].")

    assert "[c9]" not in answer.answer_text
    assert answer.answer_text == "Checks were done [c1] and something else."


async def test_removing_a_marker_does_not_leave_doubled_spaces() -> None:
    answer, _ = await _generate("Checks [c9] were done [c1].")

    assert answer.answer_text == "Checks were done [c1]."


async def test_no_evidence_means_no_labels_and_no_citations() -> None:
    client = FakeModelClient("I cannot answer that.")

    answer = await GenerationNode(model_client=client).generate(_question(), [])

    assert client.received is not None
    assert client.received.evidence == []
    assert answer.citations == []
