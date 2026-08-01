from app.application.generation_service import GenerationService
from app.domain.models import Question, RetrievedChunk


class FakeModelClient:
    def __init__(self, completion: str) -> None:
        self._completion = completion
        self.received_prompt: str | None = None

    async def generate(self, prompt: str) -> str:
        self.received_prompt = prompt
        return self._completion


def _question() -> Question:
    return Question(text="what happened on the last shift?", user_id="u1", language="en")


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


async def test_generate_sends_a_prompt_containing_the_evidence() -> None:
    model_client = FakeModelClient("Nothing cited here.")
    service = GenerationService(model_client=model_client)

    await service.generate(_question(), _chunks())

    prompt = model_client.received_prompt
    assert prompt is not None
    assert "id=log-001" in prompt
    assert "Morning shift equipment check completed." in prompt
    assert "what happened on the last shift?" in prompt


async def test_generate_turns_citation_markers_into_citations() -> None:
    model_client = FakeModelClient("Checks were done [[log-001]] and a slip occurred [[log-002]].")
    service = GenerationService(model_client=model_client)

    answer = await service.generate(_question(), _chunks())

    assert [citation.chunk_id for citation in answer.citations] == ["log-001", "log-002"]
    assert answer.citations[0].source_title == "Morning Shift Equipment Log"
    assert answer.citations[0].page_number == 3
    assert answer.citations[1].page_number is None


async def test_generate_strips_citation_markers_from_the_answer_text() -> None:
    model_client = FakeModelClient("Checks were done [[log-001]] this morning.")
    service = GenerationService(model_client=model_client)

    answer = await service.generate(_question(), _chunks())

    assert "[[" not in answer.answer_text
    assert answer.answer_text == "Checks were done this morning."


async def test_generate_drops_markers_naming_chunks_that_were_not_retrieved() -> None:
    model_client = FakeModelClient("An invented source says so [[log-999]].")
    service = GenerationService(model_client=model_client)

    answer = await service.generate(_question(), _chunks())

    # Citations are never fabricated for unknown ids; the answer is left uncited so
    # validation can reject it.
    assert answer.citations == []


async def test_generate_returns_no_citations_when_the_model_cites_nothing() -> None:
    model_client = FakeModelClient("A plain answer with no markers at all.")
    service = GenerationService(model_client=model_client)

    answer = await service.generate(_question(), _chunks())

    assert answer.citations == []
    assert answer.answer_text == "A plain answer with no markers at all."


async def test_generate_records_a_repeatedly_cited_chunk_only_once() -> None:
    model_client = FakeModelClient("First [[log-001]]. Second [[log-001]]. Third [[log-001]].")
    service = GenerationService(model_client=model_client)

    answer = await service.generate(_question(), _chunks())

    assert [citation.chunk_id for citation in answer.citations] == ["log-001"]
