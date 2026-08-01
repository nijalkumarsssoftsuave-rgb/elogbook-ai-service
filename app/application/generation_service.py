import re

from app.application.ports import ModelClientPort
from app.application.prompt_builder import CITATION_MARKER_PATTERN, PromptBuilder
from app.domain.models import Citation, GroundedAnswer, Question, RetrievedChunk


class GenerationService:
    """Builds the grounded prompt, calls the LLM, and turns the raw completion into a
    GroundedAnswer with structured citations.

    Interpreting the model's output is a generation concern, so citation *extraction*
    lives here; checking those citations against the retrieved evidence is a separate
    concern and lives in CitationValidationService.
    """

    def __init__(
        self, model_client: ModelClientPort, prompt_builder: PromptBuilder | None = None
    ) -> None:
        self._model_client = model_client
        self._prompt_builder = prompt_builder or PromptBuilder()

    async def generate(
        self, question: Question, context_chunks: list[RetrievedChunk]
    ) -> GroundedAnswer:
        prompt = self._prompt_builder.build(question, context_chunks)
        raw_completion = await self._model_client.generate(prompt)
        return self._to_grounded_answer(raw_completion, context_chunks)

    def _to_grounded_answer(
        self, raw_completion: str, context_chunks: list[RetrievedChunk]
    ) -> GroundedAnswer:
        chunk_index = {chunk.chunk_id: chunk for chunk in context_chunks}

        citations: list[Citation] = []
        seen_chunk_ids: set[str] = set()
        for cited_id in CITATION_MARKER_PATTERN.findall(raw_completion):
            chunk = chunk_index.get(cited_id)
            # A marker naming a chunk we never retrieved is dropped rather than
            # fabricated into a Citation; CitationValidationService still sees the
            # remaining citations and can reject the answer.
            if chunk is None or chunk.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk.chunk_id)
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source_title=chunk.metadata.get("source_title", "Untitled"),
                    page_number=chunk.metadata.get("page_number"),
                    score=chunk.score,
                )
            )

        answer_text = CITATION_MARKER_PATTERN.sub("", raw_completion)
        answer_text = re.sub(r"[ \t]{2,}", " ", answer_text).strip()

        return GroundedAnswer(
            answer_text=answer_text,
            citations=citations,
            # No confidence signal until a real model client can report one.
            confidence=None,
            is_grounded=True,
        )
