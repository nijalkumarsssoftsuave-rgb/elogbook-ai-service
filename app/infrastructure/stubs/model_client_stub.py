from app.domain.models import Citation, GroundedAnswer, Question, RetrievedChunk


class ModelClientStub:
    """Returns a fixed dummy answer. Stands in for the Qwen LLM client."""

    async def generate(
        self, question: Question, context_chunks: list[RetrievedChunk]
    ) -> GroundedAnswer:
        return GroundedAnswer(
            answer_text=f"This is a stub answer for: {question.text!r}",
            citations=[
                Citation(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source_title="Stub Source",
                    score=chunk.score,
                )
                for chunk in context_chunks
            ],
            confidence=0.5,
            is_grounded=True,
        )
