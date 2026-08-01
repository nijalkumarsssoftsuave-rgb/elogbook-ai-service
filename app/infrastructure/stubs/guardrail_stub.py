from app.domain.models import GroundedAnswer, Question, RetrievedChunk


class GuardrailStub:
    """No-op pass-through at all three checkpoints. Stands in for NeMo Guardrails
    (jailbreak/topic), DeBERTa-v3 (prompt injection, including indirect injection in
    retrieved evidence), Presidio (PII redaction) and the citation resolver.
    """

    async def check_input(self, question: Question) -> None:
        return None

    async def check_retrieved_chunks(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return chunks

    async def check_output(self, answer: GroundedAnswer) -> GroundedAnswer:
        return answer
