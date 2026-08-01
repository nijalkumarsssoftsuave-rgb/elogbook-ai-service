from app.application.ports import GuardrailPort
from app.domain.models import GroundedAnswer, Question, RetrievedChunk


class GuardrailService:
    """The three points where guardrails run: on the incoming question, on the retrieved
    evidence before it reaches the prompt, and on the generated answer before it is
    returned.

    All three sit behind one port for now — NeMo Guardrails, DeBERTa-v3 and Presidio are
    not independently swappable yet, so splitting them into separate ports would add
    interfaces with only one implementation each.
    """

    def __init__(self, guardrail: GuardrailPort) -> None:
        self._guardrail = guardrail

    async def check_question(self, question: Question) -> None:
        """Jailbreak and topic-restriction screening on the user's question."""
        await self._guardrail.check_input(question)

    async def screen_retrieved_chunks(
        self, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Removes evidence carrying indirect prompt injection before it is put in front
        of the model.
        """
        return await self._guardrail.check_retrieved_chunks(chunks)

    async def check_answer(self, answer: GroundedAnswer) -> GroundedAnswer:
        """Output guardrails and PII redaction on the generated answer."""
        return await self._guardrail.check_output(answer)
