from app.domain.models import GroundedAnswer, Question


class GuardrailStub:
    """No-op pass-through. Stands in for NeMo Guardrails / DeBERTa-v3 / Presidio / citation resolver."""

    async def check_input(self, question: Question) -> None:
        return None

    async def check_output(self, answer: GroundedAnswer) -> GroundedAnswer:
        return answer
