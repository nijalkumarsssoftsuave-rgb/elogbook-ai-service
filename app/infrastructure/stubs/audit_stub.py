from app.domain.models import GroundedAnswer, Question


class AuditStub:
    """No-op audit sink. Stands in for the SQL Server audit/jobs store."""

    async def record_query(
        self, question: Question, answer: GroundedAnswer, correlation_id: str
    ) -> None:
        return None
