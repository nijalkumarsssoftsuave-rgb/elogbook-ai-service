from collections import deque

from app.domain.models import AuditRecord


class AuditStub:
    """Keeps the most recent audit records in memory. Stands in for the SQL Server
    audit/jobs store.

    It retains rather than discards so the trail is actually inspectable during
    development -- an audit sink that silently drops everything is indistinguishable from
    a broken one. The buffer is bounded because this object lives for the life of the
    process; real retention is a policy for the database, not for a stub.
    """

    _MAX_RETAINED = 100

    def __init__(self) -> None:
        self.records: deque[AuditRecord] = deque(maxlen=self._MAX_RETAINED)

    async def record(self, record: AuditRecord) -> None:
        self.records.append(record)
