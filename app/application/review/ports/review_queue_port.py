from typing import Protocol

from app.domain.review import ReviewRequest


class ReviewQueuePort(Protocol):
    """Where a review task is handed off to be picked up by a human later.

    Deliberately one method and no read side. This service creates review work; it does not
    list it, claim it or resolve it -- that is a reviewer-facing surface, and a port that
    grew those would invite this service to start managing a queue it only writes to.

    Implementations are expected to be *durable*: an answer that needed review and produced
    no task is a silent gap in the audit trail, not a missing convenience. Nothing here
    guarantees that, which is exactly why it is a port -- the in-memory adapter that ships
    today is honest about being a development stand-in.
    """

    async def enqueue(self, request: ReviewRequest) -> None: ...
