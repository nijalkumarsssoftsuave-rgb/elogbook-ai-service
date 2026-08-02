from app.domain.review import ReviewRequest


class InMemoryReviewQueue:
    """A review queue that keeps tasks in this process, and loses them when it stops.

    The shipped adapter, and a development stand-in rather than a deployment target -- named
    so nobody has to read the code to find that out. It exists because the port needs one
    real implementation to be exercised end to end, and because an answer that needed review
    should produce something inspectable today rather than nothing at all.

    The durable adapters -- a `review_tasks` table on the SQL Server that already holds
    audit and jobs, or a broker if review work ever needs to fan out to several consumers --
    replace this file and nothing else. That is the whole reason ReviewQueuePort exists, and
    it is why they are not scaffolded here in advance: an adapter for a broker this
    deployment does not run yet would be untested code shaped like a decision nobody has
    made.
    """

    def __init__(self) -> None:
        self._requests: list[ReviewRequest] = []

    async def enqueue(self, request: ReviewRequest) -> None:
        self._requests.append(request)

    @property
    def pending(self) -> list[ReviewRequest]:
        """A copy, so a caller inspecting the queue cannot quietly edit it."""
        return list(self._requests)
