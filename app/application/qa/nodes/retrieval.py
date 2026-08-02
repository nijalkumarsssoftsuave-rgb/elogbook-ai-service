from app.application.retrieval_service import RetrievalService
from app.domain.models import (
    PermissionScope,
    Question,
    RetrievalSearchContext,
    RetrievedChunk,
)


class RetrievalNode:
    """The retrieval step of the QA workflow.

    It accepts the caller's permission scope, builds the retrieval search context, and
    invokes RetrievalService. Deliberately thin: it owns the shape of the request and
    nothing else. Resolving sources, searching them, merging, fusing and reranking all
    live inside the service, and the orchestrator above sees only chunks.

    Naming this step as its own object is what lets the permission scope travel with the
    question rather than being threaded through the orchestrator as one more loose
    argument -- and gives the workflow a place to stand when it becomes a LangGraph node.
    """

    def __init__(self, retrieval_service: RetrievalService) -> None:
        self._retrieval_service = retrieval_service

    async def run(
        self, question: Question, permission_scope: PermissionScope, top_k: int = 5
    ) -> list[RetrievedChunk]:
        return await self._retrieval_service.retrieve(
            RetrievalSearchContext(
                question=question, permission_scope=permission_scope, top_k=top_k
            )
        )
