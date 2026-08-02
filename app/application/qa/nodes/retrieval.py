from app.application.ports import PermissionResolverPort
from app.application.retrieval_service import RetrievalService
from app.domain.models import Question, RetrievedChunk
from app.domain.permission import PermissionScope
from app.domain.query import QueryFilters


class RetrievalNode:
    """The retrieval step of the QA workflow.

    Two moves, in this order: resolve the caller's RBAC into the search set they are
    entitled to, then retrieve within it. Resolution lives here rather than inside
    RetrievalService so that "who may see what" is decided by the component named for it,
    and the service below is left to do nothing but search.

    That split costs something worth naming: while RetrievalService took a PermissionScope
    it was structurally impossible to retrieve with a hand-made source list. Now a caller
    could in principle pass a SearchScope that never went through RBAC. Two things hold the
    line -- this node always resolves rather than accepting a scope from above, and the
    service still refuses to search an empty scope.
    """

    def __init__(
        self,
        permission_resolver: PermissionResolverPort,
        retrieval_service: RetrievalService,
    ) -> None:
        self._permission_resolver = permission_resolver
        self._retrieval_service = retrieval_service

    async def run(
        self,
        question: Question,
        permission_scope: PermissionScope,
        top_k: int = 5,
        filters: QueryFilters | None = None,
    ) -> list[RetrievedChunk]:
        search_scope = await self._permission_resolver.resolve(permission_scope)
        # The caller's filters travel beside the resolved scope, never merged into it.
        # A scope is an entitlement and a filter is a preference; combining them by
        # assignment is how a request would come to widen its own access.
        return await self._retrieval_service.retrieve(
            question, search_scope, top_k, filters or QueryFilters()
        )
