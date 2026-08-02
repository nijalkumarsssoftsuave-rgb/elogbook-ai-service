from fastapi import APIRouter, Depends

from app.api.dependencies import get_qa_service
from app.api.schemas.envelope import ApiResponse
from app.api.schemas.qa import QAQueryRequest, QAQueryResponseData
from app.application.dto import QueryRequestDTO
from app.application.qa_service import QAApplicationService
from app.core.correlation import get_correlation_id
from app.core.security import RbacContext, get_rbac_context
from app.domain.permission import PermissionScope

router = APIRouter(prefix="/api/v1/qa", tags=["qa"])


@router.post("/query", response_model=ApiResponse[QAQueryResponseData])
async def query(
    payload: QAQueryRequest,
    rbac: RbacContext = Depends(get_rbac_context),
    correlation_id: str = Depends(get_correlation_id),
    qa_service: QAApplicationService = Depends(get_qa_service),
) -> ApiResponse[QAQueryResponseData]:
    request_dto = QueryRequestDTO(
        query=payload.query,
        user_id=rbac.user_id,
        roles=rbac.roles,
        correlation_id=correlation_id,
        top_k=payload.top_k,
        # The caller's entitlements, derived from the verified token. Retrieval resolves
        # this into the sources it is allowed to search.
        permission_scope=PermissionScope.from_roles(rbac.roles),
        # Forwarded exactly as received. The endpoint validates the shape of a filter
        # and nothing else -- what it means for retrieval is decided further down.
        filters=payload.filters,
    )
    result = await qa_service.execute(request_dto)
    return ApiResponse.ok(data=QAQueryResponseData.from_dto(result), correlation_id=correlation_id)
