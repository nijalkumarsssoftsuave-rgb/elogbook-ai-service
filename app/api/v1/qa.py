from fastapi import APIRouter, Depends

from app.api.dependencies import get_qa_service
from app.api.schemas.envelope import ApiResponse
from app.api.schemas.qa import QAQueryRequest, QAQueryResponseData
from app.application.dto import QueryRequestDTO
from app.application.qa_service import QAApplicationService
from app.core.correlation import get_correlation_id
from app.core.security import RbacContext, get_rbac_context

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
    )
    result = await qa_service.execute(request_dto)
    return ApiResponse.ok(data=QAQueryResponseData.from_dto(result), correlation_id=correlation_id)
