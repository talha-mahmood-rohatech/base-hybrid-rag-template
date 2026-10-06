from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import TenantContext, get_session, get_tenant
from app.core.errors import NotFoundError
from app.models import RetrievalTrace
from app.schemas.api import TraceOut

router = APIRouter(prefix="/v1/traces", tags=["traces"])


@router.get("/{trace_id}", response_model=TraceOut)
async def get_trace(
    trace_id: uuid.UUID,
    tenant: TenantContext = Depends(get_tenant),
    session: AsyncSession = Depends(get_session),
) -> RetrievalTrace:
    trace = await session.get(RetrievalTrace, trace_id)
    if trace is None or trace.tenant_id != tenant.tenant_id:
        raise NotFoundError("Trace not found")
    return trace
