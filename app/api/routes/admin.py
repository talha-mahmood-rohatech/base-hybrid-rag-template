from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session, require_admin
from app.core.errors import NotFoundError
from app.models import Tenant
from app.schemas.api import TenantCreate, TenantCreated
from app.services.tenants import create_api_key, create_tenant

router = APIRouter(prefix="/v1/admin", tags=["admin"], dependencies=[Depends(require_admin)])


@router.post("/tenants", response_model=TenantCreated, status_code=status.HTTP_201_CREATED)
async def create_tenant_route(
    body: TenantCreate, session: AsyncSession = Depends(get_session)
) -> TenantCreated:
    tenant, key = await create_tenant(session, body.name, body.slug)
    await session.commit()
    return TenantCreated(id=tenant.id, name=tenant.name, slug=tenant.slug, api_key=key)


@router.post(
    "/tenants/{tenant_id}/api-keys", response_model=TenantCreated, status_code=status.HTTP_201_CREATED
)
async def create_key_route(
    tenant_id: uuid.UUID, session: AsyncSession = Depends(get_session)
) -> TenantCreated:
    tenant = await session.get(Tenant, tenant_id)
    if tenant is None:
        raise NotFoundError("Tenant not found")
    key = await create_api_key(session, tenant.id)
    await session.commit()
    return TenantCreated(id=tenant.id, name=tenant.name, slug=tenant.slug, api_key=key)
