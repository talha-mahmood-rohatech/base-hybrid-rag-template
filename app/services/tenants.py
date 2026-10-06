from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import ConflictError
from app.core.security import generate_api_key, hash_api_key
from app.models import ApiKey, Tenant


async def create_tenant(session: AsyncSession, name: str, slug: str) -> tuple[Tenant, str]:
    if (await session.execute(select(Tenant).where(Tenant.slug == slug))).scalar_one_or_none():
        raise ConflictError(f"Tenant '{slug}' already exists")
    tenant = Tenant(id=uuid.uuid4(), name=name, slug=slug, is_active=True)
    session.add(tenant)
    await session.flush()
    key = await create_api_key(session, tenant.id)
    return tenant, key


async def create_api_key(session: AsyncSession, tenant_id: uuid.UUID, name: str = "default") -> str:
    key = generate_api_key()
    session.add(
        ApiKey(
            id=uuid.uuid4(), tenant_id=tenant_id, name=name, key_prefix=key[:12], key_hash=hash_api_key(key)
        )
    )
    await session.flush()
    return key
