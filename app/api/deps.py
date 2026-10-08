from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime

from fastapi import Depends, Header, Request
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.container import Container
from app.core.errors import AuthenticationError
from app.core.security import constant_time_equals, hash_api_key
from app.models import ApiKey, Tenant


def get_container(request: Request) -> Container:
    return request.app.state.container


async def get_session(container: Container = Depends(get_container)) -> AsyncIterator[AsyncSession]:
    async with container.session_factory() as session:
        yield session


@dataclass(frozen=True, slots=True)
class TenantContext:
    tenant_id: uuid.UUID
    api_key_id: uuid.UUID


async def get_tenant(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    session: AsyncSession = Depends(get_session),
) -> TenantContext:
    """Resolve the tenant from the API key. Tenant identity is never taken from the request body."""
    if not x_api_key:
        raise AuthenticationError("Missing X-API-Key header")
    return await authenticate_api_key(session, x_api_key)


async def authenticate_api_key(session: AsyncSession, api_key: str) -> TenantContext:
    """Shared by HTTP (X-API-Key header) and WebSocket (first message) authentication."""
    row = (
        await session.execute(
            select(ApiKey, Tenant)
            .join(Tenant, Tenant.id == ApiKey.tenant_id)
            .where(ApiKey.key_hash == hash_api_key(api_key), ApiKey.revoked_at.is_(None))
        )
    ).first()
    if row is None or not row.Tenant.is_active:
        raise AuthenticationError("Invalid API key")
    await session.execute(
        update(ApiKey).where(ApiKey.id == row.ApiKey.id).values(last_used_at=datetime.now(UTC))
    )
    await session.commit()
    return TenantContext(tenant_id=row.Tenant.id, api_key_id=row.ApiKey.id)


def require_admin(
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
    container: Container = Depends(get_container),
) -> None:
    expected = container.settings.security.admin_api_key.get_secret_value()
    if not x_admin_key or not constant_time_equals(x_admin_key, expected):
        raise AuthenticationError("Invalid admin key")
