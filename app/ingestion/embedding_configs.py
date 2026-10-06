"""Embedding configuration registry (PostgreSQL ``embedding_configs`` <-> Qdrant collections)."""

from __future__ import annotations

import uuid

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import EmbeddingSettings
from app.core.errors import NotFoundError
from app.models import EmbeddingConfig
from app.providers.embeddings.base import EmbeddingSpec
from app.providers.vectorstores.base import CollectionSpec
from app.providers.vectorstores.qdrant import collection_name_for


def spec_of(row: EmbeddingConfig) -> EmbeddingSpec:
    return EmbeddingSpec(
        provider=row.provider,
        model=row.model,
        dimension=row.dimension,
        version=row.version,
        distance_metric=row.distance_metric,
    )


def collection_of(row: EmbeddingConfig) -> CollectionSpec:
    return CollectionSpec(
        name=row.collection_name, dimension=row.dimension, distance_metric=row.distance_metric
    )


class EmbeddingConfigService:
    def __init__(self, collection_prefix: str) -> None:
        self.collection_prefix = collection_prefix

    async def get_or_create(
        self,
        session: AsyncSession,
        *,
        provider: str,
        model: str,
        dimension: int,
        version: str,
        distance_metric: str,
        params: dict | None = None,
        make_default: bool = False,
    ) -> EmbeddingConfig:
        collection = collection_name_for(
            self.collection_prefix, provider, model, dimension, distance_metric, version
        )
        stmt = (
            insert(EmbeddingConfig)
            .values(
                id=uuid.uuid4(),
                provider=provider,
                model=model,
                dimension=dimension,
                version=version,
                distance_metric=distance_metric,
                collection_name=collection,
                params=params or {},
                is_default=False,
            )
            .on_conflict_do_nothing()
        )
        await session.execute(stmt)
        row = (
            await session.execute(
                select(EmbeddingConfig).where(
                    EmbeddingConfig.provider == provider,
                    EmbeddingConfig.model == model,
                    EmbeddingConfig.dimension == dimension,
                    EmbeddingConfig.version == version,
                    EmbeddingConfig.distance_metric == distance_metric,
                )
            )
        ).scalar_one()
        if make_default and not row.is_default:
            await session.execute(
                update(EmbeddingConfig).where(EmbeddingConfig.id != row.id).values(is_default=False)
            )
            row.is_default = True
        return row

    async def ensure_default(self, session: AsyncSession, settings: EmbeddingSettings) -> EmbeddingConfig:
        params = {
            k: v
            for k, v in {
                "query_prefix": settings.query_prefix,
                "document_prefix": settings.document_prefix,
                "max_input_tokens": settings.max_input_tokens,
                "base_url": settings.base_url,
            }.items()
            if v is not None
        }
        return await self.get_or_create(
            session,
            provider=settings.provider,
            model=settings.model,
            dimension=settings.dimension,
            version=settings.version,
            distance_metric=settings.distance_metric,
            params=params,
            make_default=True,
        )

    async def get(self, session: AsyncSession, config_id: uuid.UUID) -> EmbeddingConfig:
        row = await session.get(EmbeddingConfig, config_id)
        if row is None:
            raise NotFoundError(f"Embedding config {config_id} not found")
        return row

    async def get_default(self, session: AsyncSession) -> EmbeddingConfig:
        row = (
            await session.execute(select(EmbeddingConfig).where(EmbeddingConfig.is_default.is_(True)))
        ).scalar_one_or_none()
        if row is None:
            raise NotFoundError("No default embedding config; the application has not been initialised")
        return row
