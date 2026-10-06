"""Sparse retrieval on PostgreSQL full-text search.

Two strategies share the same index (``chunks.search_vector`` + GIN):

* :class:`PostgresBM25Search` - true Okapi BM25 (Lucene IDF variant) computed in SQL.
  Term frequencies come from ``tsvector`` positions; corpus statistics (N, avgdl, df) are
  computed over the *knowledge-base scope* (tenant + KB + active), so one tenant's corpus
  never influences another tenant's scores.
* :class:`PostgresFTSSearch` - PostgreSQL's native ``ts_rank_cd`` cover-density ranking.

Both use OR semantics over the query's normalized lexemes (natural-language questions
rarely contain *all* their words in the relevant passage).
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.errors import ProviderError
from app.providers.sparse.base import SparseSearch
from app.providers.sparse.sql_filters import compile_scope_and_filters
from app.retrieval.types import RetrievalQuery, RetrievalResult

MAX_QUERY_TERMS = 64

_TERMS_SQL = text(
    "SELECT DISTINCT unnest(tsvector_to_array(to_tsvector(CAST(:cfg AS regconfig), :q))) AS term"
)


def tsquery_lexeme(term: str) -> str:
    """Quote a normalized lexeme for tsquery input syntax (no re-normalization)."""
    return "'" + term.replace("\\", "\\\\").replace("'", "''") + "'"


def or_tsquery(terms: Sequence[str]) -> str:
    return " | ".join(tsquery_lexeme(t) for t in terms)


async def query_terms(session: AsyncSession, query: str, language: str) -> list[str]:
    rows = await session.execute(_TERMS_SQL, {"cfg": language, "q": query})
    return sorted(r[0] for r in rows)[:MAX_QUERY_TERMS]


class _PostgresSparseBase(SparseSearch):
    name = "sparse"

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory

    async def search(self, query: RetrievalQuery, *, language: str = "english") -> list[RetrievalResult]:
        try:
            async with self.session_factory() as session:
                terms = await query_terms(session, query.text, language)
                if not terms:
                    return []
                rows = (await self._execute(session, query, terms)).all()
        except Exception as exc:
            raise ProviderError(f"Sparse search failed: {exc}") from exc
        return [
            RetrievalResult(
                chunk_id=row.id,
                document_id=row.document_id,
                score=float(row.score),
                rank=i + 1,
                retriever=self.name,
            )
            for i, row in enumerate(rows)
        ]

    async def _execute(self, session: AsyncSession, query: RetrievalQuery, terms: list[str]):
        raise NotImplementedError


class PostgresBM25Search(_PostgresSparseBase):
    def __init__(
        self, session_factory: async_sessionmaker[AsyncSession], *, k1: float = 1.2, b: float = 0.75
    ) -> None:
        super().__init__(session_factory)
        self.k1 = k1
        self.b = b

    async def _execute(self, session: AsyncSession, query: RetrievalQuery, terms: list[str]):
        where, params = compile_scope_and_filters(query.scope, query.filters)
        scope_where, _ = compile_scope_and_filters(query.scope, ())
        sql = f"""
WITH terms AS (
    SELECT t.term, t.tsq
    FROM unnest(CAST(:terms AS text[]), CAST(:term_tsqs AS text[])) AS t(term, tsq)
),
stats AS (
    SELECT count(*)::float8 AS n, COALESCE(avg(c.token_count), 1)::float8 AS avgdl
    FROM chunks c WHERE {scope_where}
),
df AS (
    SELECT t.term,
           (SELECT count(*) FROM chunks c
             WHERE {scope_where} AND c.search_vector @@ CAST(t.tsq AS tsquery))::float8 AS df
    FROM terms t
),
cand AS (
    SELECT c.id, c.document_id, c.token_count, c.search_vector
    FROM chunks c
    WHERE {where} AND c.search_vector @@ CAST(:tsq AS tsquery)
),
tf AS (
    SELECT cand.id, cand.document_id, cand.token_count, u.lexeme,
           COALESCE(array_length(u.positions, 1), 1)::float8 AS tf
    FROM cand CROSS JOIN LATERAL unnest(cand.search_vector) AS u(lexeme, positions, weights)
    WHERE u.lexeme = ANY(CAST(:terms AS text[]))
)
SELECT tf.id, tf.document_id,
       SUM(
         ln(1.0 + (stats.n - df.df + 0.5) / (df.df + 0.5))
         * (tf.tf * (CAST(:k1 AS float8) + 1.0))
         / (tf.tf + CAST(:k1 AS float8) * (1.0 - CAST(:b AS float8)
                + CAST(:b AS float8) * tf.token_count / GREATEST(stats.avgdl, 1.0)))
       ) AS score
FROM tf
JOIN df ON df.term = tf.lexeme
CROSS JOIN stats
GROUP BY tf.id, tf.document_id
ORDER BY score DESC, tf.id
LIMIT :top_k
"""
        params.update(
            terms=terms,
            term_tsqs=[tsquery_lexeme(t) for t in terms],
            tsq=or_tsquery(terms),
            k1=self.k1,
            b=self.b,
            top_k=query.top_k,
        )
        return await session.execute(text(sql), params)


class PostgresFTSSearch(_PostgresSparseBase):
    async def _execute(self, session: AsyncSession, query: RetrievalQuery, terms: list[str]):
        where, params = compile_scope_and_filters(query.scope, query.filters)
        sql = f"""
SELECT c.id, c.document_id,
       ts_rank_cd(c.search_vector, CAST(:tsq AS tsquery), 32) AS score
FROM chunks c
WHERE {where} AND c.search_vector @@ CAST(:tsq AS tsquery)
ORDER BY score DESC, c.id
LIMIT :top_k
"""
        params.update(tsq=or_tsquery(terms), top_k=query.top_k)
        return await session.execute(text(sql), params)


def ids_of(results: Sequence[RetrievalResult]) -> list[uuid.UUID]:
    return [r.chunk_id for r in results]
