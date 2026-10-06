"""Compile :class:`FilterCondition` objects into a parameterised SQL fragment over ``chunks c``.

Field names are whitelisted by :func:`app.retrieval.filters.parse_filters`; only values are
bound as parameters, so user input never reaches the SQL text.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from typing import Any

from app.retrieval.filters import (
    ARRAY_FIELDS,
    INT_FIELDS,
    KEYWORD_FIELDS,
    UUID_FIELDS,
    FilterCondition,
)
from app.retrieval.types import RetrievalScope

_COLUMN = {
    "document_id": "c.document_id",
    "document_version_id": "c.document_version_id",
    "chunk_id": "c.id",
    "document_type": "c.document_type",
    "source": "c.source",
    "section": "c.section",
    "page": "c.page",
    "permissions": "c.permissions",
}
_RANGE_SQL = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}


def compile_scope_and_filters(
    scope: RetrievalScope, filters: Sequence[FilterCondition], prefix: str = "f"
) -> tuple[str, dict[str, Any]]:
    params: dict[str, Any] = {"scope_tenant": scope.tenant_id, "scope_kb": scope.knowledge_base_id}
    clauses = ["c.tenant_id = :scope_tenant", "c.knowledge_base_id = :scope_kb"]
    if scope.active_only:
        clauses.append("c.is_active")

    for i, cond in enumerate(filters):
        p = f"{prefix}{i}"
        meta_key = cond.metadata_key
        if meta_key is not None:
            if cond.op == "eq":
                clauses.append(f"c.metadata @> CAST(:{p} AS jsonb)")
                params[p] = json.dumps({meta_key: cond.value})
            elif cond.op in ("in", "any"):
                ors = []
                for j, v in enumerate(cond.value):
                    ors.append(f"c.metadata @> CAST(:{p}_{j} AS jsonb)")
                    params[f"{p}_{j}"] = json.dumps({meta_key: v})
                clauses.append("(" + " OR ".join(ors) + ")")
            elif cond.op == "range":
                params[f"{p}_key"] = meta_key
                expr = (
                    f"(CASE WHEN jsonb_typeof(c.metadata -> CAST(:{p}_key AS text)) = 'number' "
                    f"THEN (c.metadata ->> CAST(:{p}_key AS text))::numeric END)"
                )
                for op, bound in cond.value.items():
                    clauses.append(f"{expr} {_RANGE_SQL[op]} CAST(:{p}_{op} AS numeric)")
                    params[f"{p}_{op}"] = bound
            continue

        col = _COLUMN[cond.field]
        if cond.field in ARRAY_FIELDS:
            clauses.append(f"{col} && CAST(:{p} AS varchar[])")
            params[p] = list(cond.value)
        elif cond.op == "eq":
            clauses.append(f"{col} = :{p}")
            params[p] = uuid.UUID(cond.value) if cond.field in UUID_FIELDS else cond.value
        elif cond.op == "in":
            if cond.field in UUID_FIELDS:
                clauses.append(f"{col} = ANY(CAST(:{p} AS uuid[]))")
                params[p] = [uuid.UUID(v) for v in cond.value]
            elif cond.field in INT_FIELDS:
                clauses.append(f"{col} = ANY(CAST(:{p} AS integer[]))")
                params[p] = list(cond.value)
            elif cond.field in KEYWORD_FIELDS:
                clauses.append(f"{col} = ANY(CAST(:{p} AS varchar[]))")
                params[p] = list(cond.value)
        elif cond.op == "range":
            for op, bound in cond.value.items():
                clauses.append(f"{col} {_RANGE_SQL[op]} CAST(:{p}_{op} AS numeric)")
                params[f"{p}_{op}"] = bound
    return " AND ".join(clauses), params
