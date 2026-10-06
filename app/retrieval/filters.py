"""Metadata filter DSL shared by dense (Qdrant) and sparse (PostgreSQL) retrieval.

Request shape (all conditions are AND-ed)::

    {
      "document_type": "pdf",                     # equality
      "source": ["handbook", "wiki"],             # any-of
      "page": {"gte": 2, "lte": 10},              # range
      "permissions": ["group:finance", "public"], # chunk permissions intersect the list
      "metadata.department": "finance"            # custom document metadata
    }

``tenant_id`` and ``knowledge_base_id`` are deliberately *not* filterable here: isolation
is enforced by :class:`~app.retrieval.types.RetrievalScope`, built from the authenticated
tenant, never from user input.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from app.core.errors import ValidationError

Scalar = str | int | float | bool
Op = Literal["eq", "in", "range", "any"]

UUID_FIELDS = frozenset({"document_id", "document_version_id", "chunk_id"})
KEYWORD_FIELDS = frozenset({"document_type", "source", "section"})
INT_FIELDS = frozenset({"page"})
ARRAY_FIELDS = frozenset({"permissions"})
RESERVED_FIELDS = frozenset({"tenant_id", "knowledge_base_id", "is_active"})
ALLOWED_FIELDS = UUID_FIELDS | KEYWORD_FIELDS | INT_FIELDS | ARRAY_FIELDS
METADATA_KEY_RE = re.compile(r"^metadata\.([A-Za-z0-9_\-]{1,64})$")
RANGE_OPS = ("gt", "gte", "lt", "lte")


@dataclass(frozen=True, slots=True)
class FilterCondition:
    field: str  # e.g. "document_type" or "metadata.department"
    op: Op
    value: Any

    @property
    def metadata_key(self) -> str | None:
        m = METADATA_KEY_RE.match(self.field)
        return m.group(1) if m else None


def _check_scalar(field_name: str, value: Any) -> Scalar:
    if isinstance(value, str | bool | int | float):
        return value
    raise ValidationError(f"Invalid filter value for '{field_name}': {value!r}")


def _coerce(field_name: str, value: Scalar) -> Scalar:
    if field_name in UUID_FIELDS:
        try:
            return str(uuid.UUID(str(value)))
        except ValueError as exc:
            raise ValidationError(f"Filter '{field_name}' expects a UUID, got {value!r}") from exc
    if field_name in INT_FIELDS:
        if isinstance(value, bool) or not isinstance(value, int | float) or int(value) != value:
            raise ValidationError(f"Filter '{field_name}' expects an integer, got {value!r}")
        return int(value)
    if field_name in KEYWORD_FIELDS | ARRAY_FIELDS and not isinstance(value, str):
        raise ValidationError(f"Filter '{field_name}' expects a string, got {value!r}")
    return value


def parse_filters(raw: dict[str, Any] | None) -> tuple[FilterCondition, ...]:
    """Validate a user filter dict and normalize it into conditions."""
    if not raw:
        return ()
    conditions: list[FilterCondition] = []
    for field_name, value in raw.items():
        if field_name in RESERVED_FIELDS:
            raise ValidationError(
                f"Filter on '{field_name}' is not allowed; isolation is derived from your credentials"
            )
        is_meta = METADATA_KEY_RE.match(field_name) is not None
        if field_name not in ALLOWED_FIELDS and not is_meta:
            raise ValidationError(
                f"Unknown filter field '{field_name}'. Allowed: {sorted(ALLOWED_FIELDS)} or 'metadata.<key>'"
            )

        if isinstance(value, dict):
            if field_name in ARRAY_FIELDS | UUID_FIELDS | KEYWORD_FIELDS:
                raise ValidationError(f"Range filters are not supported on '{field_name}'")
            bad = set(value) - set(RANGE_OPS)
            if bad or not value:
                raise ValidationError(f"Range filter on '{field_name}' must use {RANGE_OPS}")
            rng: dict[str, float | int] = {}
            for k, v in value.items():
                if isinstance(v, bool) or not isinstance(v, int | float):
                    raise ValidationError(f"Range bound '{k}' for '{field_name}' must be numeric")
                rng[k] = v
            conditions.append(FilterCondition(field_name, "range", rng))
        elif isinstance(value, list):
            if not value:
                raise ValidationError(f"Filter list for '{field_name}' must not be empty")
            if len(value) > 256:
                raise ValidationError(f"Filter list for '{field_name}' is too long (max 256)")
            items = tuple(_coerce(field_name, _check_scalar(field_name, v)) for v in value)
            op: Op = "any" if field_name in ARRAY_FIELDS else "in"
            conditions.append(FilterCondition(field_name, op, items))
        else:
            scalar = _coerce(field_name, _check_scalar(field_name, value))
            if field_name in ARRAY_FIELDS:
                conditions.append(FilterCondition(field_name, "any", (scalar,)))
            else:
                conditions.append(FilterCondition(field_name, "eq", scalar))
    return tuple(conditions)


def filters_to_dict(conditions: tuple[FilterCondition, ...]) -> dict[str, Any]:
    return {
        c.field: {"op": c.op, "value": list(c.value) if isinstance(c.value, tuple) else c.value}
        for c in conditions
    }
