import uuid

import pytest
from qdrant_client import models as qm

from app.core.errors import ValidationError
from app.providers.sparse.sql_filters import compile_scope_and_filters
from app.providers.vectorstores.qdrant import build_qdrant_filter
from app.retrieval.filters import FilterCondition, parse_filters
from app.retrieval.types import RetrievalScope

SCOPE = RetrievalScope(tenant_id=uuid.uuid4(), knowledge_base_id=uuid.uuid4())


def test_parse_all_shapes():
    doc_id = str(uuid.uuid4())
    conds = parse_filters(
        {
            "document_type": "pdf",
            "source": ["wiki", "handbook"],
            "page": {"gte": 2, "lte": 5},
            "permissions": "group:finance",
            "document_id": doc_id.upper(),
            "metadata.department": "finance",
        }
    )
    by = {c.field: c for c in conds}
    assert by["document_type"] == FilterCondition("document_type", "eq", "pdf")
    assert by["source"].op == "in" and by["source"].value == ("wiki", "handbook")
    assert by["page"].op == "range" and by["page"].value == {"gte": 2, "lte": 5}
    assert by["permissions"] == FilterCondition("permissions", "any", ("group:finance",))
    assert by["document_id"].value == doc_id  # normalized
    assert by["metadata.department"].metadata_key == "department"


@pytest.mark.parametrize("field", ["tenant_id", "knowledge_base_id", "is_active"])
def test_isolation_fields_cannot_be_filtered(field):
    with pytest.raises(ValidationError):
        parse_filters({field: str(uuid.uuid4())})


@pytest.mark.parametrize(
    "raw",
    [
        {"unknown": 1},
        {"document_id": "not-a-uuid"},
        {"page": "two"},
        {"page": {"between": 1}},
        {"source": []},
        {"document_type": {"gte": 1}},
        {"metadata.bad key": 1},
        {"metadata.x": {"gte": "a"}},
    ],
)
def test_invalid_filters_rejected(raw):
    with pytest.raises(ValidationError):
        parse_filters(raw)


def test_qdrant_filter_always_scopes_tenant_kb_active():
    f = build_qdrant_filter(SCOPE, parse_filters({"document_type": "md"}))
    keys = [c.key for c in f.must if isinstance(c, qm.FieldCondition)]
    assert keys[:3] == ["tenant_id", "knowledge_base_id", "is_active"]
    assert f.must[0].match.value == str(SCOPE.tenant_id)
    assert f.must[1].match.value == str(SCOPE.knowledge_base_id)
    assert "document_type" in keys


def test_qdrant_filter_ops():
    f = build_qdrant_filter(
        SCOPE,
        parse_filters({"source": ["a", "b"], "page": {"gt": 1}, "metadata.score": 0.5, "permissions": ["x"]}),
    )
    conds = {c.key: c for c in f.must if isinstance(c, qm.FieldCondition)}
    assert conds["source"].match.any == ["a", "b"]
    assert conds["page"].range.gt == 1
    assert conds["metadata.score"].range.gte == 0.5  # float equality via range
    assert conds["permissions"].match.any == ["x"]


def test_sql_compile_scope_first_and_values_parameterised():
    sql, params = compile_scope_and_filters(
        SCOPE,
        parse_filters(
            {"document_type": "pdf'; DROP TABLE chunks; --", "metadata.dept": ["a", "b"], "page": {"lte": 3}}
        ),
    )
    assert sql.startswith("c.tenant_id = :scope_tenant AND c.knowledge_base_id = :scope_kb AND c.is_active")
    assert "DROP TABLE" not in sql
    assert params["scope_tenant"] == SCOPE.tenant_id
    assert params["f0"] == "pdf'; DROP TABLE chunks; --"
    assert params["f1_0"] == '{"dept": "a"}'
    assert "c.page <= CAST(:f2_lte AS numeric)" in sql
