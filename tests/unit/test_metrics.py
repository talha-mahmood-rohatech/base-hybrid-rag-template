import math

import pytest

from app.evaluation.metrics import (
    citation_precision,
    citation_validity,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

RET = ["x", "a", "y", "b", "z"]
REL = {"a": 1.0, "b": 1.0, "c": 1.0}


def test_recall_precision():
    assert recall_at_k(RET, REL, 2) == pytest.approx(1 / 3)
    assert recall_at_k(RET, REL, 5) == pytest.approx(2 / 3)
    assert precision_at_k(RET, REL, 2) == pytest.approx(1 / 2)
    assert precision_at_k(RET, REL, 5) == pytest.approx(2 / 5)
    assert recall_at_k(RET, {}, 5) == 0.0


def test_hit_and_mrr():
    assert hit_rate_at_k(RET, REL, 1) == 0.0
    assert hit_rate_at_k(RET, REL, 2) == 1.0
    assert mrr(RET, REL) == pytest.approx(1 / 2)
    assert mrr(["x", "y"], REL) == 0.0
    assert mrr(RET, REL, k=1) == 0.0


def test_ndcg_hand_computed():
    # gains at ranks 2 and 4 (binary): DCG = 1/log2(3) + 1/log2(5); IDCG (3 relevant, k=5) = 1 + 1/log2(3) + 1/2
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3) + 0.5)
    assert ndcg_at_k(RET, REL, 5) == pytest.approx(expected)
    assert ndcg_at_k(["a", "b", "c"], REL, 3) == pytest.approx(1.0)


def test_graded_ndcg_prefers_higher_grade_first():
    rel = {"a": 2.0, "b": 1.0}
    assert ndcg_at_k(["a", "b"], rel, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a"], rel, 2) < 1.0


def test_citation_metrics():
    assert citation_precision(["a", "x"], REL) == 0.5
    assert citation_precision([], REL) == 0.0
    assert citation_validity(["a", "q"], ["a", "b"]) == 0.5
    assert citation_validity([], ["a"]) == 1.0
