import uuid

import pytest

from app.retrieval.fusion.relative_score import RelativeScoreFusion
from app.retrieval.fusion.rrf import ReciprocalRankFusion
from app.retrieval.types import RetrievalResult

IDS = {name: uuid.uuid5(uuid.NAMESPACE_DNS, name) for name in "abcdefg"}
DOC = uuid.uuid4()


def results(names, retriever, scores=None):
    return [
        RetrievalResult(
            chunk_id=IDS[n],
            document_id=DOC,
            score=(scores[i] if scores else 1.0 / (i + 1)),
            rank=i + 1,
            retriever=retriever,
        )
        for i, n in enumerate(names)
    ]


def by_name(cands):
    inv = {v: k for k, v in IDS.items()}
    return {inv[c.chunk_id]: c for c in cands}


def test_rrf_exact_scores_and_order():
    dense = results(["a", "b", "c"], "dense")
    sparse = results(["c", "a", "d"], "sparse")
    fused = ReciprocalRankFusion(k=60).fuse({"dense": dense, "sparse": sparse}, top_k=10)

    expected = {
        "a": 1 / (60 + 1) + 1 / (60 + 2),
        "b": 1 / (60 + 2),
        "c": 1 / (60 + 3) + 1 / (60 + 1),
        "d": 1 / (60 + 3),
    }
    got = by_name(fused)
    for name, score in expected.items():
        assert got[name].rrf_score == pytest.approx(score, abs=1e-15)
    assert [c.chunk_id for c in fused] == [IDS[n] for n in ("a", "c", "b", "d")]
    assert [c.fused_rank for c in fused] == [1, 2, 3, 4]


def test_rrf_preserves_individual_ranks_and_scores():
    dense = results(["a", "b"], "dense", scores=[0.91, 0.42])
    sparse = results(["b", "e"], "sparse", scores=[12.5, 3.25])
    got = by_name(ReciprocalRankFusion().fuse({"dense": dense, "sparse": sparse}, top_k=10))
    assert (got["a"].dense_rank, got["a"].sparse_rank) == (1, None)
    assert (got["b"].dense_rank, got["b"].sparse_rank) == (2, 1)
    assert (got["e"].dense_rank, got["e"].sparse_rank) == (None, 2)
    assert got["b"].scores == {"dense": 0.42, "sparse": 12.5}


def test_rrf_small_k_hand_computed():
    fused = ReciprocalRankFusion(k=1).fuse(
        {"dense": results(["a", "b"], "dense"), "sparse": results(["b", "a"], "sparse")}, top_k=5
    )
    # a: 1/2 + 1/3, b: 1/3 + 1/2 -> tie, broken by best individual rank then id (deterministic)
    assert fused[0].rrf_score == pytest.approx(5 / 6)
    assert fused[1].rrf_score == pytest.approx(5 / 6)


def test_rrf_top_k_truncation_and_ranks():
    dense = results(list("abcdefg"), "dense")
    fused = ReciprocalRankFusion(k=60).fuse({"dense": dense, "sparse": []}, top_k=3)
    assert len(fused) == 3
    assert [c.fused_rank for c in fused] == [1, 2, 3]
    assert fused[0].chunk_id == IDS["a"]


def test_rrf_ignores_duplicate_hits_within_one_list():
    dense = results(["a", "a", "b"], "dense")
    got = by_name(ReciprocalRankFusion(k=60).fuse({"dense": dense}, top_k=5))
    assert got["a"].rrf_score == pytest.approx(1 / 61)


def test_rrf_weights():
    dense = results(["a"], "dense")
    sparse = results(["b"], "sparse")
    fused = ReciprocalRankFusion(k=60, weights={"sparse": 2.0}).fuse({"dense": dense, "sparse": sparse}, 5)
    assert fused[0].chunk_id == IDS["b"]
    assert fused[0].rrf_score == pytest.approx(2 / 61)


def test_rrf_is_rank_based_not_score_based():
    # Wildly different score scales must not matter.
    dense = results(["a", "b"], "dense", scores=[0.99, 0.98])
    sparse = results(["b", "a"], "sparse", scores=[1000.0, 0.001])
    got = by_name(ReciprocalRankFusion().fuse({"dense": dense, "sparse": sparse}, 5))
    assert got["a"].rrf_score == pytest.approx(got["b"].rrf_score)


def test_rrf_invalid_k():
    with pytest.raises(ValueError):
        ReciprocalRankFusion(k=0)


def test_rrf_empty():
    assert ReciprocalRankFusion().fuse({"dense": [], "sparse": []}, 10) == []


def test_relative_score_fusion():
    dense = results(["a", "b"], "dense", scores=[0.9, 0.5])
    sparse = results(["b", "c"], "sparse", scores=[10.0, 2.0])
    got = by_name(RelativeScoreFusion().fuse({"dense": dense, "sparse": sparse}, 5))
    assert got["a"].fused_score == pytest.approx(1.0)
    assert got["b"].fused_score == pytest.approx(0.0 + 1.0)
    assert got["c"].fused_score == pytest.approx(0.0)
