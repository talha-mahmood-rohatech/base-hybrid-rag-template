import uuid

import pytest

from app.providers.rerankers.base import PassthroughReranker
from app.retrieval.reranking.stage import RerankingStage
from app.retrieval.types import Candidate, ChunkView
from tests.fakes import FixedReranker, LexicalReranker


def make(n):
    out = []
    for i in range(n):
        cid = uuid.uuid4()
        view = ChunkView(
            chunk_id=cid,
            tenant_id=uuid.uuid4(),
            knowledge_base_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            document_version_id=uuid.uuid4(),
            document_name=f"d{i}",
            document_type="txt",
            source=None,
            chunk_index=0,
            text=f"text {i}",
            token_count=2,
            page=None,
            page_end=None,
            section=f"S{i}",
            checksum=str(i),
        )
        out.append(
            Candidate(
                chunk_id=cid,
                document_id=view.document_id,
                fused_score=1 / (61 + i),
                fused_rank=i + 1,
                chunk=view,
            )
        )
    return out


async def test_reranker_reorders_and_keeps_rrf_scores():
    cands = make(5)
    stage = RerankingStage(FixedReranker([0.1, 0.9, 0.5, 0.95, 0.2]))
    out = await stage.run("q", cands, top_n=3)
    assert [c.chunk.document_name for c in out] == ["d3", "d1", "d2"]
    assert [c.final_rank for c in out] == [1, 2, 3]
    assert [c.reranker_score for c in out] == [0.95, 0.9, 0.5]
    # RRF information is preserved alongside the reranker score
    assert out[0].rrf_score == pytest.approx(1 / 64) and out[0].fused_rank == 4


async def test_reranker_sees_section_header():
    seen = []

    class Spy(LexicalReranker):
        async def score(self, query, documents):
            seen.extend(documents)
            return [0.0] * len(documents)

    await RerankingStage(Spy(), include_section=True).run("q", make(1), top_n=1)
    assert seen == ["S0\ntext 0"]


async def test_ties_broken_by_fused_rank():
    out = await RerankingStage(FixedReranker([0.5, 0.5, 0.5])).run("q", make(3), top_n=3)
    assert [c.fused_rank for c in out] == [1, 2, 3]


async def test_passthrough_keeps_fusion_order_without_scores():
    out = await RerankingStage(PassthroughReranker()).run("q", list(reversed(make(4))), top_n=2)
    assert [c.fused_rank for c in out] == [1, 2]
    assert all(c.reranker_score is None for c in out)


async def test_rerank_disabled_per_request():
    stage = RerankingStage(FixedReranker([0.0, 1.0]))
    out = await stage.run("q", make(2), top_n=2, enabled=False)
    assert [c.fused_rank for c in out] == [1, 2]


async def test_score_count_mismatch_raises():
    with pytest.raises(ValueError):
        await RerankingStage(FixedReranker([0.1])).run("q", make(3), top_n=3)
