from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.evaluation.dataset import EvalDataset, Variant, VariantSet
from app.evaluation.runner import EvaluationRunner, write_reports
from app.models import Tenant

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


async def test_evaluation_compares_variants_and_cleans_up(container, tmp_path):
    from scripts.sample_docs import main as write_samples

    if not (ROOT / "samples" / "sla_agreement.pdf").exists():
        write_samples()
    dataset = EvalDataset.from_yaml(ROOT / "evals" / "datasets" / "northwind.yaml")
    variants = VariantSet(
        k_values=[1, 5],
        variants=[
            Variant(name="hybrid", generate_answer=True),
            Variant(name="sparse-only", use_dense=False),
            Variant(
                name="hybrid-1536d", embedding={"dimension": 1536, "version": "eval"}, retrieval={"rrf_k": 20}
            ),
        ],
    )
    async with container.session_factory() as s:
        tenants_before = (await s.execute(select(func.count()).select_from(Tenant))).scalar()

    reports = await EvaluationRunner(container.settings).run(dataset, variants)
    assert [r.name for r in reports] == ["hybrid", "sparse-only", "hybrid-1536d"]
    hybrid = reports[0]
    for key in (
        "recall@5",
        "precision@1",
        "mrr",
        "ndcg@5",
        "fused_mrr",
        "faithfulness",
        "answer_relevance",
        "citation_precision",
        "citation_validity",
    ):
        assert key in hybrid.metrics, key
        assert 0.0 <= hybrid.metrics[key] <= 1.0
    assert hybrid.metrics["recall@5"] > 0.5
    assert hybrid.metrics["citation_validity"] == 1.0  # never fabricated
    assert reports[2].config["embedding"].endswith("(1536d)")
    assert reports[2].config["rrf_k"] == 20

    md = write_reports(reports, dataset, tmp_path, variants.k_values)
    assert "| hybrid |" in md.read_text()

    async with container.session_factory() as s:
        assert (await s.execute(select(func.count()).select_from(Tenant))).scalar() == tenants_before
