"""CLI: python -m app.evaluation --dataset evals/datasets/northwind.yaml --variants evals/variants.yaml"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.evaluation.dataset import EvalDataset, VariantSet
from app.evaluation.runner import EvaluationRunner, write_reports


async def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Evaluate and compare RAG pipeline variants")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--out", default="reports")
    parser.add_argument("--only", nargs="*", help="run only these variant names")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging("WARNING")
    dataset = EvalDataset.from_yaml(args.dataset)
    variants = VariantSet.from_yaml(args.variants)
    if args.only:
        variants.variants = [v for v in variants.variants if v.name in set(args.only)]
    reports = await EvaluationRunner(settings).run(dataset, variants)
    md = write_reports(reports, dataset, args.out, variants.k_values)
    print(md.read_text(encoding="utf-8"))
    print(f"Reports written to {md.parent}/")
    return 0


def main() -> None:
    sys.exit(asyncio.run(_main(sys.argv[1:])))


if __name__ == "__main__":
    main()
