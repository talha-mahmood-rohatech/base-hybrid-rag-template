"""Evaluation dataset + variant definitions (YAML).

Relevance is defined at the *document + passage* level (``external_id`` and an optional
``contains`` snippet), so the same dataset stays valid across chunkers and embedding models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class EvalDocument(BaseModel):
    external_id: str
    path: str | None = None
    text: str | None = None
    filename: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _one_source(self) -> EvalDocument:
        if bool(self.path) == bool(self.text):
            raise ValueError("document needs exactly one of 'path' or 'text'")
        return self

    def load(self, base_dir: Path) -> tuple[str, bytes]:
        if self.path:
            p = (base_dir / self.path).resolve()
            return self.filename or p.name, p.read_bytes()
        assert self.text is not None
        return self.filename or f"{self.external_id}.md", self.text.encode()


class RelevantPassage(BaseModel):
    external_id: str
    contains: str | None = None
    grade: float = 1.0


class EvalQuery(BaseModel):
    id: str
    query: str
    relevant: list[RelevantPassage]
    reference_answer: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)


class EvalDataset(BaseModel):
    name: str
    documents: list[EvalDocument]
    queries: list[EvalQuery]
    base_dir: Path = Path(".")

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalDataset:
        p = Path(path)
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        data.setdefault("base_dir", str(p.parent))
        return cls.model_validate(data)


class Variant(BaseModel):
    """One pipeline configuration to evaluate. Unset fields fall back to platform settings."""

    name: str
    embedding: dict[str, Any] = Field(default_factory=dict)  # EmbeddingSettings fields
    reranker: dict[str, Any] = Field(default_factory=dict)  # RerankerSettings fields
    chunking: dict[str, Any] = Field(default_factory=dict)  # ChunkingSettings fields
    retrieval: dict[str, Any] = Field(
        default_factory=dict
    )  # dense_top_k, sparse_top_k, rrf_k, rrf_top_k, fusion
    top_k: int = 8
    rerank: bool = True
    use_dense: bool = True
    use_sparse: bool = True
    generate_answer: bool = False


class VariantSet(BaseModel):
    variants: list[Variant]
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 8])

    @classmethod
    def from_yaml(cls, path: str | Path) -> VariantSet:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
