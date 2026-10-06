from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import TenantContext, get_container, get_tenant
from app.container import Container
from app.rag.orchestrator import QueryOptions
from app.schemas.api import (
    CitationOut,
    QueryRequest,
    QueryResponse,
    RetrievalCounts,
    RetrievedChunkOut,
    UsageOut,
)

router = APIRouter(prefix="/v1/rag", tags=["rag"])


@router.post("/query", response_model=QueryResponse)
async def rag_query(
    body: QueryRequest,
    tenant: TenantContext = Depends(get_tenant),
    container: Container = Depends(get_container),
) -> QueryResponse:
    opts = body.options
    options = QueryOptions.from_settings(
        container.settings.retrieval,
        dense_top_k=opts.dense_top_k,
        sparse_top_k=opts.sparse_top_k,
        rrf_k=opts.rrf_k,
        rrf_top_k=opts.rrf_top_k,
        fusion=opts.fusion,
        top_k=body.top_k,
        rerank=opts.rerank,
        generate_answer=body.generate_answer,
        use_dense=opts.dense,
        use_sparse=opts.sparse,
    )
    result = await container.orchestrator.query(
        tenant_id=tenant.tenant_id,
        knowledge_base_id=body.knowledge_base_id,
        query=body.query,
        filters=body.filters,
        options=options,
    )
    results = []
    if body.include_results:
        for c in result.results:
            assert c.chunk is not None
            results.append(
                RetrievedChunkOut(
                    rank=c.final_rank or 0,
                    chunk_id=str(c.chunk_id),
                    document_id=str(c.document_id),
                    document_name=c.chunk.document_name,
                    page=c.chunk.page,
                    section=c.chunk.section,
                    text=c.chunk.text,
                    dense_rank=c.dense_rank,
                    sparse_rank=c.sparse_rank,
                    rrf_score=c.fused_score,
                    reranker_score=c.reranker_score,
                )
            )
    usage = None
    if result.llm is not None:
        usage = UsageOut(
            model=result.llm.model,
            prompt_tokens=result.llm.usage.prompt_tokens,
            completion_tokens=result.llm.usage.completion_tokens,
            latency_ms=result.llm.latency_ms,
        )
    return QueryResponse(
        answer=result.answer,
        citations=[CitationOut(**c.to_dict()) for c in result.citations],
        trace_id=result.trace_id,
        retrieval=RetrievalCounts(**result.counts),
        results=results,
        usage=usage,
        degraded=result.degraded,
        latency_ms=result.timings_ms.get("total_ms", 0.0),
    )
