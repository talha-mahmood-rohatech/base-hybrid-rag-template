# Reusing the Hybrid RAG Platform

This guide is for teams that want to put retrieval-augmented answers into **their own product**
on top of this platform. It has three parts:

- **Part A, integrating a product:** the HTTP API workflow. No platform code changes needed.
- **Part B, configuring and operating:** models, deployment, tuning, evaluation.
- **Part C, extending:** adding a new embedding model, reranker, LLM, file format or fusion strategy.

For the internals, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Part A: Integrating a product

### A1. The model your product works with

| Concept | What it means for you |
|---|---|
| **Tenant** | One per customer organisation (or one per product, if your product is single-tenant). Data never crosses tenants. |
| **API key** | Identifies the tenant. Keep it on your **backend**; never ship it to browsers or mobile apps. |
| **Knowledge base (KB)** | A searchable collection of documents. Create one per use case or corpus, e.g. `support-articles`, `hr-policies`. Each query targets exactly one KB. |
| **Document** | Identified by your stable `external_id`. Re-uploading the same `external_id` with changed content creates a new **version**; the old version stops being searchable once the new one is ingested. |
| **Chunk** | The retrievable unit. You get chunks back in `results` and `citations`. |

Typical layout: your product backend → this platform's API. Your backend authenticates end users,
decides which tenant and KB they may use, and passes their permissions as filters.

### A2. Onboarding (once per tenant)

```bash
# Platform admin creates the tenant; the API key is returned ONCE - store it in your secret manager.
curl -X POST $RAG/v1/admin/tenants \
  -H "X-Admin-Key: $ADMIN_KEY" -H "content-type: application/json" \
  -d '{"name": "Acme Corp", "slug": "acme"}'
# -> {"id": "...", "api_key": "rag_..."}

# Create a knowledge base (tenant key from here on)
curl -X POST $RAG/v1/knowledge-bases \
  -H "X-API-Key: $KEY" -H "content-type: application/json" \
  -d '{"name": "support-articles", "fts_language": "english",
       "chunking": {"chunk_size": 480, "chunk_overlap": 72}}'
```

KB options:

- `fts_language`: any PostgreSQL text-search configuration (`english`, `german`, `french`,
  `simple`, ...). It controls stemming and stop words for BM25.
- `chunking`: optional per-KB override of the platform chunking defaults.
- `embedding_config_id`: optional. Pins a specific embedding model; defaults to the platform default.

### A3. Loading documents

Upload with `multipart/form-data`. Supported formats: **PDF, DOCX, Markdown, HTML, TXT**.

```bash
curl -X POST $RAG/v1/documents -H "X-API-Key: $KEY" \
  -F knowledge_base_id=$KB \
  -F external_id=kb-article-1042 \
  -F file=@refund-policy.pdf \
  -F source=helpcenter \
  -F 'metadata={"product": "billing", "region": "eu"}' \
  -F 'permissions=["public"]'
```

Alternatively, send raw text with `-F text="..." -F filename=article.md`.

| Field | Guidance |
|---|---|
| `external_id` | **Use your system's stable id** (CMS article id, file path, URL). It drives versioning; without it the filename is used. |
| `metadata` | Free-form JSON object, stored on every chunk. Filter with `metadata.<key>`. |
| `permissions` | List of principal strings (e.g. `group:finance`, `user:42`, `public`). Default `["public"]`. |
| `source` | Free text (system of origin); filterable. |
| `force=true` | Re-ingest even if the content is unchanged (e.g. after changing chunking settings). |

The response tells you what happened:

| `outcome` | Meaning |
|---|---|
| `created` | New document, version 1 queued |
| `new_version` | Content changed, version n+1 queued; the previous version stays live until it is ingested |
| `unchanged` | Same checksum as an existing version: nothing to do (safe to re-sync everything nightly) |
| `requeued` | `force=true` or a previously failed version: queued again |

Ingestion is asynchronous. Poll the job until it finishes:

```bash
curl $RAG/v1/ingestion/jobs/$JOB_ID -H "X-API-Key: $KEY"
# status: queued | running | succeeded | failed ; stats: chunks, tokens, timings ; error on failure
```

Other operations:

- **Delete:** `DELETE /v1/documents/{id}` removes all versions, chunks and vectors.
- **Re-ingest:** `POST /v1/ingestion/jobs {"document_id": "..."}` re-processes the latest version.

**Sync pattern:** run your connector periodically and upload everything. Unchanged documents cost
one checksum comparison.

### A4. Asking questions

```bash
curl -X POST $RAG/v1/rag/query -H "X-API-Key: $KEY" -H "content-type: application/json" -d '{
  "knowledge_base_id": "'$KB'",
  "query": "Can I get a refund on an annual plan?",
  "filters": {"metadata.product": "billing", "permissions": ["public", "group:finance"]},
  "top_k": 8,
  "generate_answer": true
}'
```

Response:

| Field | Use it for |
|---|---|
| `answer` | Text with inline markers like `[1]`. Every marker refers to an entry in `citations`. |
| `citations[]` | `citation_id`, `document_id`, `document_name`, `document_version_id`, `chunk_id`, `page`, `page_end`, `section`, `source`, `snippet`, `rrf_score`, `reranker_score`. Render them as source links or footnotes. |
| `results[]` | The top reranked chunks (full text, ranks, scores). Hide them from end users, or use them for "related passages". Omit with `"include_results": false`. |
| `retrieval` | Candidate counts per stage (`dense_candidates`, `sparse_candidates`, `rrf_candidates`, `reranked_candidates`, `context_chunks`). |
| `trace_id` | Log it with your request; use it for debugging and user feedback. |
| `degraded` | `true` if one retriever failed and the answer used only the other. |

**Rendering citations.** Replace each `[n]` in `answer` with a link or footnote built from
`citations[n]`, for example `document_name`, plus `page` for PDFs or `section` for structured
documents. The platform has already removed any `[n]` the model invented, so every remaining
marker is backed by a supplied chunk.

**No answer.** When the KB holds no relevant content, `answer` is exactly
`"I don't know based on the provided documents."` and `citations` is empty. Detect this to show a
fallback (contact support, search page, ...).

**Retrieval only.** Set `"generate_answer": false` to get ranked, reranked chunks without calling
the LLM. Use this for search UIs, or when your product calls its own LLM with your own prompt.

### A5. Filters and access control

Filters are AND-ed. Each value is either a scalar (equality), a list (any-of), or a range object.

```json
{
  "document_type": "pdf",
  "source": ["helpcenter", "wiki"],
  "document_id": "<uuid>",
  "page": {"gte": 2, "lte": 10},
  "section": "Refund Policy > Eligibility",
  "permissions": ["public", "group:finance", "user:42"],
  "metadata.region": "eu",
  "metadata.priority": {"gte": 3}
}
```

**Access control is your backend's job.** The platform enforces *tenant* isolation by itself.
*Within* a tenant, document-level permissions are applied only when you send a `permissions`
filter. Always send the end user's principals, including `"public"` if public documents should
be visible. You cannot filter on `tenant_id` or `knowledge_base_id`: that returns 422, because
isolation always comes from the API key.

### A6. Per-request tuning

```json
"options": {"dense_top_k": 50, "sparse_top_k": 50, "rrf_k": 60, "rrf_top_k": 40,
            "fusion": "rrf", "rerank": true, "dense": true, "sparse": true}
```

The defaults are good for most products. Common adjustments:

| Need | Setting |
|---|---|
| Lower latency, accepting some quality loss | `"rerank": false` (skips the cross-encoder, keeps RRF order) |
| Exact codes or IDs dominate the queries | Keep both retrievers; BM25 handles exact tokens. Use `"dense": false` only for diagnostics. |
| Smaller LLM context | Lower `top_k` (e.g. 4) |

### A7. Debugging an answer

```bash
curl $RAG/v1/traces/$TRACE_ID -H "X-API-Key: $KEY"
```

The trace shows, for every candidate, its dense rank, sparse rank, RRF score and reranker score.
It also lists why chunks were dropped (`near_duplicate`, `token_budget`, ...), the exact model and
token usage, timings per stage, and any invalid citations the model attempted. Typical diagnoses:

| Symptom | Look at | Likely fix |
|---|---|---|
| Right document never appears | `dense_results` and `sparse_results` | Content not ingested (check job), filters too strict, wrong KB |
| Retrieved but ranked low | `fused_results` vs `reranked_results` | Chunk too large or noisy (adjust chunking); try another reranker model |
| Retrieved but not in context | `dropped_chunks` | Raise `CONTEXT__MAX_CONTEXT_TOKENS` or lower `top_k` |
| Answer ignores good context | `llm.raw_answer`, `selected_chunks` | Use a stronger LLM |

### A8. Errors

All errors look like `{"error": {"code": "...", "message": "...", "details": {}}}`.

| HTTP | `code` | When |
|---|---|---|
| 401 | `unauthorized` | Missing or invalid `X-API-Key` / `X-Admin-Key` |
| 404 | `not_found` | Resource does not exist **or belongs to another tenant** |
| 409 | `conflict` | Duplicate KB name or tenant slug; same `external_id` uploaded with a different file type |
| 422 | `validation_error`, `unsupported_document`, `document_parse_error` | Bad input, unknown filter, unsupported or unparseable file |
| 502 | `provider_error` | Embedding, reranker, LLM or store call failed. Retry with backoff. |

### A9. Minimal Python client

```python
import httpx, time

class RagClient:
    def __init__(self, base_url: str, api_key: str):
        self.http = httpx.Client(base_url=base_url, headers={"X-API-Key": api_key}, timeout=300)

    def upload(self, kb_id: str, external_id: str, path: str, metadata: dict | None = None,
               permissions: list[str] | None = None) -> dict:
        import json
        with open(path, "rb") as f:
            r = self.http.post("/v1/documents", files={"file": (path.split("/")[-1], f)}, data={
                "knowledge_base_id": kb_id, "external_id": external_id,
                "metadata": json.dumps(metadata or {}), "permissions": json.dumps(permissions or ["public"]),
            })
        r.raise_for_status()
        return r.json()

    def wait(self, job_id: str, poll_s: float = 2.0) -> dict:
        while True:
            job = self.http.get(f"/v1/ingestion/jobs/{job_id}").json()
            if job["status"] in ("succeeded", "failed"):
                return job
            time.sleep(poll_s)

    def ask(self, kb_id: str, question: str, principals: list[str], **filters) -> dict:
        r = self.http.post("/v1/rag/query", json={
            "knowledge_base_id": kb_id, "query": question,
            "filters": {"permissions": principals, **filters}, "include_results": False,
        })
        r.raise_for_status()
        return r.json()
```

---

## Part B: Configuring and operating

### B1. Deploy

```bash
cp .env.example .env     # set SECURITY__ADMIN_API_KEY and choose providers (below)
docker compose up -d --build
```

- The services are `postgres`, `qdrant`, `migrate` (runs Alembic once), `api`, and optionally
  `ollama` + `ollama-pull` (profile `local-llm`) and `worker` (profile `worker`).
- Data persists in the `pg_data`, `qdrant_data`, `blob_data`, `model_cache` and `ollama_data`
  volumes.
- Back up PostgreSQL and Qdrant together. PostgreSQL is the source of truth, but vectors are
  expensive to recompute.

### B2. Choosing models (no code changes)

| Profile | Settings |
|---|---|
| Fully local (default) | `EMBEDDING__PROVIDER=fastembed`, `EMBEDDING__MODEL=mixedbread-ai/mxbai-embed-large-v1`, `EMBEDDING__DIMENSION=1024`; `RERANKER__PROVIDER=fastembed`, `RERANKER__MODEL=jinaai/jina-reranker-v2-base-multilingual`; `LLM__PROVIDER=openai_compatible`, `LLM__BASE_URL=http://ollama:11434/v1`, `LLM__MODEL=qwen2.5:3b` |
| OpenAI embeddings (3072-d) | `EMBEDDING__PROVIDER=openai`, `EMBEDDING__MODEL=text-embedding-3-large`, `EMBEDDING__DIMENSION=3072`, `EMBEDDING__API_KEY=...`. Chunking can return to 800/120. |
| Claude for answers | `LLM__PROVIDER=anthropic`, `LLM__MODEL=claude-opus-5-5`, `LLM__API_KEY=...` (optional: `LLM__EFFORT=low\|medium\|high`) |
| Hosted reranker | `RERANKER__PROVIDER=cohere`, `RERANKER__MODEL=rerank-v3.5`, `RERANKER__API_KEY=...` |
| Self-hosted OpenAI-compatible LLM (vLLM, LM Studio) | `LLM__PROVIDER=openai_compatible`, `LLM__BASE_URL=http://host:8000/v1`, `LLM__MODEL=...` |
| BM25 alternative | `RETRIEVAL__SPARSE_PROVIDER=postgres_fts` |

For fastembed embedding models, set `EMBEDDING__QUERY_PREFIX` / `EMBEDDING__DOCUMENT_PREFIX` if the
model needs instruction prefixes. Known models already have defaults in
`providers/embeddings/fastembed_provider.py`.

### B3. Changing the embedding model safely

1. Set the new `EMBEDDING__*` values and restart. A new `embedding_configs` row and Qdrant
   collection are created and become the default **for new KBs only**.
2. Existing KBs keep answering from their original model and collection. Vector spaces are never
   mixed.
3. To migrate a corpus, create a new KB (it picks up the new default), re-upload the documents,
   compare both KBs with the evaluation tool (B5), then switch your product's `knowledge_base_id`.

### B4. Scaling

| Pressure | Action |
|---|---|
| Ingestion throughput | Set `WORKER__EMBEDDED=false` on `api` and run N `worker` containers (`docker compose --profile worker up -d --scale worker=N`). Jobs are claimed with `SKIP LOCKED`. |
| Query throughput | Run more `api` replicas behind a load balancer. They are stateless; models load per replica. |
| Latency | Use a GPU or hosted LLM (generation dominates), reduce `top_k`, use a smaller reranker (`Xenova/ms-marco-MiniLM-L-12-v2`), or set `rerank:false` for latency-critical paths. |
| Large vector collections | `QDRANT__ON_DISK_VECTORS=true`, tune `QDRANT__HNSW_M` / `QDRANT__SEARCH_HNSW_EF`, or run a Qdrant cluster. |
| Multi-node file storage | Implement `BlobStore` for S3/GCS (`app/ingestion/blobstore.py`). |

### B5. Measuring quality for your product

1. Write a golden set like `evals/datasets/northwind.yaml`: your documents plus 30–100 real user
   questions. Relevance is defined by `external_id` plus a `contains` snippet, so the set
   survives chunking and model changes.
2. List the configurations to compare in a variants file like `evals/variants.yaml`: embedding
   model, reranker, `rrf_k`, top-k values, chunking, dense-only or sparse-only.
3. Run the comparison:

   ```bash
   python -m app.evaluation --dataset evals/datasets/my_product.yaml --variants evals/my_variants.yaml
   ```

4. Read `reports/eval-*.md`:
   - **Retrieval:** Recall@K, Precision@K, MRR, NDCG.
   - **Reranker uplift:** `fused_mrr` (before reranking) vs `mrr` (after).
   - **Answers** (variants with `generate_answer: true`): faithfulness, answer relevance,
     citation precision and citation validity.

Each variant runs in a throwaway tenant, so this is safe against a production database.

### B6. Operational checklist

- [ ] `SECURITY__ADMIN_API_KEY` changed. The app refuses the default in staging and production.
- [ ] `ENVIRONMENT=production`, `LOG_LEVEL=INFO` (logs are JSON on stdout).
- [ ] Monitor `/health` (PostgreSQL + Qdrant), job failure rate (`ingestion_jobs.status='failed'`),
      and p95 of `retrieval_traces.total_latency_ms`.
- [ ] Retention policy for `retrieval_traces` (they contain queries and answers).
- [ ] Backups for PostgreSQL and Qdrant; blob volume included.

---

## Part C: Extending the platform

Every pluggable piece follows the same three steps:

1. **Implement** the interface (`providers/<kind>/base.py`, or `retrieval/fusion/base.py` for
   fusion).
2. **Register** it in `app/providers/registry.py` (and the loader registry for file formats).
3. **Allow** the new name in the `Literal[...]` of the matching settings class in
   `app/core/config.py`.

The orchestrator, API and tests need no changes. Add a unit test next to the existing ones.

### C1. New embedding provider (e.g. Voyage, Bedrock, a GPU service)

```python
# app/providers/embeddings/myvendor_provider.py
class MyVendorEmbeddingProvider(EmbeddingProvider):
    def __init__(self, spec: EmbeddingSpec, *, api_key: str, **kw):
        super().__init__(spec, max_input_tokens=8192)
        self._client = httpx.AsyncClient(...)

    async def _embed_documents(self, texts: list[str]) -> list[Sequence[float]]:
        ...  # batch call; return one vector per text, in order

    async def _embed_query(self, text: str) -> Sequence[float]:
        ...
```

```python
# app/providers/registry.py -> build_embedding_provider
if spec.provider == "myvendor":
    from app.providers.embeddings.myvendor_provider import MyVendorEmbeddingProvider
    return MyVendorEmbeddingProvider(spec, api_key=_secret(settings.api_key))
```

```python
# app/core/config.py -> EmbeddingSettings
provider: Literal["fastembed", "openai", "hashing", "myvendor"] = "fastembed"
```

You don't write dimension checks: the base class validates every vector against
`spec.dimension`, and Qdrant upserts validate again.

### C2. New reranker

```python
class MyReranker(Reranker):
    provider = "myvendor"
    def __init__(self, model: str, ...): self.model = model
    async def score(self, query: str, documents: Sequence[str]) -> list[float]:
        ...  # one score per document, same order; higher = more relevant
```

Register it in `build_reranker` and add the name to `RerankerSettings.provider`. Scores only need
to be comparable within one call.

### C3. New LLM

```python
class MyLLM(LLMProvider):
    provider = "myvendor"
    async def generate(self, messages: list[ChatMessage], *, max_tokens: int, temperature: float = 0.0) -> LLMResponse:
        ...  # return LLMResponse(text, model, provider, LLMUsage(prompt, completion), latency_ms, finish_reason)
```

Register it in `build_llm` and add the name to `LLMSettings.provider`. The system prompt and
citation rules live in `app/generation/prompts.py` and apply to every provider.

### C4. New file format (e.g. PPTX, XLSX, email)

```python
# app/ingestion/loaders/pptx.py
class PPTXLoader(DocumentLoader):
    document_type = "pptx"
    extensions = (".pptx",)
    mime_types = ("application/vnd.openxmlformats-officedocument.presentationml.presentation",)

    def load(self, data: bytes, filename: str | None = None) -> ParsedDocument:
        # one Segment per slide: Segment(text=..., page=slide_no, section=slide_title)
        ...
```

Add `PPTXLoader()` to the default list in `app/ingestion/loaders/registry.py`. Emit `page` and/or
`section` on segments. The chunker, citations and filters use them automatically.

### C5. New fusion strategy

Subclass `FusionStrategy` (`retrieval/fusion/base.py`). Use the `collect()` and `finalize()`
helpers so ranks and scores are preserved and tie-breaking stays deterministic. Then register it
in `build_fusion` and add the name to `RetrievalSettings.fusion` **and**
`RetrievalOptions.fusion` (`app/schemas/api.py`) to expose it per request.

### C6. Other extension points

| Need | Where |
|---|---|
| Different chunking (semantic, per-format) | Implement `Chunker` (`ingestion/chunking/base.py`) and select it in `IngestionPipeline.chunker_for` |
| Another sparse engine (OpenSearch, Qdrant sparse vectors) | Implement `SparseSearch`; honour `query.scope` and `query.filters` |
| Another vector DB | Implement `VectorStore`; always apply the scope filter. The isolation tests in `tests/unit/test_embeddings_and_qdrant.py` show the required behaviour. |
| Object storage for uploads | Implement `BlobStore` (`ingestion/blobstore.py`) |
| SSO / JWT instead of API keys | Replace `get_tenant` in `app/api/deps.py`; it must return a `TenantContext` derived from verified credentials |
| Custom answer prompt | `app/generation/prompts.py` (keep the `[n]` citation contract, or update `generation/citations.py` with it) |

### C7. Testing your extension

```bash
pytest tests/unit                       # fast, no services
docker compose up -d postgres qdrant
pytest                                  # adds integration tests on real PostgreSQL + Qdrant
```

Integration tests build the whole platform with deterministic test doubles (`hashing` embeddings,
`tests/fakes.py` reranker, `extractive` LLM). To test your provider end to end, construct a
`Container` with it: `Container.create(settings, reranker=MyReranker(...))`, or set the settings
that select it, and reuse the helpers in `tests/integration/helpers.py`.
