# Hybrid RAG Platform

A reusable, multi-tenant **hybrid retrieval-augmented generation** platform that product teams
call over HTTP. It is a modular monolith (FastAPI + PostgreSQL + Qdrant). No Redis, Kafka or
Elasticsearch is required.

**Documentation:**

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): how it is built (data model, ingestion and query
  pipelines, BM25/RRF/reranking details, tenancy model, design decisions).
- [docs/REUSE_GUIDE.md](docs/REUSE_GUIDE.md): how to reuse it. Integrating a product over the
  API, choosing models and operating it, and extending it with new providers or formats.
- [docs/ZTBL.md](docs/ZTBL.md): worked example. ZTBL Islamic Banking knowledge base answered
  with Groq.

```text
                         Product Apps
                             |
                     RAG API (FastAPI, API-key -> tenant)
                             |
                     RAG Orchestrator  (app/rag/orchestrator.py)
                             |
                +------------+------------+          asyncio.gather (concurrent)
                |                         |
          Dense Retrieval           Sparse Retrieval
       (query embedding ->       (Okapi BM25 computed in
        Qdrant ANN, top 50)       PostgreSQL over tsvector, top 50)
                |                         |
                +------------+------------+
                             |
               Reciprocal Rank Fusion (k=60) -> top 40
                             |
           Hydrate from PostgreSQL (source of truth, active versions only)
                             |
           Cross-encoder reranker (jina-reranker-v2) -> top 8
                             |
         Context builder (dedup, token budget, grouped ordering, [n] ids)
                             |
                            LLM  (only supplied context, must cite [n])
                             |
          Answer + validated structured citations + persisted trace
```

```text
Upload (PDF/DOCX/MD/HTML/TXT) -> checksum / version -> ingestion job (PostgreSQL queue)
   -> Load -> Parse -> Clean -> Structure-aware chunk -> Embed
   -> Qdrant (vectors + filter payload)  +  PostgreSQL (chunk text, metadata, tsvector)
   -> atomic activation of the new version
```

## What is implemented

| Area | Implementation |
|---|---|
| Vector DB | **Qdrant only** (`QdrantVectorStore` behind `VectorStore`). One collection **per embedding configuration** (provider/model/dimension/distance/version), shared by all tenants. `tenant_id` payload index uses `is_tenant=True`. Indexes on `knowledge_base_id`, `document_id`, `document_version_id`, `chunk_id`, `document_type`, `source`, `section`, `permissions`, `page`, `is_active`. |
| Embeddings | `EmbeddingProvider` interface: `fastembed` (local ONNX, default `mixedbread-ai/mxbai-embed-large-v1`, 1024-d), `openai` (any OpenAI-compatible API; sends `dimensions` for 1536/3072), `hashing` (deterministic, tests). The dimension is validated on every embed and every upsert. Existing collections are checked against their config. No PCA. |
| Sparse | `SparseSearch` interface: `postgres_bm25` (default, true **Okapi BM25** computed in SQL from `tsvector` term frequencies with KB-scoped N/avgdl/df) and `postgres_fts` (`ts_rank_cd`). GIN index. |
| Fusion | `FusionStrategy` interface: `ReciprocalRankFusion` (default, `RRF(d)=Σ w_i/(k+rank_i(d))`, k=60) and `RelativeScoreFusion`. Dense rank, sparse rank and RRF score are kept on each candidate. |
| Reranking | `Reranker` interface: `fastembed` cross-encoder (default `jinaai/jina-reranker-v2-base-multilingual`), `cohere` (Rerank API), `none`. Both `rrf_score` and `reranker_score` are returned and traced. |
| Generation | `LLMProvider` interface: `openai_compatible` (OpenAI, vLLM, **Ollama**), `groq` (Groq Cloud), `anthropic` (Claude, default `claude-opus-5-5`), `extractive` (offline, quotes context verbatim). |
| Citations | The context is numbered `[1]..[n]` with document/section/page headers. Citation markers in the answer are parsed and validated. **Fabricated ids are stripped** and recorded in the trace. Each citation returns `document_id, document_name, document_version_id, chunk_id, page, section, source, snippet, rrf_score, reranker_score`. |
| Multi-tenancy | Tenant → KnowledgeBase → Document → DocumentVersion → Chunk. Every row/vector carries `tenant_id` + `knowledge_base_id`. The tenant is derived **only from the API key**. Filters on `tenant_id`/`knowledge_base_id` are rejected (422). Other tenants' resources return 404. Qdrant results are re-checked, and hydration re-filters by tenant in SQL. |
| Ingestion | Checksums (SHA-256), idempotent uploads (`unchanged`), versioning with active/inactive versions, deterministic chunk ids (safe retries), page/section/char offsets, metadata and permissions preserved on chunks and payloads. Jobs run on a PostgreSQL queue (`FOR UPDATE SKIP LOCKED`) with retries and backoff, either in the API process or as a separate worker. |
| Observability | One `retrieval_traces` row per request: query, filters, config (models, k values), dense/sparse ranks and scores, fused list with RRF scores, reranker scores, selected chunks with a `why` block, dropped chunks with a reason, context tokens, LLM model/usage/latency, raw and cleaned answer, citations, stage timings, errors/degradation. |
| Voice | Ported from the Leap voice agent. The browser streams 16 kHz PCM over `/v1/voice/ws`; **Silero VAD** (ONNX, no torch) on the server decides turns, with an echo hold and barge-in. **Groq Whisper** STT has a reliability gate and a domain-vocabulary hint. The hybrid RAG answers with citations, and **Soniox** TTS speaks it, streamed in growing sentence groups and cached on disk. REST helpers: `/v1/voice/transcribe`, `/speak`, `/ask`. Browser demo at `/voice/`. |
| Evaluation | Recall@K, Precision@K, Hit@K, MRR, NDCG@K, pre-rerank MRR (reranker uplift), faithfulness, answer relevance, citation precision and validity. Compares embedding models, rerankers, fusion params, dense/sparse-only, and chunking variants on isolated temporary tenants. |

## Quick start (Docker)

```bash
cp .env.example .env            # set SECURITY__ADMIN_API_KEY
docker compose up -d --build    # postgres, qdrant, migrate, api (+ ollama via COMPOSE_PROFILES=local-llm)
docker compose logs -f api      # first start downloads the embedding + reranker models (~2 GB)
python scripts/demo.py --api http://localhost:8000 --admin-key <your admin key>
```

**Voice pages:**

- **User page: http://localhost:8000/** (redirects to `/voice/`). One mic button and the conversation:
  tap, ask, hear the answer. No keys or setup. It answers from `VOICE__PUBLIC_KNOWLEDGE_BASE_ID`; or
  share a link `/voice/?key=<tenant key>&kb=<kb id>`, which is removed from the address bar on load.
- **Developer console: http://localhost:8000/voice/console.html.** Shows the connection settings, the
  turn stages with timings, transcript confidence, cited sources, a latency breakdown, replay, and the
  retrieval trace (dense and sparse ranks, RRF and reranker scores).

`scripts/ui_e2e.py` tests both pages in a real Edge or Chrome, with a TTS-spoken question as the
fake microphone (`pip install playwright`). To run the voice pipeline end to end without a microphone, use `scripts/voice_e2e.py`, which speaks the questions with TTS.

Services and volumes: `postgres` (`pg_data`), `qdrant` (`qdrant_data`), `api` (models in
`model_cache`, raw uploads in `blob_data`), `ollama` (`ollama_data`). `migrate` runs
`alembic upgrade head` before the API starts. To run ingestion in a dedicated process, set
`WORKER__EMBEDDED=false` and enable the `worker` profile.

`scripts/demo.py` runs the complete flow against the live stack: it uploads all five formats,
waits for the jobs, inspects Qdrant, runs a query, prints ranks, scores, the answer and its
citations, fetches the trace and checks tenant isolation.

## API

All `/v1` endpoints except `/v1/admin/*` require `X-API-Key: <tenant key>`. Admin endpoints require
`X-Admin-Key`. OpenAPI is served at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/admin/tenants` | Create tenant, returns its API key (shown once) |
| POST | `/v1/knowledge-bases` | Create KB (`name`, `fts_language`, optional `chunking` overrides, optional `embedding_config_id`) |
| POST | `/v1/documents` | Multipart upload (`file` or `text`, `knowledge_base_id`, `external_id`, `metadata` JSON, `permissions` JSON, `source`, `force`). Returns outcome `created`/`new_version`/`unchanged`/`requeued` and the job. |
| GET/DELETE | `/v1/documents/{id}` | Document with versions / delete (also removes vectors) |
| POST | `/v1/ingestion/jobs` | (Re-)ingest `document_id` (latest version) or `document_version_id` |
| GET | `/v1/ingestion/jobs/{id}` | Job status, attempts, error, stats, stage timings |
| POST | `/v1/rag/query` | Hybrid RAG query |
| GET | `/v1/traces/{trace_id}` | Full retrieval trace (tenant-scoped) |
| WS | `/v1/voice/ws` | Voice conversation: PCM in, transcripts, cited answers and speech out (protocol in `app/voice/messages.py`) |
| POST | `/v1/voice/transcribe`, `/v1/voice/speak`, `/v1/voice/ask` | Speech-to-text, text-to-speech, and one-shot spoken question to spoken answer |
| GET | `/health` | PostgreSQL + Qdrant checks and active model configuration |

```json
POST /v1/rag/query
{
  "knowledge_base_id": "6f0e...",
  "query": "What is our refund policy?",
  "filters": {"document_type": "pdf", "metadata.department": "finance", "page": {"gte": 2}},
  "top_k": 8,
  "generate_answer": true,
  "options": {"dense_top_k": 50, "sparse_top_k": 50, "rrf_k": 60, "rrf_top_k": 40, "rerank": true}
}
```

```json
{
  "answer": "Annual subscriptions cancelled within 30 days receive a full refund [1]...",
  "citations": [{"citation_id": 1, "document_id": "...", "document_name": "refund_policy.md",
                 "chunk_id": "...", "page": null, "section": "Refund Policy > Eligibility", "...": "..."}],
  "trace_id": "...",
  "retrieval": {"dense_candidates": 50, "sparse_candidates": 50, "rrf_candidates": 40,
                "reranked_candidates": 8, "context_chunks": 8},
  "results": [{"rank": 1, "dense_rank": 2, "sparse_rank": 1, "rrf_score": 0.0325, "reranker_score": 2.91, "...": "..."}]
}
```

**Filters** (AND-ed): `document_id`, `document_version_id`, `document_type`, `source`, `section`,
`page` (eq or `{gt,gte,lt,lte}`), `permissions` (any-of), and `metadata.<key>` (eq, any-of or
range). Each filter is applied the same way to Qdrant payloads and to PostgreSQL. In SQL, field
names are whitelisted and values are always bound parameters.

## Configuration

Settings are resolved in this order: environment variables / `.env` (nested with `__`, e.g.
`RETRIEVAL__RRF_K=60`), then `config/rag.yaml`, then code defaults (`app/core/config.py`).
Models and providers can be changed without code changes:

```bash
EMBEDDING__PROVIDER=openai  EMBEDDING__MODEL=text-embedding-3-large  EMBEDDING__DIMENSION=3072  EMBEDDING__API_KEY=...
RERANKER__PROVIDER=cohere   RERANKER__MODEL=rerank-v3.5  RERANKER__API_KEY=...
LLM__PROVIDER=anthropic     LLM__MODEL=claude-opus-5-5   LLM__API_KEY=...
```

The default embedding config is registered in `embedding_configs` at startup. New KBs are pinned
to it, and existing KBs keep their own config and collection. A new model therefore never mixes
vector spaces: create a new KB, or re-ingest into one, to migrate.

**Chunking:** the platform default is 800/120 tokens (`config/rag.yaml`). `mxbai-embed-large-v1`
reads at most 512 tokens, so the local stack's `.env.example` uses 480/72, and the API logs a
warning if `chunk_size` exceeds the embedder's limit. Use 800/120 with long-context embedders.

## Development

```bash
py -3.11 -m venv .venv && .venv/Scripts/pip install -e ".[local-models,anthropic,dev]"   # Python 3.11 only
docker compose up -d postgres qdrant
alembic upgrade head
uvicorn app.main:app_factory --factory --reload
```

CLI: `python -m app.cli create-tenant --name Acme --slug acme`,
`python -m app.cli ingest-dir --tenant acme --kb policies samples/`,
`python -m app.cli query --tenant acme --kb policies "What is the refund window?"`.

### Tests

```bash
pytest                         # unit + integration (integration needs postgres + qdrant from compose)
pytest tests/unit              # no services needed
RUN_MODEL_TESTS=1 FASTEMBED_CACHE_PATH=.cache/models pytest -m models   # real mxbai + jina reranker
```

The integration suite migrates a separate `rag_test` database from scratch (downgrade, then
upgrade) and uses real PostgreSQL and Qdrant. It covers:

- BM25 scores checked against an independent Python BM25 (`rel=1e-9`);
- exact RRF arithmetic over real dense and sparse lists;
- concurrency of dense and sparse retrieval, and degraded mode;
- tenant isolation at the store, orchestrator and API level;
- versioning and activation, idempotent re-ingestion, retries with a terminal failure;
- the end-to-end HTTP flow with all formats, traces and filters;
- the evaluation runner.

### Evaluation

```bash
python -m app.evaluation --dataset evals/datasets/northwind.yaml --variants evals/variants.yaml
```

Each variant ingests the dataset into its own temporary tenant and KB, with its own embedding
config and Qdrant collection, and the data is removed afterwards. Results are written to
`reports/eval-*.md|json`. Relevance is defined as *document + passage snippet*, so a golden set
stays valid across chunkers and embedding models. Faithfulness and answer relevance use a
deterministic heuristic judge by default. `LLMJudge` is available for model-graded scoring.

## Project layout

```text
app/
  api/            routes, auth deps, serializers
  core/           config, errors, logging, tokens, security
  db/ models/     SQLAlchemy base/session, ORM entities
  ingestion/      loaders/ (pdf, docx, md, html, txt), cleaning, chunking/, pipeline, service, worker, blobstore
  retrieval/      dense.py, sparse.py, hybrid.py, fusion/ (rrf, relative_score), reranking/, filters, repository
  generation/     context builder, prompts, citation validation
  providers/      embeddings/, vectorstores/qdrant.py, sparse/ (BM25/FTS), rerankers/, llms/, registry.py
  rag/            orchestrator
  observability/  trace persistence
  evaluation/     metrics, judges, dataset, runner, CLI
alembic/          migrations
config/rag.yaml   pipeline defaults
evals/            golden dataset + variants
samples/          sample corpus (generated by scripts/sample_docs.py)
scripts/          demo.py, sample_docs.py
tests/            unit/ and integration/
```

## Design notes and limits (V1)

- **PostgreSQL is the source of truth**. Qdrant `is_active` flags are an optimisation, and
  every hit is re-hydrated from PostgreSQL with tenant, KB and active filters. A failure between
  the two stores can never surface stale or foreign chunks.
- **BM25 statistics are per knowledge base**, so one tenant's corpus never affects another's scores.
  For very common terms the candidate set is scored in full. Very large KBs (millions of chunks
  per KB) may need a candidate cap or a dedicated sparse engine. `SparseSearch` is the extension point.
- **Permissions** are stored on chunks and payloads (default `["public"]`) and are enforced only
  when the caller passes a `permissions` filter. Products are expected to pass the end user's
  principals.
- Scanned PDFs (no text layer) need OCR, which is not included.
- Raw uploads are stored on a local volume (`BlobStore`). Swap in an S3/GCS implementation for
  multi-node deployments.
