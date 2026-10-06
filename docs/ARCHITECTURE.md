# How the Hybrid RAG Platform Is Built

This document explains the internals: the request and ingestion flows, the data model, every
pipeline stage, and the reasons behind the main design choices. To *use* the platform from a
product, read [REUSE_GUIDE.md](REUSE_GUIDE.md) instead.

---

## 1. Shape of the system

The platform is a **modular monolith**: one Python 3.11 codebase, deployed as one API process
(plus an optional worker process). It depends on two stateful services.

| Component | Role |
|---|---|
| **FastAPI app** (`app/main.py`) | HTTP API, auth, request validation, embedded ingestion worker |
| **PostgreSQL 17** | Source of truth: tenants, KBs, documents, versions, chunk text + metadata, BM25 index (`tsvector`), job queue, traces, embedding configs |
| **Qdrant 1.19** | The only vector database: dense embeddings + filterable payload |
| **Model providers** | Embeddings, reranker, LLM. They run locally (fastembed ONNX, Ollama) or are called over HTTP (OpenAI, Anthropic, Cohere) |

There is no Redis, Kafka or Elasticsearch. The job queue lives in PostgreSQL and BM25 runs in
PostgreSQL.

### Code layout and dependency direction

```text
api/            HTTP only: auth, (de)serialization, error mapping
  │
rag/            RAGOrchestrator: composes the query pipeline
ingestion/      DocumentService (upload/versioning), IngestionPipeline, IngestionWorker
  │
retrieval/      DenseRetriever, SparseRetriever, HybridRetriever, fusion/, reranking/, filters, repository
generation/     ContextBuilder, prompts, citation validation
  │
providers/      Interfaces (base.py) + concrete implementations + registry.py (the only factory)
  │
models/ db/ core/   ORM entities, sessions, config, errors, tokens
```

The upper layers depend only on the **interfaces** in `providers/*/base.py`. The concrete classes
are selected in one place, `app/providers/registry.py`, and wired together in `app/container.py`.
For example, the orchestrator never imports `qdrant_client` or `fastembed`.

### The eight extension interfaces

| Interface | File | Implementations shipped |
|---|---|---|
| `EmbeddingProvider` | `providers/embeddings/base.py` | `FastEmbedProvider`, `OpenAIEmbeddingProvider`, `HashingEmbeddingProvider` |
| `VectorStore` | `providers/vectorstores/base.py` | `QdrantVectorStore` |
| `SparseSearch` | `providers/sparse/base.py` | `PostgresBM25Search`, `PostgresFTSSearch` |
| `FusionStrategy` | `retrieval/fusion/base.py` | `ReciprocalRankFusion`, `RelativeScoreFusion` |
| `Reranker` | `providers/rerankers/base.py` | `FastEmbedCrossEncoderReranker`, `CohereReranker`, `PassthroughReranker` |
| `LLMProvider` | `providers/llms/base.py` | `OpenAICompatibleLLM`, `AnthropicLLM`, `ExtractiveLLM` |
| `Chunker` | `ingestion/chunking/base.py` | `RecursiveStructureChunker` |
| `DocumentLoader` | `ingestion/loaders/base.py` | `PDFLoader`, `DocxLoader`, `MarkdownLoader`, `HTMLLoader`, `TextLoader` |

---

## 2. Data model

```text
Tenant ──< ApiKey
  └──< KnowledgeBase ──> EmbeddingConfig ──(1:1)── Qdrant collection
         └──< Document                     (external_id unique per KB)
                └──< DocumentVersion       (checksum unique per document; one is_active)
                       └──< Chunk          (text, offsets, page, section, tsvector; one Qdrant point)
IngestionJob   (queue row per version to process)
RetrievalTrace (one row per RAG request)
```

These rules hold throughout:

- **Every row below `Tenant` carries `tenant_id`**, and every row below `KnowledgeBase` carries
  `knowledge_base_id`. Isolation filters therefore never depend on joins.
- `chunks` **denormalises** `document_type`, `source`, `permissions` and `metadata` from the
  document. Sparse search can then apply exactly the same filters as Qdrant does on its payload.
- **Embeddings are never stored in PostgreSQL.** The link to Qdrant is the chunk id, which is
  also the Qdrant point id.
- Chunk ids are deterministic: `uuid5(document_version_id, chunk_index)`. Re-running ingestion
  for a version overwrites the same points instead of duplicating them.

Schema migrations live in `alembic/versions/`. `docker compose` runs them via the `migrate`
service before the API starts.

---

## 3. Ingestion flow

```text
POST /v1/documents
   │  DocumentService.upload (app/ingestion/service.py)
   │   - detect type (extension → MIME → magic bytes)
   │   - SHA-256 checksum
   │   - same (KB, external_id) + same checksum → "unchanged" (no work)
   │   - otherwise create Document / new DocumentVersion (status=pending), store raw bytes in BlobStore
   │   - insert IngestionJob(status=queued)
   ▼
IngestionWorker (app/ingestion/worker.py)
   │   claims a job:  UPDATE ... WHERE id = (SELECT ... FOR UPDATE SKIP LOCKED LIMIT 1)
   ▼
IngestionPipeline.run (app/ingestion/pipeline.py)
   1. Load      raw bytes from the BlobStore
   2. Parse     DocumentLoader → ParsedDocument(segments=[Segment(text, page, section)])
   3. Clean     NFKC, control chars, whitespace, PDF hyphenation (ingestion/cleaning.py)
   4. Chunk     RecursiveStructureChunker → ChunkDraft(text, char offsets, page range, section, tokens)
   5. Embed     EmbeddingProvider.embed_documents("title\nsection\n\nchunk text"), dimension validated
   6. Qdrant    delete leftovers of this version, upsert points (is_active=false)
   7. Postgres  delete leftovers, bulk insert chunks (is_active=false), compute tsvector in SQL
   8. Activate  one transaction: new version + its chunks active, previous version inactive
                (skipped if a *newer* version is already active), then flip Qdrant is_active flags
```

On failure, the job is retried with exponential backoff (`5·2^attempt` s, capped at 300 s). It
becomes `failed` after `WORKER__MAX_ATTEMPTS`, and the error is stored on both the job and the
version. Jobs left `running` by a crashed worker are re-queued after `WORKER__LEASE_TIMEOUT_S`.

### Parsing (loaders)

Every loader produces **segments**, and every segment carries the structure needed for citations.

| Format | Segment unit | Structure captured |
|---|---|---|
| PDF (`pypdf`) | one per page | 1-based `page`. Scanned PDFs without a text layer are rejected (no OCR in V1). |
| DOCX (`python-docx`) | paragraphs grouped under headings | `Heading N` / `Title` styles build the section path; tables become `cell \| cell` rows; list items get a `- ` prefix |
| Markdown (`markdown-it`) | text between headings | Heading hierarchy becomes the section path (`A > B > C`); code fences are ignored for headings |
| HTML (`BeautifulSoup`/`lxml`) | block elements under h1–h6 | `script`, `style`, `nav`, `header`, `footer`, `aside` and `form` are stripped; `<main>`/`<article>` is preferred |
| TXT | whole file | Encoding fallback utf-8 → utf-16 (BOM) → cp1252 |

### Chunking

`RecursiveStructureChunker` (`ingestion/chunking/recursive.py`) works in three steps:

1. **Group** consecutive segments that share a section. A chunk never crosses a section
   boundary, so every chunk has exactly one `section`. Chunks may span pages, and the page
   range (`page`, `page_end`) is computed from character offsets.
2. **Split** each group recursively on increasingly fine separators (`\n\n`, `\n`, `. `, `? `,
   `! `, `; `, `, `, space, character) until each piece is at most `chunk_size`. Separators stay
   attached to their piece, so pieces are exact substrings and the offsets are exact.
3. **Merge** pieces greedily up to `chunk_size`, carrying the trailing pieces as
   `chunk_overlap` into the next chunk.

Sizes are counted in tokens (tiktoken `cl100k_base`, with an offline approximation fallback) or in
characters. The platform default is 800/120 tokens. Keep `chunk_size` within the embedding
model's input limit; the API logs a warning when it is exceeded.

---

## 4. Query flow

`RAGOrchestrator.query` (`app/rag/orchestrator.py`):

```text
  auth: X-API-Key → tenant_id                       (never from the request body)
  get KB (404 if missing OR another tenant's)       → its EmbeddingConfig + fts_language
  parse_filters(filters)                            (whitelisted fields; tenant/KB filters → 422)
  scope = RetrievalScope(tenant_id, kb_id, active_only=True)
         │
         ├─ DenseRetriever:  embed_query → Qdrant query_points(filter = scope + filters), top 50
         │                                                          } asyncio.gather (concurrent)
         ├─ SparseRetriever: BM25 in PostgreSQL (scope + filters), top 50
         │                    (if one side fails: continue with the other, trace marked degraded)
         ▼
  FusionStrategy (RRF k=60)  → top 40 candidates with dense_rank, sparse_rank, rrf_score
         ▼
  ChunkRepository.get_many   hydrate text/metadata from PostgreSQL, re-filtered by tenant, KB, is_active
         ▼
  RerankingStage             cross-encoder scores (query, "section\nchunk") pairs → top 8 by reranker_score
         ▼
  ContextBuilder             dedup → token budget → ordering → "[n] Document: … | Section: … | Page: …"
         ▼
  LLMProvider.generate       system prompt: use only the context, cite [n], never invent citations
         ▼
  extract_citations          keep valid [n]; strip fabricated ids (recorded in trace); build Citation objects
         ▼
  TraceRecorder.save         persisted even when the request fails
```

### Dense retrieval (Qdrant)

**Collection strategy.** There is one collection per embedding configuration, named for example
`rag_chunks__fastembed__mixedbread_ai_mxbai_embed_large_v1__1024d__cosine__v1`. All tenants share
it. There is never one collection per document or per tenant: that design does not scale to many
tenants. Each knowledge base is pinned to one `EmbeddingConfig`. Changing the default model
creates a new config and a new collection, so vector spaces are never mixed.

**Payload indexes.** `tenant_id` is a keyword index created with `is_tenant=True`, which makes
Qdrant co-locate each tenant's vectors. The other indexed fields are `knowledge_base_id`,
`document_id`, `document_version_id`, `chunk_id`, `document_type`, `source`, `section`,
`permissions`, `page` and `is_active`. Custom metadata is stored under `metadata.*` and can be
filtered without an index.

**Validation.**

- Each vector's length is checked against the config dimension twice: when the provider returns
  it and again just before upsert. Non-finite values are rejected.
- An existing collection with a different size or distance metric raises
  `EmbeddingConfigMismatchError`.

**Defence in depth.** Even after Qdrant filtering, any point whose payload tenant or KB does not
match the scope is dropped and logged.

### Sparse retrieval (BM25 in PostgreSQL)

`chunks.search_vector` is a weighted `tsvector`: title and section get weight A, body text gets
weight B. It has a GIN index. `PostgresBM25Search` computes **Okapi BM25** (Lucene IDF) in a
single SQL statement:

```text
score(d) = Σ_t  ln(1 + (N − df_t + 0.5)/(df_t + 0.5)) · tf_{t,d}·(k1+1) / (tf_{t,d} + k1·(1 − b + b·|d|/avgdl))
```

- **Query terms:** the normalised lexemes of `to_tsvector(kb.fts_language, query)`, combined with
  **OR** semantics. Natural-language questions rarely contain every word of the answer.
- **tf:** the number of positions of the lexeme in the chunk's `tsvector`.
- **|d|:** the chunk's token count.
- **N, avgdl, df:** computed over the **knowledge base's active chunks only**. One tenant's corpus
  never influences another tenant's scores.
- **Filters:** user filters are compiled by `providers/sparse/sql_filters.py` into a
  parameterised SQL fragment. Field names are whitelisted and every value is a bound parameter.

`PostgresFTSSearch` (`ts_rank_cd`) is available as an alternative
(`RETRIEVAL__SPARSE_PROVIDER=postgres_fts`).

### Reciprocal Rank Fusion

```text
RRF(d) = Σ_i  w_i / (k + rank_i(d))        k = 60, w_i = 1, rank is 1-based, absent lists contribute 0
```

RRF is **rank-based**: cosine similarities and BM25 scores live on different scales and are never
compared. This is not related to embedding dimensionality. Ties are broken deterministically by
best individual rank, then by chunk id. Each candidate keeps `ranks{dense,sparse}`,
`scores{dense,sparse}` and `fused_score`, which is the RRF score.

### Reranking

A cross-encoder reads the query and each passage *together*, which is far more precise than the
bi-encoder similarity used for retrieval. Using it only on the 40 fused candidates keeps its cost
bounded. The default model is `jinaai/jina-reranker-v2-base-multilingual` (local ONNX, 1K-token
context). The reranker input includes the section path. Both `rrf_score` and `reranker_score` are
kept and returned.

### Context building and citations

- **Deduplication.** A candidate is dropped if its checksum exactly matches a selected chunk, or
  if its 5-gram shingle Jaccard similarity with a selected chunk is at least 0.85.
- **Budget.** Candidates are added greedily in rerank order while they fit in
  `CONTEXT__MAX_CONTEXT_TOKENS`. Oversized chunks are skipped, never truncated.
- **Ordering.** The default, `grouped`, puts documents in order of their best rank and chunks of
  one document in reading order. `relevance` keeps pure rerank order.
- **Citation ids.** `[1]..[n]` are assigned after ordering. Each block has a header naming the
  document, section and page(s).
- **Answer validation.** The answer is scanned for `[n]` / `[n, m]`. Ids that exist in the context
  become `Citation` objects carrying `document_id, document_name, document_version_id, chunk_id,
  page, page_end, section, source, snippet, rrf_score, reranker_score`. Any other id is removed
  from the answer text and listed in `trace.llm.invalid_citation_ids`.
- **Empty context.** If there is nothing to cite, the LLM is not called and the answer is
  `"I don't know based on the provided documents."`.

### Traces

Every request writes one `retrieval_traces` row containing:

- query, filters and the effective config (models, all k values, collection);
- the dense and sparse result lists with ranks and scores;
- the fused list (RRF score and component ranks) and the reranked list;
- selected chunks with a `why` block, and dropped chunks with a reason (`duplicate_exact`,
  `near_duplicate`, `token_budget`, `inactive_or_missing`);
- context token count, LLM model, usage, latency and raw answer;
- citations, per-stage timings, and errors (including degraded retrieval).

This answers "why did (or didn't) this chunk end up in the answer?" without re-running anything.

---

## 5. Multi-tenancy and security model

| Layer | Guarantee |
|---|---|
| Authentication | API keys are random and stored only as SHA-256 hashes. The tenant comes from the key. Admin endpoints use a separate `X-Admin-Key`, compared in constant time. The default admin key is refused in staging and production. |
| Request validation | Filters on `tenant_id`, `knowledge_base_id` and `is_active` are rejected with 422. Unknown filter fields are rejected. |
| Resource lookup | Another tenant's KB, document, job or trace returns **404**, the same response as for a missing one, so existence is not leaked. |
| Qdrant | The scope filter (`tenant_id`, `knowledge_base_id`, `is_active`) is always added by the store, and results are re-checked. |
| PostgreSQL | Every sparse query and every hydration query includes the tenant and KB predicates. |
| Statistics | BM25 corpus statistics are scoped per KB. |

Integration tests (`tests/integration/test_tenant_isolation.py`) verify each of these layers.

---

## 6. Configuration and startup

`app/core/config.py` defines typed settings (Pydantic v2) in these sections: `database`,
`qdrant`, `embedding`, `chunking`, `retrieval`, `reranker`, `llm`, `context`, `storage`,
`worker`, `security` and `tokenizer`.

Sources, in priority order:

1. environment variables (nested with `__`);
2. `.env`;
3. `config/rag.yaml`;
4. code defaults.

At startup, `Container.create` builds every provider from settings. It then registers the default
`EmbeddingConfig` in PostgreSQL, ensures that config's Qdrant collection exists with its payload
indexes, and warms up the local models. Finally it starts the embedded ingestion worker unless
`WORKER__EMBEDDED=false`.

---

## 7. Design decisions

| Decision | Why |
|---|---|
| PostgreSQL is the source of truth; Qdrant holds only vectors and filter payload | One transactional place for activation and versioning. Hydration re-checks `is_active`, so a partial failure between the two stores can never surface stale chunks. |
| BM25 in PostgreSQL instead of Elasticsearch | No extra stateful service; filters and isolation share the same SQL. `SparseSearch` is the escape hatch for very large KBs. |
| One Qdrant collection per embedding config | Different models and dimensions never mix. Re-indexing to a new model is "new KB/config, re-ingest". Avoids collection explosion. |
| RRF as the default fusion | Robust to incomparable score scales; needs no per-corpus tuning. |
| Cross-encoder only on the top 40 | Reranking quality where it matters, at bounded cost. |
| Job queue in PostgreSQL | Durable, transactional with the data it processes, no broker to operate. `SKIP LOCKED` allows any number of workers. |
| Deterministic chunk ids | Idempotent retries and re-ingestion; no duplicate vectors. |
| Citation validation after generation | The model cannot invent sources: fabricated ids are removed, never returned. |

### Known limits (V1)

- **Sparse scoring cost:** BM25 scores every chunk that matches any query term. KBs with millions
  of chunks may need a candidate cap or a dedicated sparse engine.
- **No OCR:** scanned PDFs without a text layer are rejected.
- **Local blob storage:** raw uploads are stored on a local volume. Multi-node deployments need an
  object-storage `BlobStore`.
- **Permissions are opt-in:** they are enforced only when the caller sends a `permissions` filter.
- **CPU latency:** on CPU-only machines, LLM generation dominates latency (minutes with a local 3B
  model). Use a GPU or a hosted LLM in production.
