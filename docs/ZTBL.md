# ZTBL Islamic Banking Assistant (Groq)

A hybrid RAG knowledge base over ZTBL's Islamic Banking FAQs, products and Shariah fatwa,
answered by a Groq-hosted LLM. It runs on this platform with configuration only; no
ZTBL-specific code.

## Corpus (`corpora/ztbl/`)

| File | Content | Why this file |
|---|---|---|
| `ZTBL_FAQs_Islamic_Banking.docx` | 52 FAQs (Parts 1–3), account and deposit products, financing products | This is the *formatted* source. Every question is a Word heading, so each answer is indexed and cited under its question (`Part 2 > Question No. 12) What is Murabaha? ...`), and every product limit sits under its product name. The plain-text export loses the questions. |
| `ZTBL_Shariah_Fatwa.md` | Part 5 fatwa text and Shariah Board signatories, copied verbatim from the text export | The Word file has no fatwa section. |

The load produces 148 FAQ/product chunks and 2 fatwa chunks.

## Run it

```bash
# 1. .env: Groq as the LLM (key from https://console.groq.com/keys); remove COMPOSE_PROFILES=local-llm
LLM__PROVIDER=groq
LLM__MODEL=openai/gpt-oss-120b
LLM__API_KEY=gsk_...

# 2. Start the stack
docker compose up -d --build

# 3. Load the corpus (idempotent; tenant key saved to .secrets/ztbl.json, which is git-ignored)
python scripts/load_corpus.py --dir corpora/ztbl --tenant-slug ztbl --tenant-name "Zarai Taraqiati Bank Limited" \
    --kb islamic-banking --api http://localhost:8000 --admin-key <SECURITY__ADMIN_API_KEY> --source ztbl-website

# 4. Ask
python scripts/ask.py --tenant-slug ztbl --kb islamic-banking --api http://localhost:8000 \
    "What is the financing limit for a rice transplanter?"
```

To update the content, edit or replace a file in `corpora/ztbl/` and re-run step 3. Unchanged files
are skipped. A changed file becomes a new version, and the old version stops being served once the
new one is ingested.

Product apps call `POST /v1/rag/query` with the tenant key (see [REUSE_GUIDE.md](REUSE_GUIDE.md)).
Use `filters` to narrow results, e.g. `{"document_type": "md"}` for fatwa-only answers.

## Groq notes

- `openai/gpt-oss-120b` answers in under 1 s for typical queries (about 1k prompt tokens).
- gpt-oss cites with fullwidth markers (`【7】`). The platform normalises them to `[7]` and validates
  them like any other citation.
- Rate-limit responses (HTTP 429) are retried automatically, honouring `Retry-After`.
- Other chat models on the account can be swapped in through `LLM__MODEL`. List them with
  `GET https://api.groq.com/openai/v1/models`.

## Evaluation

`evals/datasets/ztbl.yaml` holds 20 golden questions covering accounts, financing limits, Islamic
finance concepts and the fatwa. Run them with:

```bash
python -m app.evaluation --dataset evals/datasets/ztbl.yaml --variants evals/ztbl_variants.yaml
```

Each variant re-embeds the corpus in a temporary tenant, which takes several minutes per variant on CPU.

### Measured results (live stack, Groq `openai/gpt-oss-120b`, CPU-only machine)

| Setup | hit@1 | hit@3 | MRR@8 | Rerank time per query |
|---|---|---|---|---|
| RRF only (no reranker) | 0.60 | 0.85 | 0.747 | – |
| MiniLM-L12 reranker, top 20 | 0.80 | 1.00 | 0.900 | 6 s |
| jina-reranker-v2, top 20, capped at 800 chars | 0.80 | 1.00 | 0.892 | 14 s |
| **jina-reranker-v2, top 20 (configured)** | **0.90** | **1.00** | **0.942** | **20 s** |
| jina-reranker-v2, top 40 | 0.90 | 1.00 | 0.942 | 51 s |

End to end with the configured setup:

- Correctness: 20/20 answers correct and citing the right chunk; 0 fabricated citations.
- Latency: median 24.5 s per query, of which reranking is 21 s and the Groq LLM 1.6 s.

The cross-encoder is what makes many ZTBL questions answerable. Product limits share identical
wording ("Maximum financing limit shall be PKR … per customer") and differ only in the section
path. The reranker reads that path; RRF alone cannot tell the products apart. On CPU it is also the
latency bottleneck. A GPU or a hosted reranker (`RERANKER__PROVIDER=cohere`) brings total latency
down to a few seconds, and `Xenova/ms-marco-MiniLM-L-12-v2` is the fastest local option.
