"""Answer-accuracy evaluation: ask the live RAG each question and grade it against a reference answer.

    python scripts/eval_answers.py --qa "ztbl -50 qs & ans.txt" --tenant-slug ztbl --kb islamic-banking \
        --api http://localhost:8000

Input: a YAML list of {id, question, reference_answer}, or a text file in the format
    1. Question text?
     Answer: reference answer (may span several lines / list items)
Grading: an LLM judge (Groq, OpenAI-compatible; key/model from .env unless overridden) compares
the RAG answer with the reference and returns correct / partially_correct / incorrect with the
missing facts and any contradictions. Results go to reports/answers-<timestamp>.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUESTION_RE = re.compile(r"^\s*(\d+)\.\s+(.+\?)\s*$")
SCORES = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}

JUDGE_SYSTEM = """You grade answers produced by a retrieval-augmented assistant for a bank.
Compare the ASSISTANT ANSWER with the REFERENCE ANSWER for the QUESTION.

Verdicts:
- "correct": contains every key fact of the reference (numbers, names, conditions, list items) and nothing that contradicts it. Extra correct detail and different wording are fine. Citation markers like [3] are irrelevant.
- "partially_correct": right in substance but misses at least one key fact of the reference, or is vague where the reference is specific. No contradictions.
- "incorrect": contradicts the reference, answers a different question, or declines to answer ("I don't know").

Reply with JSON only:
{"verdict": "correct|partially_correct|incorrect", "missing": ["key facts from the reference that are absent"], "contradictions": ["statements that conflict with the reference"], "reason": "one sentence"}"""


def parse_qa_text(text: str) -> list[dict]:
    """Parse '1. Question?' followed by an 'Answer:' block (lists and line breaks allowed)."""
    items: list[dict] = []
    current: dict | None = None
    expected = 1
    for raw in text.splitlines():
        line = raw.strip()
        m = QUESTION_RE.match(raw)
        if m and int(m.group(1)) == expected:
            current = {"id": f"q{expected:02d}", "question": m.group(2).strip(), "answer_lines": []}
            items.append(current)
            expected += 1
            continue
        if current is None or not line or set(line) <= {"_", "-", "—"}:
            continue
        if line.lower().startswith("answer:"):
            line = line[len("answer:") :].strip()
        current["answer_lines"].append(line)
    out = []
    for it in items:
        ref = " ".join(it.pop("answer_lines")).strip()
        # Drop a trailing section header that belongs to the next block (no "?" and short, Title Case).
        out.append({**it, "reference_answer": ref})
    return out


def load_items(path: Path) -> list[dict]:
    if path.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data["items"] if isinstance(data, dict) else data
    return parse_qa_text(path.read_text(encoding="utf-8"))


def strip_section_headers(items: list[dict], text: str) -> None:
    """Remove section-title lines (e.g. 'Mudarabah & Musharakah') that the text parser glued onto answers."""
    headers = {
        ln.strip()
        for ln in text.splitlines()
        if ln.strip()
        and not QUESTION_RE.match(ln)
        and not ln.strip().lower().startswith("answer:")
        and len(ln.strip()) < 70
        and not ln.strip().endswith((".", ":", "?"))
        and not re.match(r"^\d+\.", ln.strip())
        and not set(ln.strip()) <= {"_", "-", "—"}
    }
    for it in items:
        for h in headers:
            if it["reference_answer"].endswith(" " + h):
                it["reference_answer"] = it["reference_answer"][: -len(h) - 1].rstrip()


class Judge:
    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        self.model = model
        self.http = httpx.Client(
            base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=120
        )

    def grade(self, question: str, reference: str, answer: str) -> dict:
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": f"QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{reference}\n\nASSISTANT ANSWER:\n{answer}",
                },
            ],
        }
        for attempt in range(6):
            try:
                r = self.http.post("/chat/completions", json=body)
            except httpx.TransportError:  # e.g. TLS interception by endpoint security software
                time.sleep(min(30.0, 2.0**attempt))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(30.0, float(r.headers.get("retry-after", 2**attempt))))
                continue
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"] or "{}"
            m = re.search(r"\{.*\}", content, re.S)
            result = json.loads(m.group(0) if m else "{}")
            if result.get("verdict") in SCORES:
                return result
        return {"verdict": "incorrect", "missing": [], "contradictions": [], "reason": "judge failed"}


def main() -> None:
    from app.core.config import Settings

    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True, help="YAML or text file with questions and reference answers")
    ap.add_argument("--tenant-slug", required=True)
    ap.add_argument("--kb", required=True)
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--judge-model", help="defaults to LLM__MODEL")
    ap.add_argument("--judge-base-url", default="https://api.groq.com/openai/v1")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default=str(ROOT / "reports"))
    args = ap.parse_args()

    qa_path = Path(args.qa)
    items = load_items(qa_path)
    if qa_path.suffix.lower() not in (".yaml", ".yml"):
        strip_section_headers(items, qa_path.read_text(encoding="utf-8"))
    items = items[: args.limit] if args.limit else items
    print(f"{len(items)} questions loaded from {qa_path.name}")

    settings = Settings()
    judge = Judge(
        args.judge_model or settings.llm.model,
        settings.llm.api_key.get_secret_value() if settings.llm.api_key else "",
        args.judge_base_url,
    )
    key = json.loads((ROOT / ".secrets" / f"{args.tenant_slug}.json").read_text())["api_key"]
    rag = httpx.Client(base_url=args.api, headers={"X-API-Key": key}, timeout=900)
    kb = next((k for k in rag.get("/v1/knowledge-bases").json() if k["name"] == args.kb), None)
    if kb is None:
        sys.exit(f"knowledge base '{args.kb}' not found")

    results = []
    for it in items:
        t = time.time()
        r = rag.post(
            "/v1/rag/query",
            json={"knowledge_base_id": kb["id"], "query": it["question"], "top_k": args.top_k},
        )
        r.raise_for_status()
        res = r.json()
        latency = time.time() - t
        trace = rag.get(f"/v1/traces/{res['trace_id']}").json()
        timings = trace.get("timings_ms", {})
        grade = judge.grade(it["question"], it["reference_answer"], res["answer"] or "")
        row = {
            **it,
            "rag_answer": res["answer"],
            "verdict": grade["verdict"],
            "score": SCORES[grade["verdict"]],
            "missing": grade.get("missing", []),
            "contradictions": grade.get("contradictions", []),
            "reason": grade.get("reason", ""),
            "citations": [
                {"document": c["document_name"], "section": c["section"], "chunk_id": c["chunk_id"]}
                for c in res["citations"]
            ],
            "latency_s": round(latency, 1),
            "search_s": round(timings.get("retrieval_wall_ms", 0) / 1000, 2),
            "rerank_s": round(timings.get("rerank_ms", 0) / 1000, 1),
            "llm_s": round(timings.get("generate_ms", 0) / 1000, 1),
            "llm_tokens": (trace.get("llm") or {}).get("usage", {}).get("total_tokens"),
            "trace_id": res["trace_id"],
        }
        results.append(row)
        print(
            f"{row['id']} {row['verdict']:<18} total {latency:5.1f}s (search {row['search_s']:.2f}s, "
            f"rerank {row['rerank_s']:.1f}s, llm {row['llm_s']:.1f}s)  {it['question'][:60]}",
            flush=True,
        )

    n = len(results)
    counts = {v: sum(r["verdict"] == v for r in results) for v in SCORES}
    accuracy = sum(r["score"] for r in results) / n
    strict = counts["correct"] / n
    with_cit = sum(1 for r in results if r["citations"]) / n
    summary = {
        "questions": n,
        "accuracy_weighted": round(accuracy, 3),
        "accuracy_strict": round(strict, 3),
        "counts": counts,
        "answers_with_citations": round(with_cit, 3),
        "median_latency_s": statistics.median(r["latency_s"] for r in results),
        "median_search_s": statistics.median(r["search_s"] for r in results),
        "median_rerank_s": statistics.median(r["rerank_s"] for r in results),
        "median_llm_s": statistics.median(r["llm_s"] for r in results),
        "judge_model": judge.model,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (out / f"answers-{stamp}.json").write_text(
        json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        f"# Answer accuracy: {qa_path.name} ({stamp})",
        "",
        f"- Questions: {n}",
        f"- Accuracy (correct = 1, partial = 0.5): **{accuracy:.1%}**",
        f"- Strict accuracy (fully correct only): **{strict:.1%}**",
        f"- Correct / partially correct / incorrect: {counts['correct']} / {counts['partially_correct']} / {counts['incorrect']}",
        f"- Answers with citations: {with_cit:.0%}",
        f"- Median latency: {summary['median_latency_s']:.1f} s (search {summary['median_search_s']:.2f} s, "
        f"rerank {summary['median_rerank_s']:.1f} s, LLM {summary['median_llm_s']:.1f} s)",
        f"- Judge: {judge.model}",
        "",
        "| # | Question | Verdict | Total (s) | Search (s) | Rerank (s) | LLM (s) | Missing / contradictions |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        issues = "; ".join(r["missing"] + [f"CONTRADICTS: {c}" for c in r["contradictions"]]) or "-"
        lines.append(
            f"| {r['id']} | {r['question']} | {r['verdict']} | {r['latency_s']:.1f} | {r['search_s']:.2f} | "
            f"{r['rerank_s']:.1f} | {r['llm_s']:.1f} | {issues.replace('|', '/')} |"
        )
    md = out / f"answers-{stamp}.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))
    print(f"report: {md}")


if __name__ == "__main__":
    main()
