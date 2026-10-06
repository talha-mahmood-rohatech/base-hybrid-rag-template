import uuid

import pytest

from app.core.tokens import get_token_counter
from app.core.utils import sha256_hex
from app.generation.citations import extract_citations
from app.generation.context import ContextBuilder
from app.generation.prompts import NO_ANSWER, build_messages
from app.providers.llms.extractive import ExtractiveLLM
from app.retrieval.types import Candidate, ChunkView

TC = get_token_counter()


def cand(text, *, doc=None, idx=0, rank=1, name="doc.md", page=None, section=None, rerank=None):
    doc = doc or uuid.uuid4()
    cid = uuid.uuid4()
    view = ChunkView(
        chunk_id=cid,
        tenant_id=uuid.uuid4(),
        knowledge_base_id=uuid.uuid4(),
        document_id=doc,
        document_version_id=uuid.uuid4(),
        document_name=name,
        document_type="md",
        source=None,
        chunk_index=idx,
        text=text,
        token_count=TC.count(text),
        page=page,
        page_end=page,
        section=section,
        checksum=sha256_hex(text),
    )
    return Candidate(
        chunk_id=cid,
        document_id=doc,
        fused_score=0.01,
        fused_rank=rank,
        final_rank=rank,
        reranker_score=rerank,
        chunk=view,
    )


A = "Monthly subscriptions can be refunded in full within 30 days of the initial purchase date."
B = "Usage-based charges such as compute hours and egress bandwidth are never refundable."
C = "Refunds are issued to the original payment method within 5 to 10 business days."


def test_exact_and_near_duplicates_removed():
    near = A.replace("date.", "date!")
    built = ContextBuilder(TC).build([cand(A), cand(A), cand(near), cand(B)])
    assert [b.text for b in built.blocks] == [A, B]
    reasons = sorted(d["reason"] for d in built.dropped)
    assert reasons == ["duplicate_exact", "near_duplicate"]


def test_token_budget_skips_oversize_and_keeps_fitting():
    big = " ".join([B] * 30)
    builder = ContextBuilder(TC, max_tokens=TC.count(A) + TC.count(C) + 60)
    built = builder.build([cand(A), cand(big), cand(C)])
    assert [b.text for b in built.blocks] == [A, C]
    assert built.dropped[0]["reason"] == "token_budget"


def test_grouped_ordering_keeps_document_reading_order():
    d1, d2 = uuid.uuid4(), uuid.uuid4()
    c1 = cand(A, doc=d1, idx=5, rank=1)
    c2 = cand(B, doc=d2, idx=0, rank=2)
    c3 = cand(C, doc=d1, idx=2, rank=3)
    built = ContextBuilder(TC, ordering="grouped").build([c1, c2, c3])
    assert [b.candidate for b in built.blocks] == [c3, c1, c2]
    assert [b.citation_id for b in built.blocks] == [1, 2, 3]
    relevance = ContextBuilder(TC, ordering="relevance").build([c1, c2, c3])
    assert [b.candidate for b in relevance.blocks] == [c1, c2, c3]


def test_block_headers_carry_source_metadata():
    built = ContextBuilder(TC).build([cand(A, name="policy.pdf", page=2, section="Refunds")])
    assert built.text.startswith("[1] Document: policy.pdf | Section: Refunds | Page: 2\n")


def test_citation_extraction_valid_and_fabricated():
    built = ContextBuilder(TC).build([cand(A, name="a.md", section="Elig", page=1), cand(B, name="b.md")])
    res = extract_citations("Refunds take 30 days [1]. Usage is final [2][7]. Also [3, 1].", built)
    assert res.answer == "Refunds take 30 days [1]. Usage is final [2]. Also [1]."
    assert res.invalid_ids == [7, 3]
    assert [c.citation_id for c in res.citations] == [1, 2]
    first = res.citations[0]
    assert first.document_name == "a.md" and first.section == "Elig" and first.page == 1
    assert first.chunk_id == str(built.blocks[0].candidate.chunk_id)
    assert first.document_id == str(built.blocks[0].candidate.document_id)


def test_no_citations_when_answer_has_none():
    built = ContextBuilder(TC).build([cand(A)])
    res = extract_citations(NO_ANSWER, built)
    assert res.citations == [] and res.answer == NO_ANSWER


async def test_extractive_llm_only_quotes_context_with_valid_citations():
    built = ContextBuilder(TC).build([cand(A), cand(B), cand(C)])
    resp = await ExtractiveLLM().generate(
        build_messages("How long do refunds take to be issued?", built.text), max_tokens=200
    )
    res = extract_citations(resp.text, built)
    assert res.invalid_ids == []
    assert res.citations
    for c in res.citations:
        assert c.snippet in (A, B, C)


async def test_extractive_llm_no_answer():
    built = ContextBuilder(TC).build([cand(A)])
    resp = await ExtractiveLLM().generate(
        build_messages("Who won the football match?", built.text), max_tokens=50
    )
    assert resp.text == NO_ANSWER


@pytest.mark.parametrize("ordering", ["grouped", "relevance"])
def test_empty_context(ordering):
    built = ContextBuilder(TC, ordering=ordering).build([])
    assert built.blocks == [] and built.text == ""


def test_fullwidth_citation_markers_are_normalised():
    # gpt-oss style markers, including the "†source" variant and an invented id
    built = ContextBuilder(TC).build([cand(A, name="a.md"), cand(B, name="b.md")])
    res = extract_citations(
        "Refunds take 30 days【1】. Usage is final【2†source】. Also【1, 2】 and【9】.", built
    )
    assert res.answer == "Refunds take 30 days[1]. Usage is final[2]. Also[1][2] and."
    assert [c.citation_id for c in res.citations] == [1, 2]
    assert res.invalid_ids == [9]
