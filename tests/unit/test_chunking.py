import itertools

import pytest

from app.core.tokens import ApproximateTokenCounter, get_token_counter
from app.ingestion.chunking.recursive import RecursiveStructureChunker
from app.ingestion.loaders.base import ParsedDocument, Segment

TC = get_token_counter()
SENT = "Refund requests for annual plans are processed by the billing team within ten days."


def para(n: int, prefix: str = "") -> str:
    return " ".join(f"{prefix}{i}: {SENT}" for i in range(n))


def chunker(size=100, overlap=20, unit="tokens", counter=TC):
    return RecursiveStructureChunker(counter, chunk_size=size, chunk_overlap=overlap, unit=unit)


def test_offsets_are_exact_substrings():
    doc = ParsedDocument(
        segments=[Segment(para(40), page=1, section="A"), Segment(para(30), page=2, section="A")]
    )
    full = doc.text
    chunks = chunker().chunk(doc)
    assert len(chunks) > 3
    for c in chunks:
        assert full[c.char_start : c.char_end] == c.text
        assert c.index == chunks.index(c)


def test_chunk_size_respected():
    doc = ParsedDocument(segments=[Segment(para(80))])
    for c in chunker(size=100, overlap=20).chunk(doc):
        assert c.token_count <= 105  # BPE merges across piece boundaries can only shrink counts; small slack


def test_overlap_between_consecutive_chunks():
    doc = ParsedDocument(segments=[Segment(para(60))])
    chunks = chunker(size=100, overlap=30).chunk(doc)
    for prev, nxt in itertools.pairwise(chunks):
        assert nxt.char_start < prev.char_end, "consecutive chunks must overlap"
        overlap_text = doc.text[nxt.char_start : prev.char_end]
        assert 0 < TC.count(overlap_text) <= 30 + 5


def test_no_overlap_when_zero():
    doc = ParsedDocument(segments=[Segment(para(60))])
    chunks = chunker(size=100, overlap=0).chunk(doc)
    for prev, nxt in itertools.pairwise(chunks):
        assert nxt.char_start >= prev.char_end


def test_chunks_never_cross_sections():
    doc = ParsedDocument(
        segments=[Segment(para(15, "x"), section="Intro"), Segment(para(15, "y"), section="Intro > Refunds")]
    )
    chunks = chunker(size=80, overlap=10).chunk(doc)
    sections = [c.section for c in chunks]
    assert sections == sorted(sections, key=lambda s: s != "Intro")
    for c in chunks:
        if c.section == "Intro":
            assert "y0" not in c.text
        else:
            assert "x14" not in c.text


def test_page_mapping_spans_pages():
    doc = ParsedDocument(segments=[Segment(para(3, "p1-"), page=1), Segment(para(3, "p2-"), page=2)])
    chunks = chunker(size=400, overlap=0).chunk(doc)
    assert len(chunks) == 1
    assert (chunks[0].page, chunks[0].page_end) == (1, 2)

    small = chunker(size=30, overlap=0).chunk(doc)
    assert small[0].page == 1
    assert small[-1].page == 2


def test_character_unit():
    doc = ParsedDocument(segments=[Segment(para(20))])
    for c in chunker(size=300, overlap=50, unit="characters").chunk(doc):
        assert len(c.text) <= 300


def test_hard_split_of_unbreakable_text():
    doc = ParsedDocument(segments=[Segment("x" * 5000)])
    chunks = chunker(size=50, overlap=0, unit="characters").chunk(doc)
    assert all(len(c.text) <= 50 for c in chunks)
    assert "".join(c.text for c in chunks) == "x" * 5000


def test_tiny_noise_chunks_dropped():
    doc = ParsedDocument(segments=[Segment("12", page=1), Segment(SENT, page=2, section="S")])
    chunks = chunker().chunk(doc)
    assert [c.text for c in chunks] == [SENT]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunker(size=50, overlap=50)


def test_approximate_counter_works_offline():
    c = chunker(size=50, overlap=10, counter=ApproximateTokenCounter())
    assert len(c.chunk(ParsedDocument(segments=[Segment(para(20))]))) > 1
