import pytest

from app.core.errors import DocumentParseError, UnsupportedDocumentError
from app.ingestion.loaders.registry import LoaderRegistry
from scripts.sample_docs import (
    EMPLOYEE_HANDBOOK_BLOCKS,
    REFUND_POLICY_MD,
    SECURITY_HANDBOOK_HTML,
    SLA_PAGES,
    make_docx,
    make_pdf,
)

REG = LoaderRegistry()


def test_markdown_sections():
    doc = REG.get("md").load(REFUND_POLICY_MD.encode())
    assert doc.title == "Northwind Cloud Refund Policy"
    sections = [s.section for s in doc.segments]
    assert "Northwind Cloud Refund Policy > Eligibility" in sections
    assert "Northwind Cloud Refund Policy > Exceptions" in sections
    elig = next(s for s in doc.segments if s.section.endswith("Eligibility"))
    assert "within 30 days" in elig.text
    assert "#" not in elig.text


def test_markdown_code_fence_is_not_a_heading():
    md = "# Title\n\n```\n# not a heading\n```\n\n## Real\nbody\n"
    doc = REG.get("md").load(md.encode())
    assert [s.section for s in doc.segments] == ["Title", "Title > Real"]
    assert "# not a heading" in doc.segments[0].text


def test_html_sections_and_noise_removed():
    doc = REG.get("html").load(SECURITY_HANDBOOK_HTML.encode())
    assert doc.title == "Northwind Security Handbook"
    text = doc.text
    assert "analytics" not in text and "Home" not in text and "Copyright" not in text
    incident = next(s for s in doc.segments if s.section and s.section.endswith("Incident Reporting"))
    assert "within 24 hours" in incident.text
    assert "- Lost or stolen laptops" in incident.text


def test_docx_headings_lists_and_tables():
    doc = REG.get("docx").load(make_docx(EMPLOYEE_HANDBOOK_BLOCKS, title="Handbook"))
    assert doc.title == "Handbook"
    expense = next(s for s in doc.segments if s.section.endswith("Expense Reimbursement"))
    assert "Hotel | 180 EUR" in expense.text
    remote = next(s for s in doc.segments if s.section.endswith("Remote Work"))
    assert "- Core collaboration hours" in remote.text


def test_pdf_pages_preserved():
    doc = REG.get("pdf").load(make_pdf(SLA_PAGES, title="SLA"))
    assert doc.page_count == 3
    assert [s.page for s in doc.segments] == [1, 2, 3]
    assert "99.95%" in doc.segments[0].text
    assert "Severity 1" in doc.segments[2].text
    assert doc.title == "SLA"


def test_pdf_garbage_raises():
    with pytest.raises(DocumentParseError):
        REG.get("pdf").load(b"%PDF-1.4 not really a pdf")


def test_txt_and_encoding_fallback():
    doc = REG.get("txt").load("Caf\xe9 menu\nprices".encode("cp1252"))
    assert "Café" in doc.text


@pytest.mark.parametrize(
    "filename,mime,data,expected",
    [
        ("a.PDF", None, b"", "pdf"),
        ("notes.markdown", None, b"", "md"),
        ("page.htm", None, b"", "html"),
        ("x.docx", None, b"", "docx"),
        ("readme", "text/plain", b"", "txt"),
        ("blob", None, b"%PDF-1.7 ...", "pdf"),
        ("blob", None, b"<!DOCTYPE html><html>", "html"),
    ],
)
def test_type_detection(filename, mime, data, expected):
    assert REG.detect_type(filename, mime, data) == expected


def test_unsupported_type():
    with pytest.raises(UnsupportedDocumentError):
        REG.detect_type("image.png", "image/png", b"\x89PNG")
    with pytest.raises(UnsupportedDocumentError):
        REG.get("xlsx")
