"""HTML loader: extracts readable block text, sectioned by h1-h6; drops scripts/nav/chrome."""

from __future__ import annotations

from bs4 import BeautifulSoup, Tag

from app.ingestion.loaders.base import DocumentLoader, HeadingStack, ParsedDocument, Segment, decode_text

_DROP = (
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    "iframe",
)
_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_BLOCKS = {"p", "li", "pre", "blockquote", "td", "th", "dt", "dd", "figcaption", "caption", "summary"}


class HTMLLoader(DocumentLoader):
    document_type = "html"
    extensions = (".html", ".htm", ".xhtml")
    mime_types = ("text/html", "application/xhtml+xml")

    def load(self, data: bytes, filename: str | None = None) -> ParsedDocument:
        soup = BeautifulSoup(decode_text(data), "lxml")
        title = soup.title.get_text(strip=True) if soup.title else None
        for tag in soup(_DROP):
            tag.decompose()
        root = soup.find("main") or soup.find("article") or soup.body or soup

        stack = HeadingStack()
        segments: list[Segment] = []
        buf: list[str] = []

        def flush() -> None:
            if buf:
                segments.append(Segment(text="\n".join(buf), section=stack.path))
                buf.clear()

        for el in root.find_all(list(_HEADINGS) + list(_BLOCKS)):
            if not isinstance(el, Tag):
                continue
            if el.name in _HEADINGS:
                flush()
                htext = el.get_text(" ", strip=True)
                stack.push(_HEADINGS[el.name], htext)
                if title is None and el.name == "h1":
                    title = htext
                continue
            # Skip blocks nested inside another block (e.g. <p> inside <li>) - parent covers them.
            if any(p.name in _BLOCKS for p in el.parents if isinstance(p, Tag)):
                continue
            sep = "\n" if el.name == "pre" else " "
            text = el.get_text(sep, strip=el.name != "pre")
            if text.strip():
                buf.append(("- " if el.name == "li" else "") + text.strip())
        flush()
        if not segments:
            body = root.get_text("\n", strip=True)
            if body:
                segments.append(Segment(text=body))
        return ParsedDocument(segments=segments, title=title, parser="beautifulsoup-lxml")
