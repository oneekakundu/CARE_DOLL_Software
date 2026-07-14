from __future__ import annotations

from pathlib import Path

from RAG.parser import DoclingParser
from RAG.utils import DocumentFile


class _FakeConverter:
    def convert(self, path: Path):
        raise RuntimeError("bad_alloc")


def test_parser_falls_back_to_pymupdf_when_docling_crashes(tmp_path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    document = DocumentFile(path=pdf_path, content_hash="abc123")

    parser = DoclingParser()
    parser._converter = _FakeConverter()
    monkeypatch.setattr(parser, "_configure_pdf_pipeline", lambda *args, **kwargs: None)
    monkeypatch.setattr(parser, "_extract_with_pymupdf", lambda doc: "## Heading\nFallback text")

    parsed = parser.parse(document)

    assert parsed.sections, "parser should return at least one section after fallback"
    assert "Fallback text" in parsed.sections[0].text
