"""Convert documents with Docling into structured sections for offline RAG."""

from __future__ import annotations

import gc
import re
import time
from typing import Any

from .logger import get_logger
from .utils import DocumentFile, ParsedDocument, Section

LOGGER = get_logger(__name__)


class DoclingParser:
    """Docling adapter tuned for searchable PDFs and structured RAG chunking."""

    def __init__(self) -> None:
        self._converter = None

    def parse(self, document: DocumentFile) -> ParsedDocument:
        start_time = time.perf_counter()
        metadata: dict[str, Any] = {
            "source": document.path.name,
            "path": str(document.path),
            "content_hash": document.content_hash,
            "document_type": document.path.suffix.lower(),
            "chunk_id": None,
        }
        try:
            if document.path.suffix.lower() == ".md":
                markdown = document.path.read_text(encoding="utf-8")
            else:
                markdown = self._extract_markdown(document)
            sections = self._sections_from_markdown(
                markdown,
                source=document.path.name,
                path=document.path,
                content_hash=document.content_hash,
                document_type=document.path.suffix.lower(),
            )
            if not sections:
                sections = [
                    Section(
                        heading="Document",
                        text=self._clean_text(markdown),
                        heading_level=0,
                        metadata={**metadata, "section_heading": "Document"},
                    )
                ]
            elapsed = time.perf_counter() - start_time
            LOGGER.info("Document Parsed: %s | sections=%d | elapsed=%.3fs", document.path.name, len(sections), elapsed)
            LOGGER.info("Number of Sections: %s -> %d", document.path.name, len(sections))
            return ParsedDocument(source=document.path.name, sections=sections, metadata=metadata)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.exception("Document Failed: %s", document.path.name)
            metadata["error"] = str(exc)
            return ParsedDocument(source=document.path.name, sections=[], metadata=metadata)
        finally:
            gc.collect()
            LOGGER.info("Memory Released: %s", document.path.name)

    def _extract_markdown(self, document: DocumentFile) -> str:
        if self._converter is None:
            self._converter = self._build_converter()
        if document.path.suffix.lower() == ".pdf":
            self._configure_pdf_pipeline(do_ocr=False, force_backend_text=True)
            try:
                result = self._converter.convert(document.path)
                markdown = self._export_markdown(result)
                if not self._has_meaningful_text(markdown):
                    LOGGER.warning("No readable text extracted for %s; retrying with OCR fallback", document.path.name)
                    self._configure_pdf_pipeline(do_ocr=True, force_backend_text=False)
                    try:
                        result = self._converter.convert(document.path)
                        markdown = self._export_markdown(result)
                    except Exception as exc:
                        LOGGER.warning("OCR fallback failed for %s: %s", document.path.name, exc)
                        return self._extract_with_pymupdf(document)
                return markdown
            except Exception as exc:
                LOGGER.warning("Docling extraction failed for %s: %s; falling back to PyMuPDF", document.path.name, exc)
                return self._extract_with_pymupdf(document)
            finally:
                self._configure_pdf_pipeline(do_ocr=False, force_backend_text=True)
        try:
            result = self._converter.convert(document.path)
            return self._export_markdown(result)
        except Exception as exc:
            LOGGER.warning("Document conversion failed for %s: %s", document.path.name, exc)
            if document.path.suffix.lower() == ".pdf":
                return self._extract_with_pymupdf(document)
            raise

    def _extract_with_pymupdf(self, document: DocumentFile) -> str:
        try:
            import fitz
        except ImportError as exc:  # pragma: no cover - defensive
            raise RuntimeError("PyMuPDF is required for PDF fallback extraction") from exc

        doc = fitz.open(document.path)
        try:
            sections: list[str] = []
            for page_number in range(doc.page_count):
                page = doc.load_page(page_number)
                text = page.get_text("text")
                if text and text.strip():
                    sections.append(f"## Page {page_number + 1}\n{text.strip()}")
            markdown = "\n\n".join(sections)
            if not self._has_meaningful_text(markdown):
                raise RuntimeError("PyMuPDF returned no readable text")
            return markdown
        finally:
            doc.close()

    @staticmethod
    def _build_converter() -> Any:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise RuntimeError("Docling is required. Install project requirements before running RAG.") from exc
        return DocumentConverter()

    def _configure_pdf_pipeline(self, *, do_ocr: bool, force_backend_text: bool) -> None:
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption
        except ImportError as exc:  # pragma: no cover - defensive
            raise RuntimeError("Docling PDF pipeline configuration is unavailable") from exc

        format_to_options = getattr(self._converter, "format_to_options", None)
        if format_to_options is None or not hasattr(format_to_options, "__setitem__"):
            LOGGER.debug("Converter %s does not support PDF pipeline option injection; skipping configuration", type(self._converter).__name__)
            return

        pipeline_options = PdfPipelineOptions(
            do_ocr=do_ocr,
            force_backend_text=force_backend_text,
            do_table_structure=True,
            do_code_enrichment=False,
            do_formula_enrichment=False,
            generate_page_images=False,
            generate_picture_images=False,
        )
        format_to_options[InputFormat.PDF] = PdfFormatOption(pipeline_options=pipeline_options)

    @staticmethod
    def _export_markdown(result: Any) -> str:
        if result is None:
            raise RuntimeError("Docling returned no conversion result")
        document = getattr(result, "document", None)
        if document is None:
            raise RuntimeError("Docling produced no document object")
        markdown = document.export_to_markdown()
        if not isinstance(markdown, str) or not markdown.strip():
            raise RuntimeError("Docling produced no readable markdown")
        return markdown

    @staticmethod
    def _has_meaningful_text(markdown: str) -> bool:
        return bool(re.sub(r"\s+", "", markdown).strip())

    @staticmethod
    def _clean_text(text: str) -> str:
        normalized = re.sub(r"\r\n?", "\n", text)
        lines = [line.rstrip() for line in normalized.splitlines() if line.strip()]
        return "\n".join(lines).strip()

    def _sections_from_markdown(
        self,
        markdown: str,
        *,
        source: str,
        path: Any,
        content_hash: str,
        document_type: str,
    ) -> list[Section]:
        sections: list[Section] = []
        heading_stack: list[str] = []
        buffer: list[str] = []
        current_level = 0

        def flush_section(level: int, title: str) -> None:
            text = self._clean_text("\n".join(buffer))
            if text:
                heading_name = " > ".join(heading_stack) if heading_stack else title or "Document"
                metadata = {
                    "source": source,
                    "path": str(path),
                    "content_hash": content_hash,
                    "document_type": document_type,
                    "section_heading": heading_name,
                    "heading_level": level,
                    "page_number": None,
                    "chunk_id": None,
                }
                sections.append(
                    Section(
                        heading=heading_name,
                        text=text,
                        heading_level=level,
                        metadata=metadata,
                    )
                )

        for line in markdown.splitlines():
            match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if match:
                flush_section(current_level, "")
                level = len(match.group(1))
                title = match.group(2).strip()
                while len(heading_stack) >= level:
                    heading_stack.pop()
                heading_stack.append(title)
                current_level = level
                buffer = []
            else:
                buffer.append(line)

        flush_section(current_level, "Document")
        return sections
