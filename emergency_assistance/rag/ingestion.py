from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import fitz

from rag.schemas import DocumentChunk

LOGGER = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Cleans extracted document text by normalizing whitespaces, joining hyphens, and removing control characters."""
    if not text:
        return ""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Join hyphenated words at line endings (e.g. "dis-\nconnect" -> "disconnect")
    text = re.sub(r"(\w+)-\n+(\w+)", r"\1\2", text)
    # Remove control characters except tab and newline
    text = "".join(ch for ch in text if ch >= " " or ch in ("\n", "\t"))
    # Normalize spaces on each line
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    # Re-join lines, filtering out empty lines
    return "\n".join(line for line in lines if line)


class DocumentLoader:
    """Extracts text from PDFs, with optional OCR for scanned pages."""

    def __init__(self, enable_ocr: bool = True) -> None:
        self._enable_ocr = enable_ocr

    def load_pdf(self, path: Path) -> list[tuple[int, str]]:
        """Reads a PDF file page by page, extracting and cleaning text. Applies OCR if needed."""
        LOGGER.info("Opening PDF file: '%s'", path.name)
        try:
            document = fitz.open(path)
        except Exception as error:
            LOGGER.error("Unable to read PDF '%s': %s", path.name, error)
            raise ValueError(f"Unable to read PDF '{path.name}': {error}") from error
            
        pages: list[tuple[int, str]] = []
        with document:
            for page_number, page in enumerate(document, start=1):
                raw_text = page.get_text("text").strip()
                if not raw_text and self._enable_ocr:
                    LOGGER.info("Page %d of '%s' appears empty. Attempting OCR...", page_number, path.name)
                    raw_text = self._ocr_page(page)
                
                cleaned = clean_text(raw_text)
                if cleaned:
                    pages.append((page_number, cleaned))
                    
        LOGGER.info("Successfully extracted text from %d pages in '%s'", len(pages), path.name)
        return pages

    def _ocr_page(self, page: fitz.Page) -> str:
        """Runs easyocr on a PDF page rendered as a pixmap image."""
        try:
            import easyocr
            import numpy as np
            # Render page to high-quality image (2x zoom)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
            # easyocr Reader instance. gpu=False is safer for strict CPU offline systems
            reader = easyocr.Reader(["en"], gpu=False)
            results = reader.readtext(image, detail=0)
            return " ".join(results)
        except Exception as error:
            LOGGER.warning("OCR failed or was unavailable for page %d: %s", page.number + 1, error)
            return ""


class TextChunker:
    """Chunks text into sliding-window passages with overlap and extracts heading/section metadata."""

    def __init__(self, chunk_size: int = 700, overlap: int = 120) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self._chunk_size = chunk_size
        self._overlap = overlap

    def detect_section(self, text: str) -> str:
        """Heuristic to detect heading or section title on a page."""
        lines = text.split("\n")
        header_patterns = [
            r"^(?:SECTION|Section)\s+\d+.*",
            r"^[A-Z][A-Z\s,\-]{3,40}$",        # Short uppercase line
            r"^\d+\.\d+\s+[A-Z].*",             # E.g. "1.1 Safety"
            r"^\d+\s+[A-Z].*",                  # E.g. "1 Emergency Instructions"
            r"^(?:WARNING|CAUTION|IMPORTANT|NOTE):.*",
        ]
        # Inspect the first 5 lines of text on the page
        for line in lines[:5]:
            stripped = line.strip()
            for pattern in header_patterns:
                if re.match(pattern, stripped):
                    return stripped
        return "General"

    def chunk(self, source_path: Path, page: int, text: str) -> list[DocumentChunk]:
        """Splits text into overlapping chunks, capturing full metadata."""
        section = self.detect_section(text)
        words = text.split()
        result: list[DocumentChunk] = []
        start = 0
        
        while start < len(words):
            section_text = " ".join(words[start:start + self._chunk_size])
            digest = hashlib.sha256(
                f"{source_path.name}:{page}:{start}:{section_text}".encode("utf-8")
            ).hexdigest()[:16]
            
            chunk_obj = DocumentChunk(
                id=digest,
                text=section_text,
                document_name=source_path.name,
                page=page,
                section=section,
                source_file=str(source_path.resolve()),
                original_text=section_text,
            )
            result.append(chunk_obj)
            
            if start + self._chunk_size >= len(words):
                break
            start += self._chunk_size - self._overlap
            
        return result
