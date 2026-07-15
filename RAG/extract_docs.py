#!/usr/bin/env python3
"""
Extract documents using Docling with dynamic directory syncing and memory management.
"""

import gc
import json
import logging
from pathlib import Path
from typing import Any

# Configure logging to show subcategory and file details clearly
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("extract_docs")

# Target Pathing
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "documents" / "raw"
EXTRACTED_DIR = PROJECT_ROOT / "documents" / "extracted"


def process_pdf(converter: Any, pdf_path: Path, extracted_subfolder: Path) -> None:
    """Processes a single PDF file, extracting it to Markdown and JSON formats."""
    category = pdf_path.parent.name
    filename = pdf_path.name
    base_name = pdf_path.stem

    md_output_path = extracted_subfolder / f"{base_name}.md"
    json_output_path = extracted_subfolder / f"{base_name}.json"

    # Memory Check & GC prior to processing
    gc.collect()

    # Skip files that have already been converted to avoid redundant processing
    if md_output_path.exists() and json_output_path.exists():
        logger.info(f"[{category}] Skipping {filename} (already converted)")
        return

    logger.info(f"[{category}] Processing {filename}...")

    try:
        # Convert document
        result = converter.convert(pdf_path)

        if not result or not getattr(result, "document", None):
            raise RuntimeError(f"Docling returned empty or invalid result for {filename}")

        doc = result.document

        # Export to Markdown
        markdown_content = doc.export_to_markdown()
        md_output_path.write_text(markdown_content, encoding="utf-8")
        logger.info(f"[{category}] Successfully exported Markdown -> {md_output_path.name}")

        # Export to JSON
        doc_dict = doc.export_to_dict()
        with open(json_output_path, "w", encoding="utf-8") as json_file:
            json.dump(doc_dict, json_file, ensure_ascii=False, indent=2)
        logger.info(f"[{category}] Successfully exported JSON -> {json_output_path.name}")

    except Exception as exc:
        logger.error(f"[{category}] Error processing {filename}: {exc}", exc_info=True)
    finally:
        # Prevent memory overload by running garbage collection after processing each file
        gc.collect()


def sync_and_extract() -> None:
    """Walks through the raw directory, syncs directories, and extracts PDF files."""
    if not RAW_DIR.exists():
        logger.error(f"Raw directory does not exist at: {RAW_DIR}")
        return

    # Ensure the base extracted directory exists
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize Docling DocumentConverter
    # We do this once to avoid reloading models into memory for every file
    logger.info("Initializing Docling DocumentConverter...")
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
    except ImportError as exc:
        logger.error("Failed to import Docling. Please make sure docling is installed.")
        logger.error(exc)
        return

    # Walk through every subfolder inside raw/
    # Note: We check all directories under RAW_DIR (depth-first or simply sorted)
    subfolders = sorted([p for p in RAW_DIR.rglob("*") if p.is_dir()])
    
    if not subfolders:
        logger.info("No subfolders found inside the raw directory.")
        return

    for raw_subfolder in subfolders:
        # Dynamic Directory Syncing
        relative_path = raw_subfolder.relative_to(RAW_DIR)
        extracted_subfolder = EXTRACTED_DIR / relative_path

        # If subfolder doesn't exist under extracted/, create it automatically
        if not extracted_subfolder.exists():
            logger.info(f"Creating matching extracted subfolder: {extracted_subfolder}")
            extracted_subfolder.mkdir(parents=True, exist_ok=True)

        # Process all PDF files inside the subfolder
        for file_path in sorted(raw_subfolder.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() == ".pdf":
                process_pdf(converter, file_path, extracted_subfolder)


if __name__ == "__main__":
    logger.info("Starting document ingestion & extraction pipeline...")
    sync_and_extract()
    logger.info("Ingestion pipeline execution complete.")
