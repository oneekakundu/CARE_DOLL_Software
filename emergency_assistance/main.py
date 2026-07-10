from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import uvicorn

from app.api import create_app
from assistant.emergency import EmergencyAssistant
from assistant.llm import ModelManager
from config.settings import Settings
from database.repository import ConversationRepository
from rag.ingestion import DocumentLoader, TextChunker
from rag.vector_store import EmbeddingManager, FaissStore
from speech.interfaces import FasterWhisperTranscriber, PiperSpeaker

LOGGER = logging.getLogger("main")


def configure_logging(log_dir: Path, log_level: str) -> None:
    """Configures structured logs written to logs/app.log and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"
    
    # Define log format
    log_format = "%(asctime)s - %(levelname)s - [%(name)s] - [%(filename)s:%(lineno)d] - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    LOGGER.info("Logging configured. Log level set to %s. File: %s", log_level, log_file)


def run_startup_sequence(settings: Settings) -> tuple[
    ConversationRepository,
    EmbeddingManager,
    FaissStore,
    ModelManager,
    FasterWhisperTranscriber,
    PiperSpeaker,
]:
    """Runs the startup and configuration sequence in a strict, prioritized order."""
    LOGGER.info("=== STARTING CARE EMERGENCY ASSISTANT STARTUP SEQUENCE ===")
    
    # 1. Load configuration: settings already loaded and validated before calling this
    LOGGER.info("[1/9] Configuration loaded and validated successfully.")

    # 2. Initialize logging: configure_logging called before calling this
    LOGGER.info("[2/9] Logging initialized.")

    # 3. Initialize SQLite database
    LOGGER.info("[3/9] Initializing SQLite database...")
    repository = ConversationRepository(settings.database_path)
    repository.initialize()

    # 4. Load embedding model
    LOGGER.info("[4/9] Initializing and loading embedding model...")
    embedder = EmbeddingManager(settings.embedding_model)
    try:
        embedder.load_model()
    except Exception as error:
        LOGGER.warning("Could not load embedding model at startup. It will load lazily later: %s", error)

    # 5. Load FAISS index (fallback if unavailable)
    LOGGER.info("[5/9] Checking and loading FAISS vector index...")
    store = FaissStore(settings.vector_store_dir, embedder)
    try:
        store._ensure_loaded()
    except Exception as error:
        LOGGER.warning("FAISS store loading failed or skipped: %s", error)

    # 6. Verify Ollama connection
    LOGGER.info("[6/9] Verifying Ollama connection and model existence...")
    ollama_client = ModelManager(settings.ollama_url, settings.ollama_model)
    ollama_status = ollama_client.get_status()
    if ollama_status["status"] != "healthy":
        LOGGER.warning("Ollama connection issue: %s", ollama_status["message"])
    else:
        LOGGER.info("Ollama client connection verified: %s", ollama_status["message"])

    # 7. Initialize Whisper
    LOGGER.info("[7/9] Verifying Whisper package availability...")
    transcriber = FasterWhisperTranscriber(model_size="base")
    if not transcriber.check_available():
        LOGGER.warning("Whisper packages are not available in current python environment.")
    else:
        LOGGER.info("Whisper package is available.")

    # 8. Initialize Piper
    LOGGER.info("[8/9] Verifying Piper TTS executable path...")
    speaker = PiperSpeaker(command=settings.piper_command)
    if not speaker.check_available():
        LOGGER.warning("Piper executable '%s' was not found or is not executable.", settings.piper_command)
    else:
        LOGGER.info("Piper executable is verified.")

    LOGGER.info("[9/9] Startup sequence completed. Ready to start API/CLI.")
    LOGGER.info("=== CARE STARTUP SEQUENCE COMPLETED SUCCESSFULLY ===")
    return repository, embedder, store, ollama_client, transcriber, speaker


def ingest(settings: Settings) -> int:
    """Ingests PDFs from settings.raw_data_dir and indexes them in FAISS vector store."""
    LOGGER.info("Starting ingestion workflow...")
    pdfs = list(settings.raw_data_dir.glob("*.pdf"))
    if not pdfs:
        msg = f"No PDF files found in raw directory: '{settings.raw_data_dir}'"
        LOGGER.error(msg)
        raise FileNotFoundError(msg)
        
    LOGGER.info("Found %d PDF file(s) to process: %s", len(pdfs), [p.name for p in pdfs])
    loader = DocumentLoader(settings.enable_ocr)
    chunker = TextChunker()
    
    chunks = []
    for pdf in pdfs:
        try:
            pages = loader.load_pdf(pdf)
            for page, text in pages:
                chunks.extend(chunker.chunk(pdf, page, text))
        except Exception as error:
            LOGGER.error("Failed to extract pages from PDF '%s': %s", pdf.name, error)
            
    LOGGER.info("Extracted %d chunks from document library.", len(chunks))
    
    embedder = EmbeddingManager(settings.embedding_model)
    store = FaissStore(settings.vector_store_dir, embedder)
    count = store.build(chunks)
    
    # Save a chunks JSON metadata copy
    processed_file = settings.processed_data_dir / "chunks.json"
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    import json
    processed_file.write_text(json.dumps([c.to_dict() for c in chunks], indent=2), encoding="utf-8")
    
    print(f"Indexed {count} chunks from {len(pdfs)} PDF(s).")
    LOGGER.info("Ingestion completed successfully. Index count: %d.", count)
    return count


def run_cli(settings: Settings) -> None:
    """Runs the CARE emergency assistant in terminal chat mode."""
    LOGGER.info("Initializing Assistant components for CLI mode...")
    repository = ConversationRepository(settings.database_path)
    repository.initialize()
    embedder = EmbeddingManager(settings.embedding_model)
    store = FaissStore(settings.vector_store_dir, embedder)
    ollama_client = ModelManager(settings.ollama_url, settings.ollama_model)
    
    assistant = EmergencyAssistant(store, ollama_client, repository, settings.top_k)
    
    print("\n==============================================")
    print("CARE CLI Emergency Assistant (Type 'quit' to exit)")
    print("==============================================")
    
    # Show FAISS status
    faiss_status = store.get_status()
    if not faiss_status["available"]:
        print("WARNING: FAISS vector database is missing. Assistant will operate in fallback mode.")
        LOGGER.warning("FAISS store missing during CLI chat initialization.")
    else:
        print(f"Loaded {faiss_status['chunks_count']} manual chunk(s). RAG active.")
        
    while True:
        try:
            query = input("\nYou: ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit"):
                LOGGER.info("Exiting CLI chat mode.")
                break
            
            print("CARE:")
            reply = assistant.answer(query)
            print(reply.response)
            if reply.sources:
                print("\n[Sources]")
                for src in reply.sources:
                    print(f"- {src['source']}, page {src['page']}, section {src['section']} (score: {src['score']})")
            if reply.used_fallback:
                print("\n[Notice: Running in safe offline fallback mode.]")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as error:
            print(f"CARE Error: {error}")
            LOGGER.exception("CLI error occurred")


def main() -> None:
    # 1. Parse arguments
    parser = argparse.ArgumentParser(description="CARE AI Emergency Assistance")
    parser.add_argument("--ingest", action="store_true", help="Index PDFs from data/raw")
    parser.add_argument("--cli", action="store_true", help="Run terminal chat")
    args = parser.parse_args()

    try:
        # 2. Load and validate settings
        settings = Settings.from_environment()
        
        # 3. Configure logging
        configure_logging(settings.log_dir, settings.log_level)
        
        if args.ingest:
            ingest(settings)
        elif args.cli:
            run_cli(settings)
        else:
            # Run startup verification sequence
            run_startup_sequence(settings)
            
            # 9. Start FastAPI via Uvicorn
            LOGGER.info("Starting FastAPI Uvicorn server...")
            uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
    except Exception as error:
        print(f"Fatal Startup Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
