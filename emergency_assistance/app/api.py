from __future__ import annotations

import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from assistant.emergency import EmergencyAssistant
from assistant.llm import ModelManager
from config.settings import Settings
from database.repository import ConversationRepository
from rag.ingestion import DocumentLoader, TextChunker
from rag.vector_store import EmbeddingManager, FaissStore
from speech.interfaces import FasterWhisperTranscriber, PiperSpeaker

LOGGER = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000, examples=["My EV battery is smoking."])


class QueryResponse(BaseModel):
    response: str
    sources: list[dict[str, object]]
    used_fallback: bool


def cleanup_temporary_files(*paths: Path) -> None:
    """Background task to delete temporary files created during voice processing."""
    for path in paths:
        try:
            if path.exists():
                path.unlink()
                LOGGER.info("Deleted temporary audio file: %s", path)
        except Exception as error:
            LOGGER.warning("Failed to clean up temporary file '%s': %s", path, error)


def create_app(settings: Settings) -> FastAPI:
    # Set up repositories and clients
    repository = ConversationRepository(settings.database_path)
    embedder = EmbeddingManager(settings.embedding_model)
    store = FaissStore(settings.vector_store_dir, embedder)
    
    ollama_client = ModelManager(settings.ollama_url, settings.ollama_model)
    assistant = EmergencyAssistant(store, ollama_client, repository, settings.top_k)
    
    # Initialize speech adapters
    transcriber = FasterWhisperTranscriber(model_size="base")
    speaker = PiperSpeaker(command=settings.piper_command)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        LOGGER.info("Starting up CARE Emergency Assistant API...")
        
        # 1. Initialize SQLite Database
        try:
            repository.initialize()
        except Exception as error:
            LOGGER.error("Startup Database Error: %s", error)
            
        # 2. Check FAISS Index Presence
        try:
            store._ensure_loaded()
        except Exception as error:
            LOGGER.warning("Startup FAISS store check: %s", error)
            
        # 3. Check Ollama Status
        ollama_status = ollama_client.get_status()
        if ollama_status["status"] != "healthy":
            LOGGER.warning("Startup Ollama connectivity warning: %s", ollama_status["message"])
        else:
            LOGGER.info("Startup Ollama connection verified.")
            
        yield
        LOGGER.info("Shutting down CARE Emergency Assistant API...")

    app = FastAPI(title="CARE Emergency Assistance", version="0.2.0", lifespan=lifespan)

    @app.get("/")
    def read_root() -> dict[str, str]:
        """Root endpoint returning basic metadata."""
        return {
            "project": "CARE Emergency Assistant",
            "status": "running"
        }

    @app.get("/health")
    def health() -> dict[str, object]:
        """Provides detailed health status of all system sub-components."""
        db_ok = repository.check_connection()
        faiss_status = store.get_status()
        ollama_status = ollama_client.get_status()
        whisper_ok = transcriber.check_available()
        piper_ok = speaker.check_available()
        
        overall_healthy = (
            db_ok 
            and faiss_status["available"] 
            and ollama_status["status"] == "healthy"
            and whisper_ok
            and piper_ok
        )

        return {
            "status": "healthy" if overall_healthy else "degraded",
            "components": {
                "sqlite": "connected" if db_ok else "unreachable",
                "faiss": {
                    "status": "available" if faiss_status["available"] else "missing",
                    "chunks_loaded": faiss_status["chunks_count"]
                },
                "ollama": ollama_status,
                "whisper": "available" if whisper_ok else "unavailable",
                "piper": "available" if piper_ok else "unavailable"
            }
        }

    @app.post("/ask", response_model=QueryResponse)
    def ask(request: QueryRequest) -> QueryResponse:
        """Processes a text query against indexed manuals and returns the safety response."""
        try:
            reply = assistant.answer(request.query)
            return QueryResponse(
                response=reply.response, 
                sources=reply.sources, 
                used_fallback=reply.used_fallback
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            LOGGER.exception("Internal assistant query processing failed")
            raise HTTPException(status_code=500, detail=f"Assistant failed: {error}") from error

    @app.post("/voice")
    async def voice(
        background_tasks: BackgroundTasks, 
        file: UploadFile = File(...)
    ) -> FileResponse:
        """Accepts an audio file upload, transcribes it, runs the emergency assistant, and returns synthesized audio."""
        LOGGER.info("Received audio upload: %s", file.filename)
        
        # Save upload to temporary file
        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_in:
            content = await file.read()
            temp_in.write(content)
            temp_in_path = Path(temp_in.name)

        temp_out_path = Path(tempfile.mktemp(suffix=".wav"))
        
        try:
            # 1. Transcribe speech
            query_text = transcriber.transcribe(temp_in_path)
            if not query_text.strip():
                raise ValueError("No speech or transcription detected in audio file.")
            
            # 2. Get answer from Assistant
            reply = assistant.answer(query_text)
            
            # 3. Generate TTS WAV output
            speaker.speak(reply.response, temp_out_path)
            
            # Register background cleanup task
            background_tasks.add_task(cleanup_temporary_files, temp_in_path, temp_out_path)
            
            # Return wave file with custom metadata headers
            return FileResponse(
                path=str(temp_out_path),
                media_type="audio/wav",
                headers={
                    "X-Query-Text": query_text.encode("utf-8").decode("latin1"),
                    "X-Response-Text": reply.response.encode("utf-8").decode("latin1"),
                    "X-Used-Fallback": str(reply.used_fallback)
                }
            )
        except ValueError as error:
            cleanup_temporary_files(temp_in_path, temp_out_path)
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            LOGGER.exception("Failed to process voice request")
            cleanup_temporary_files(temp_in_path, temp_out_path)
            raise HTTPException(status_code=500, detail=f"Voice processing failed: {error}") from error

    @app.post("/ingest")
    def trigger_ingest() -> dict[str, object]:
        """Manually triggers extraction and indexing of PDF files in the raw data directory."""
        LOGGER.info("API request to trigger document ingestion.")
        try:
            pdfs = list(settings.raw_data_dir.glob("*.pdf"))
            if not pdfs:
                raise FileNotFoundError(f"No PDF files found in raw directory: '{settings.raw_data_dir}'")
                
            loader = DocumentLoader(settings.enable_ocr)
            chunker = TextChunker()
            
            chunks = []
            for pdf in pdfs:
                pages = loader.load_pdf(pdf)
                for page, text in pages:
                    chunks.extend(chunker.chunk(pdf, page, text))
            
            count = store.build(chunks)
            
            # Re-ensure store reads index
            store._index = None
            store._ensure_loaded()
            
            return {
                "status": "success",
                "message": f"Successfully indexed {count} chunks from {len(pdfs)} PDF(s).",
                "indexed_chunks": count,
                "files": [pdf.name for pdf in pdfs]
            }
        except FileNotFoundError as error:
            LOGGER.warning("Ingestion triggered but failed: %s", error)
            raise HTTPException(status_code=404, detail=str(error)) from error
        except Exception as error:
            LOGGER.exception("PDF ingestion pipeline failed")
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {error}") from error

    return app
