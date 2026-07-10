from __future__ import annotations

import sys
from unittest.mock import MagicMock

# ==========================================
# 0. Mock external dependencies before imports
# ==========================================
mock_faiss = MagicMock()
mock_st = MagicMock()
mock_fitz = MagicMock()
mock_easyocr = MagicMock()

# Configure mock behavior for imports
sys.modules["faiss"] = mock_faiss
sys.modules["sentence_transformers"] = mock_st
sys.modules["fitz"] = mock_fitz
sys.modules["easyocr"] = mock_easyocr

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from assistant.emergency import EmergencyAssistant, ResponseValidator
from assistant.llm import ModelManager, PromptBuilder
from config.settings import Settings
from database.repository import ConversationRepository
from rag.ingestion import TextChunker, clean_text
from rag.schemas import DocumentChunk
from rag.vector_store import EmbeddingManager, FaissStore
from speech.interfaces import FasterWhisperTranscriber, PiperSpeaker


# ==========================================
# 1. Text Cleaning & Chunking Tests
# ==========================================

def test_text_cleaning() -> None:
    raw = "Hello   World!\r\nThis is a dis-\nconnect test. \x01 Control character."
    cleaned = clean_text(raw)
    assert cleaned == "Hello World!\nThis is a disconnect test. Control character."


def test_chunker_preserves_source_and_splits_long_text() -> None:
    chunker = TextChunker(chunk_size=4, overlap=1)
    chunks = chunker.chunk(Path("manual.pdf"), 2, "one two three four five six")
    assert [chunk.text for chunk in chunks] == ["one two three four", "four five six"]
    assert chunks[0].document_name == "manual.pdf"
    assert chunks[0].page == 2
    assert chunks[0].section == "General"


def test_chunker_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        TextChunker(chunk_size=3, overlap=3)


# ==========================================
# 2. Database Tests
# ==========================================

def test_sqlite_repository() -> None:
    # Use a temporary file for testing instead of ':memory:' to allow separate connections
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as temp_db:
        db_path = Path(temp_db.name)
        
    try:
        repo = ConversationRepository(db_path)
        repo.initialize()
        assert repo.check_connection() is True
        
        # Save test record
        repo.save("Help, EV is smoking!", "1. Call 911.\n2. Evacuate.")
        
        # Verify insert using direct sqlite connection
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT query, response FROM conversations")
            row = cursor.fetchone()
            assert row is not None
            assert row[0] == "Help, EV is smoking!"
            assert "Call 911" in row[1]
    finally:
        if db_path.exists():
            db_path.unlink()


# ==========================================
# 3. Prompt Builder Tests
# ==========================================

def test_prompt_builder() -> None:
    prompt = PromptBuilder.build("Smoking battery", "Manual context text here.")
    assert "CARE" in prompt
    assert "Smoking battery" in prompt
    assert "Manual context text here." in prompt
    assert "911" in prompt


# ==========================================
# 4. Response Validator Tests
# ==========================================

def test_response_validator_technical_refusal() -> None:
    query = "How do I cut the high voltage orange cable?"
    response = "Cut it with scissors."
    validated = ResponseValidator.validate(response, query, context_available=False)
    assert "I do not have the specific manufacturer manual" in validated
    assert "scissors" not in validated


def test_response_validator_emergency_prepending() -> None:
    query = "My EV is on fire!"
    response = "Get a fire extinguisher."
    validated = ResponseValidator.validate(response, query, context_available=True)
    assert "Always call local emergency services" in validated
    assert "Get a fire extinguisher" in validated


def test_response_validator_hallucination_detection() -> None:
    query = "Help, my battery is hot and smoking!"
    response = "I don't know what to do."
    validated = ResponseValidator.validate(response, query, context_available=True)
    assert "Move to a safe location" in validated
    assert "Call emergency services" in validated


# ==========================================
# 5. Embedding & FAISS Store Tests
# ==========================================

def test_embedding_manager() -> None:
    mock_st_instance = MagicMock()
    mock_st_instance.encode.return_value = np.zeros((384,), dtype=np.float32)
    mock_st.SentenceTransformer.return_value = mock_st_instance

    manager = EmbeddingManager("mock-model")
    query_emb = manager.embed_query("test query")
    assert query_emb.shape == (384,)
    
    docs_emb = manager.embed_documents(["doc1", "doc2"])
    assert docs_emb.shape == (2, 384)
    mock_st_instance.encode.assert_called()


def test_faiss_store_lifecycle() -> None:
    # Setup FAISS Index Mock
    mock_idx = MagicMock()
    mock_idx.search.return_value = (np.array([[0.9]]), np.array([[0]]))
    mock_idx.ntotal = 1
    mock_faiss.IndexFlatIP.return_value = mock_idx
    mock_faiss.read_index.return_value = mock_idx

    # Mock embedder
    mock_embedder = MagicMock(spec=EmbeddingManager)
    mock_embedder.embed_documents.return_value = np.zeros((1, 384), dtype=np.float32)
    mock_embedder.embed_query.return_value = np.zeros((384,), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        store_dir = Path(tmpdir)
        store = FaissStore(store_dir, mock_embedder)
        
        # Build index
        chunk = DocumentChunk("id1", "sample text", "doc.pdf", 1, "Intro", "/path/doc.pdf", "sample text")
        store.build([chunk])
        
        assert mock_faiss.write_index.called
        assert (store_dir / "chunks.json").exists()
        
        # Test Retrieval
        matches = store.retrieve("query text", top_k=1)
        assert len(matches) == 1
        assert matches[0].chunk.id == "id1"
        assert matches[0].score == 0.9


# ==========================================
# 6. Assistant Orchestration Tests
# ==========================================

def test_assistant_orchestration() -> None:
    mock_store = MagicMock(spec=FaissStore)
    mock_store.retrieve.return_value = [
        MagicMock(chunk=DocumentChunk("id1", "manual details", "doc.pdf", 2, "Safety", "/doc.pdf", "manual details"), score=0.8)
    ]
    
    mock_llm = MagicMock(spec=ModelManager)
    mock_llm.generate.return_value = "1. Follow manual instructions.\n2. Call emergency number."
    
    mock_repo = MagicMock(spec=ConversationRepository)

    assistant = EmergencyAssistant(mock_store, mock_llm, mock_repo, top_k=1)
    reply = assistant.answer("What to do if battery is hot?")
    
    assert reply.response == "1. Follow manual instructions.\n2. Call emergency number."
    assert len(reply.sources) == 1
    assert reply.sources[0]["source"] == "doc.pdf"
    assert reply.used_fallback is False
    assert mock_repo.save.called


# ==========================================
# 7. FastAPI Endpoint Tests
# ==========================================

@pytest.fixture
def test_client() -> TestClient:
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        # Create dummy env directories
        (base / "data" / "raw").mkdir(parents=True, exist_ok=True)
        (base / "data" / "vector_store").mkdir(parents=True, exist_ok=True)
        (base / "database").mkdir(parents=True, exist_ok=True)
        (base / "logs").mkdir(parents=True, exist_ok=True)
        
        settings = Settings(
            base_dir=base,
            host="127.0.0.1",
            port=8000,
            ollama_url="http://127.0.0.1:11434",
            ollama_model="llama3.2:3b",
            embedding_model="BAAI/bge-small-en-v1.5",
            top_k=2,
            enable_ocr=False,
            piper_command="piper",
            log_level="INFO"
        )
        
        # Mock HTTP tags endpoint for health check
        with patch("httpx.get") as mock_http_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"models": [{"name": "llama3.2:3b"}]}
            mock_resp.text = "Ollama is running"
            mock_http_get.return_value = mock_resp
            
            app = create_app(settings)
            client = TestClient(app)
            yield client


def test_api_root(test_client: TestClient) -> None:
    res = test_client.get("/")
    assert res.status_code == 200
    assert res.json() == {"project": "CARE Emergency Assistant", "status": "running"}


def test_api_health(test_client: TestClient) -> None:
    res = test_client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "components" in data
    assert "sqlite" in data["components"]
    assert "ollama" in data["components"]


@patch("assistant.emergency.EmergencyAssistant.answer")
def test_api_ask(mock_answer: MagicMock, test_client: TestClient) -> None:
    from assistant.emergency import AssistantReply
    mock_answer.return_value = AssistantReply("1. Run away.", [], False)
    
    res = test_client.post("/ask", json={"query": "EV crash"})
    assert res.status_code == 200
    assert res.json()["response"] == "1. Run away."
