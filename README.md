# CARE — Emergency Assistant Software Suite

This workspace contains the CARE Emergency Assistant software components.

## Sub-Projects

### 1. [Emergency Assistance](./emergency_assistance)
The core offline-first AI Emergency Assistant app. It includes:
- RAG document ingestion pipeline for safety manuals.
- Ollama local LLM orchestration.
- Whisper audio transcription (Speech-to-Text).
- Piper audio synthesis (Text-to-Speech).
- SQLite conversation audit logging.
- FastAPI endpoints and CLI interface.

Please navigate to the [emergency_assistance](./emergency_assistance) directory for setup and running instructions.

---

## Workspace Setup

To configure the workspace virtual environment and dependencies:

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

# Ollama commands
ollama list
ollama run qwen2.5:3b