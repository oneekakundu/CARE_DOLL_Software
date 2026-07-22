# CARE DOLL — Offline AI Emergency Assistant

CARE DOLL is an offline-first AI emergency assistant designed primarily for Electric Vehicle (EV) accident assistance, first-aid guidance, and safety reference. It combines local speech-to-text, Retrieval-Augmented Generation (RAG), local LLM inference via Ollama, and a dual-engine text-to-speech system to operate completely without internet connectivity.

## Project Architecture

```text
Voice Input ──► Speech-to-Text ──► RAG Retrieval ──► Local LLM ──► Text-to-Speech ──► Audio Output
(Microphone)    (Faster-Whisper)   (FAISS + BGE)     (Qwen 2.5:3b)  (Piper / XTTS)    (sounddevice)
```

## Repository Structure

```text
CARE_DOLL_Software/
├── Voice_Input/           # Audio recording and Speech-to-Text pipeline
├── Voice_Output/          # Dual-engine TTS pipeline (Piper & XTTS v2)
├── LLM/                   # Ollama local LLM integration layer
├── RAG/                   # Document extraction, embedding, and vector search
├── emergency_assistance/  # FastAPI endpoints, database audit logging, & CLI module
├── documents/             # Raw PDF manuals and pre-extracted Markdown docs
├── data/                  # Audio cache, vector index, and voice profile data
├── models/                # Local model storage (Piper ONNX weights)
├── piper/                 # Piper TTS executable binaries
├── tests/                 # Unit and integration test suite
├── main.py                # Primary application entry point
├── requirements.txt       # Core Python dependencies
└── README.md              # Developer documentation entry point
```

## Main Components

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Speech-to-Text** | Faster-Whisper | Converts spoken microphone audio to text queries |
| **Document Retrieval** | FAISS + BGE-small-en-v1.5 | Searches pre-indexed safety manuals for context |
| **Local LLM** | Ollama (Qwen 2.5:3b) | Generates grounded safety responses from context |
| **Default Emergency TTS** | Piper (`en_US-lessac-medium.onnx`) | Synthesizes rapid, lightweight emergency voice responses |
| **Personalized TTS** | XTTS v2 (Optional) | Synthesizes personalized companion speech |
| **Audio Playback** | sounddevice | Plays output WAV audio through system speakers |

## Environment Setup

CARE DOLL requires Python 3.12+ and system FFmpeg binaries. Setup the Python virtual environment (`.venv`) on Windows PowerShell:

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install project dependencies
pip install -r requirements.txt
```

## Ollama Setup

The LLM response engine requires [Ollama](https://ollama.com) installed and running locally on your system. Pull and start the Qwen 2.5 (3B) model:

```powershell
# Verify Ollama is installed and running
ollama list

# Download and run the Qwen 2.5:3b model
ollama run qwen2.5:3b
```

> **Note:** Ollama must be running as a background service or in a separate terminal. The LLM is managed locally by Ollama and is not bundled inside the Python package.

## Running the Application

Run the complete pipeline from voice input or text query using `main.py`:

```powershell
# Interactive voice input mode (records microphone input)
python main.py

# Direct text query mode (bypasses voice recording)
python main.py --query "How to handle an EV battery fire?"
```

## Voice Output Behavior

The Voice Output subsystem routes audio synthesis according to request context and profile readiness:

* **Emergency Mode:** Always uses **Piper** for guaranteed fast, reliable speech.
* **Companion Mode (Profile Not Ready):** Falls back to **Piper** if profile checks fail.
* **Companion Mode (Profile Ready):** Uses **XTTS v2** for personalized companion speech.

## Documentation

For detailed technical references, consult:

* [`description.txt`](file:///d:/CARE_DOLL_Software/description.txt): Voice Output subsystem overview, API usage, and readiness rules.
* [`explanation.txt`](file:///d:/CARE_DOLL_Software/explanation.txt): Comprehensive system architecture, model rationale, lazy loading design, and emergency reliability fallback chain.

## Troubleshooting

* **Ollama Connection Error:** Ensure the Ollama service is running (`ollama serve` or Ollama desktop app) and `qwen2.5:3b` is pulled.
* **Piper Binary / Model Missing:** Verify `piper/piper.exe` and `models/en_US-lessac-medium.onnx` exist in the repository root.
* **XTTS v2 / Torch Compatibility:** XTTS v2 requires compatible PyTorch and soundfile binaries. If XTTS fails, CARE DOLL automatically falls back to Piper.
* **FFmpeg Not Found:** Ensure FFmpeg is installed (`winget install Gyan.FFmpeg`) and available on your system `PATH`.
* **Voice Profile Readiness Failure:** Ensure `data/voices/profiles/primary_user/` contains valid `profile.json` with `"consent_confirmed": true`, `"enabled": true`, and a non-empty `reference.wav`.