# CARE DOLL RAG Module

This folder implements the offline retrieval part of CARE DOLL. It finds the most relevant passages from the emergency manuals; it never generates an answer and does not use OpenAI, Ollama, or another LLM.

## Ingestion & Extraction Flow

The document ingestion is divided into two phases to optimize retrieval startup and memory consumption:

1. **Pre-Extraction Phase (`extract_docs.py`)**:
   - Walks raw PDF documents under `documents/raw/`.
   - Uses Docling to parse and convert PDFs into clean Markdown (`.md`) and structured JSON (`.json`) files.
   - Outputs the converted files into `documents/extracted/`.

2. **RAG Retrieval Phase**:
   - Reads the pre-extracted `.md` files from `documents/extracted/` (skipping the `.json` files to avoid duplicates).
   - Checks index status. Since the files are already in Markdown, this skips heavy PyTorch/EasyOCR model loading during retrieval runtime, allowing instantaneous startups.
   - The selected chunker splits the text into metadata-preserving chunks.
   - Creates embeddings using `BAAI/bge-small-en-v1.5`.
   - Indexes and retrieves using FAISS.

## Usage

### Run Full Pipeline (Voice Input -> Speech-to-Text -> RAG)
Run it from the project root:

```powershell
.\.venv\Scripts\python.exe main.py
```

### Run Full Pipeline with Text Query (Bypass Microphone)
To test a query directly without recording audio:

```powershell
.\.venv\Scripts\python.exe main.py --query "The battery of the car is getting heated, what should I do?"
```

### Run Retrieval from the Latest Text File
To run retrieval from the newest existing transcribed query file without recording:

```powershell
.\.venv\Scripts\python.exe -m RAG.rag_pipeline
```

### Pre-Extract New raw PDFs
If you add new PDFs to `documents/raw/`, run the extraction script to sync and extract them first:

```powershell
.\.venv\Scripts\python.exe -m RAG.extract_docs
```

## Terminal output

The RAG output contains the original question followed by up to `top_k` retrieved chunks. Every chunk shows:

- similarity score (cosine similarity because embeddings are normalized)
- document filename
- Docling section heading
- chunk ID
- chunk text

There is intentionally no generated answer after these results.

## Changing the chunking method

Open [config.py](config.py) and change the `CHUNKER` constant near the top. The current method is:

```python
CHUNKER = "semantic"
```

Available values are:

| Value | Method | Use it when |
| --- | --- | --- |
| `semantic` | Heading-aware chunks split at BGE sentence topic boundaries. | Default; best first choice for manuals with mixed topics. |
| `hybrid` | Semantic chunking, heading boundaries, token size limit, and overlap. | Comparing a semantic method with stronger context continuity. |
| `fixed` | Heading-aware sentence packing by configured token size and overlap. | A baseline for chunking experiments. |
| `sentence_window` | Three-sentence windows within each heading. | Short, direct procedures where local wording matters. |
| `hierarchical` | Semantic chunks prefixed with their section heading. | Queries that benefit strongly from section context. |

For example:

```python
CHUNKER = "hybrid"
```

The next run automatically rebuilds the FAISS index because the chunker name is part of the saved index signature. No other code needs to change.

## Other settings

`RAGConfig` in [config.py](config.py) contains all retrieval settings:

- `chunk_size` — maximum chunk size, default `500` tokens
- `chunk_overlap` — context repeated between chunks, default `100` tokens
- `embedding_model` — local Sentence Transformers model
- `top_k` — number of displayed chunks, default `5`
- `document_path` — default `documents/extracted/`
- `vector_store_path` — default `data/vector_store/`

Changing the embedding model, chunk size, or overlap also triggers an automatic FAISS rebuild. Adding, changing, or removing a supported file under `documents/extracted/` does the same.

## Module responsibilities

| File | Responsibility |
| --- | --- |
| `extract_docs.py` | Pre-extracts raw PDFs using Docling, syncing categories and exporting `.md` and `.json` files. |
| `document_loader.py` | Discovers `.md` documents under `documents/extracted/` and creates content hashes. |
| `parser.py` | Parses Markdown documents into structured, heading-based sections (skips Docling converter runtime load). |
| `chunker.py` | Provides and selects chunking strategies. |
| `embedder.py` | Loads the local BGE embedding model and creates vectors. |
| `vector_store.py` | Saves/loads FAISS and metadata. |
| `retriever.py` | Performs top-K vector similarity search. |
| `rag_pipeline.py` | Coordinates indexing, question loading, and terminal output. |
| `test_rag.py` | Runs a smoke test for indexing and retrieval. |

## Testing

```powershell
.\.venv\Scripts\python.exe -m RAG.test_rag
```

The first run may download the open-source Docling and BGE model assets into the local cache. Later runs use those local assets and reuse `data/vector_store/index.faiss` when the source documents and RAG configuration have not changed.
