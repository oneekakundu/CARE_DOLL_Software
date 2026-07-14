# CARE DOLL RAG Module

This folder implements the offline retrieval part of CARE DOLL. It finds the most relevant passages from the emergency manuals; it never generates an answer and does not use OpenAI, Ollama, or another LLM.

## Flow

```text
documents/
  → Docling parses PDFs and other supported documents
  → selected chunker creates metadata-preserving chunks
  → BAAI/bge-small-en-v1.5 creates local embeddings
  → FAISS stores/searches the vectors
  → top matching chunks are displayed in the terminal
```

`main.py` runs the complete implemented workflow:

```text
Microphone → Voice_Input/record_audio.py → audio WAV
→ Voice_Input/Speech_to_text.py → data/text/<recording>.txt
→ RAG retrieval → terminal chunks
```

Run it from the project root after activating the virtual environment:

```powershell
.\.venv\Scripts\python.exe main.py
```

To run retrieval from the newest existing text file without recording new audio:

```powershell
.\.venv\Scripts\python.exe -m RAG.rag_pipeline
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
- `document_path` — default `documents/`
- `vector_store_path` — default `data/vector_store/`

Changing the embedding model, chunk size, or overlap also triggers an automatic FAISS rebuild. Adding, changing, or removing a supported file under `documents/` does the same.

## Module responsibilities

| File | Responsibility |
| --- | --- |
| `document_loader.py` | Discovers documents and creates content hashes. |
| `parser.py` | Uses Docling to convert source documents into heading-based sections. |
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
