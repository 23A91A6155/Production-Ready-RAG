# Production-Ready RAG Pipeline

A clean, modular Retrieval-Augmented Generation (RAG) pipeline built in Python that demonstrates the full RAG architecture from document ingestion to grounded answer generation.

---

## Architecture

```
INGESTION PHASE:
  Raw Documents (.txt / .md)
      → Document Loader      (src/document_loader.py)
      → Text Chunker         (src/text_chunker.py)     [fixed-size + overlap]
      → Embedding Generator  (src/embedding_generator.py) [text-embedding-3-small]
      → Vector Store         (src/vector_store.py)     [ChromaDB, cosine similarity]

QUERY PHASE:
  User Question
      → Query Embedding      (src/embedding_generator.py)
      → Similarity Search    (src/vector_store.py)
      → Top-K Retrieval      (src/retrieval_pipeline.py)
      → Context Builder      (src/prompt_builder.py)
      → LLM Generation       (src/llm_generator.py)   [gpt-4o-mini]
      → Answer + Citations
```

---

## Project Structure

```
Production-Ready RAG/
├── documents/                  # Place your .txt / .md files here
│   ├── rag_overview.txt        # Sample document (included)
│   └── vector_databases.md     # Sample document (included)
├── vector_store_db/            # ChromaDB persists here (auto-created)
├── src/
│   ├── __init__.py
│   ├── document_loader.py      # Load + clean raw documents
│   ├── text_chunker.py         # Fixed-size overlapping chunking (tiktoken)
│   ├── embedding_generator.py  # OpenAI text-embedding-3-small
│   ├── vector_store.py         # ChromaDB cosine similarity store
│   ├── retrieval_pipeline.py   # Embed query + Top-K search
│   ├── prompt_builder.py       # Exact RAG prompt template
│   └── llm_generator.py        # OpenAI gpt-4o-mini completion
├── ingest.py                   # Ingestion phase runner
├── query.py                    # Query phase runner (interactive CLI)
├── main.py                     # Combined runner (ingest + query)
├── config.yaml                 # All tuneable parameters
├── requirements.txt
├── .env.example                # API key template
└── README.md
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10 or higher
- An OpenAI API key

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

```bash
# Copy the template
copy .env.example .env

# Edit .env and add your key:
# OPENAI_API_KEY=sk-your-key-here
```

### 4. Add Your Documents

Place `.txt` or `.md` files in the `documents/` directory.
Two sample documents are already included for testing.

### 5. Run Ingestion

```bash
python ingest.py
```

This reads all documents, chunks them, generates embeddings, and stores everything in ChromaDB locally. **Run this only once** (or with `--force` to re-ingest).

### 6. Ask Questions

```bash
# Interactive mode
python query.py

# Single question
python query.py --question "What is cosine similarity?"

# Combined (auto-ingest + query loop)
python main.py
```

---

## Configuration

All parameters are in `config.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `documents_dir` | `./documents` | Directory containing raw documents |
| `vector_store_dir` | `./vector_store_db` | ChromaDB persistence directory |
| `collection_name` | `rag_chunks` | ChromaDB collection name |
| `chunk_size` | `400` | Tokens per chunk (300–500) |
| `chunk_overlap` | `80` | Overlapping tokens between chunks |
| `top_k` | `4` | Chunks retrieved per query (3–5) |
| `embedding_model` | `text-embedding-3-small` | OpenAI embedding model |
| `llm_model` | `gpt-4o-mini` | OpenAI chat model |

---

## Example Session

```
$ python main.py

======================================================
  PRODUCTION-READY RAG PIPELINE
======================================================
  Architecture:
    INGESTION : Documents → Loader → Chunker → Embedder → VectorStore
    QUERY     : Question → Embed → Search → Context → LLM → Answer
======================================================

[STEP 1/4] Initializing vector store...
[STEP 2/4] Loading documents from './documents'...
  [LOAD] 'rag_overview.txt' — 4823 characters loaded.
  [LOAD] 'vector_databases.md' — 3912 characters loaded.
[STEP 3/4] Chunking documents (size=400, overlap=80)...
  [CHUNK] 'rag_overview.txt' — 912 tokens → 3 chunks
  [CHUNK] 'vector_databases.md' — 748 tokens → 2 chunks
[STEP 4/4] Generating embeddings using 'text-embedding-3-small'...
  ✓ Generated 5 embedding vector(s).
  [STORE] Upserted 5 chunks into 'rag_chunks'.

  Question: What is the purpose of chunk overlap in RAG?

======================================================
  ANSWER
======================================================

Chunk overlap is used to avoid losing context at chunk boundaries.
If a key sentence spans the end of one chunk and the beginning of
the next, overlap ensures that at least one chunk captures it fully.
A typical overlap is 10 to 20 percent of the chunk size.

======================================================
  SOURCES USED
======================================================

  [CHUNK 1]
  Document : rag_overview.txt
  Chunk #  : 1
  Similarity: 0.8923
  Preview  : ...overlap is used to avoid losing context at chunk boundaries...

  [CHUNK 2]
  Document : rag_overview.txt
  Chunk #  : 0
  Similarity: 0.8541
  Preview  : ...fixed-size chunking with overlap to preserve cross-boundary context...
```

---

## Behavior When Answer Is Not in Context

When a question cannot be answered from the retrieved chunks, the system returns:

```
I don't know.
```

This is enforced by the prompt template, which explicitly instructs the LLM to use this exact phrase when the context does not contain the answer.

---

## Key Design Decisions

1. **Token-based chunking** using `tiktoken` ensures chunk sizes are accurate in terms of what embedding models actually see.

2. **Cosine similarity** is used (not Euclidean) because it is direction-based and insensitive to vector magnitude — ideal for text embeddings.

3. **ChromaDB PersistentClient** stores data to disk so ingestion runs only once. The vector store loads instantly on subsequent runs.

4. **Idempotent upserts** mean re-running ingestion on the same documents does not create duplicates.

5. **Temperature 0.0** is used for LLM generation to produce deterministic, factual answers that closely follow the context.

6. **Same embedding model** is enforced for both ingestion and query — using different models would produce incompatible vector spaces.

---

## Module Responsibilities

| Module | Phase | Responsibility |
|---|---|---|
| `document_loader.py` | Ingestion | Scan dir, read files, clean text |
| `text_chunker.py` | Ingestion | Fixed-size overlapping token chunks |
| `embedding_generator.py` | Both | Convert text → float vectors |
| `vector_store.py` | Both | ChromaDB upsert + cosine search |
| `retrieval_pipeline.py` | Query | Embed query + retrieve Top-K |
| `prompt_builder.py` | Query | Build exact RAG prompt |
| `llm_generator.py` | Query | Send prompt → get answer |

---

## Adding Your Own Documents

1. Place any `.txt` or `.md` file in the `documents/` directory.
2. Run `python ingest.py --force` to re-ingest with the new document included.
3. Ask questions that relate to the new document's content.
