"""
ingest.py
=========
INGESTION PHASE — Orchestrator

Run this script once to process your documents and populate the vector store.
After ingestion, you can run query.py (or main.py) without re-ingesting.

Architecture flow implemented here:
  Raw Documents
  → Document Loader       (src/document_loader.py)
  → Text Chunker          (src/text_chunker.py)
  → Embedding Generator   (src/embedding_generator.py)
  → Vector Store          (src/vector_store.py)

Usage:
  python ingest.py                  # Run ingestion with config.yaml settings
  python ingest.py --force          # Re-ingest even if data already exists
"""

import argparse
import sys
import time
from pathlib import Path

# Load .env before any src imports that need API keys
from dotenv import load_dotenv
load_dotenv()

import yaml

from src.document_loader import load_documents
from src.text_chunker import chunk_documents
from src.embedding_generator import get_embeddings
from src.vector_store import VectorStore


def load_config(config_path: str = "config.yaml") -> dict:
    """Load and return the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        print(f"[ERROR] Config file '{config_path}' not found.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_ingestion(config: dict, force: bool = False) -> None:
    """
    Execute the full ingestion pipeline.

    Args:
        config: Configuration dictionary from config.yaml.
        force:  If True, re-ingest even if the vector store already has data.
    """
    print("=" * 60)
    print("  RAG PIPELINE - INGESTION PHASE")
    print("=" * 60)
    start_time = time.time()

    # ----------------------------------------------------------------
    # 1. Initialize Vector Store
    # ----------------------------------------------------------------
    print("\n[STEP 1/4] Initializing vector store...")
    store = VectorStore(
        persist_dir=config["vector_store_dir"],
        collection_name=config["collection_name"],
    )

    # Check if ingestion is necessary
    if not store.is_empty() and not force:
        print(
            f"\n  [SKIP] Vector store already contains {store.count()} chunks.\n"
            f"         Use '--force' to re-ingest: python ingest.py --force\n"
            f"         Or run queries directly: python query.py"
        )
        return

    # ----------------------------------------------------------------
    # 2. Load Documents
    # ----------------------------------------------------------------
    print(f"\n[STEP 2/4] Loading documents from '{config['documents_dir']}'...")
    documents = load_documents(config["documents_dir"])
    print(f"  [OK] Loaded {len(documents)} document(s).")

    # ----------------------------------------------------------------
    # 3. Chunk Documents
    # ----------------------------------------------------------------
    print(f"\n[STEP 3/4] Chunking documents "
          f"(size={config['chunk_size']}, overlap={config['chunk_overlap']})...")
    chunks = chunk_documents(
        documents=documents,
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
    )
    print(f"  [OK] Created {len(chunks)} chunk(s) total.")

    # ----------------------------------------------------------------
    # 4. Generate Embeddings
    # ----------------------------------------------------------------
    print(f"\n[STEP 4/4] Generating embeddings using '{config['embedding_model']}'...")
    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = get_embeddings(chunk_texts, model=config["embedding_model"])
    print(f"  [OK] Generated {len(embeddings)} embedding vector(s).")

    # ----------------------------------------------------------------
    # 5. Store in Vector Database
    # ----------------------------------------------------------------
    print(f"\nStoring embeddings in vector store...")
    store.upsert_chunks(chunks=chunks, embeddings=embeddings)
    print(f"  [OK] Vector store now contains {store.count()} chunk(s).")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("  INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Documents processed : {len(documents)}")
    print(f"  Chunks created      : {len(chunks)}")
    print(f"  Embeddings stored   : {len(embeddings)}")
    print(f"  Vector store path   : {config['vector_store_dir']}")
    print(f"  Time elapsed        : {elapsed:.2f}s")
    print("\n  You can now run queries:")
    print("    python query.py")
    print("    python main.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG Pipeline — Ingestion Phase: load, chunk, embed, and store documents."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest documents even if the vector store already has data.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the configuration YAML file (default: config.yaml).",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    try:
        run_ingestion(config=config, force=args.force)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except EnvironmentError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR] {e}")
        raise
