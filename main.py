"""
main.py
=======
COMBINED RUNNER — Ingestion + Query

This is the single entry point that:
  1. Runs the ingestion phase (skips if already done)
  2. Enters the interactive query loop

Use this for a complete end-to-end demonstration of the RAG pipeline.

Usage:
  python main.py            # Auto-ingest if needed, then query loop
  python main.py --force    # Force re-ingestion, then query loop
"""

import argparse
import sys
from pathlib import Path

# Load .env before any src imports that need API keys
from dotenv import load_dotenv
load_dotenv()

import yaml

from ingest import run_ingestion, load_config as _load_config
from query import interactive_loop
from src.vector_store import VectorStore


def load_config(config_path: str = "config.yaml") -> dict:
    """Load and return the YAML configuration file."""
    return _load_config(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "RAG Pipeline — Main Runner: "
            "auto-ingests documents if needed, then enters the interactive query loop."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion even if the vector store already has data.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the configuration YAML file (default: config.yaml).",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    print("\n" + "=" * 60)
    print("  PRODUCTION-READY RAG PIPELINE")
    print("=" * 60)
    print("  Architecture:")
    print("    INGESTION : Documents -> Loader -> Chunker -> Embedder -> VectorStore")
    print("    QUERY     : Question -> Embed -> Search -> Context -> LLM -> Answer")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Phase 1: Ingestion (auto-skips if already done)
    # ----------------------------------------------------------------
    try:
        run_ingestion(config=config, force=args.force)
    except (FileNotFoundError, ValueError, EnvironmentError) as e:
        print(f"\n[ERROR] Ingestion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR during ingestion] {e}")
        raise

    # ----------------------------------------------------------------
    # Phase 2: Query loop
    # ----------------------------------------------------------------
    try:
        store = VectorStore(
            persist_dir=config["vector_store_dir"],
            collection_name=config["collection_name"],
        )
        interactive_loop(config=config, store=store)
    except EnvironmentError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR during query] {e}")
        raise
