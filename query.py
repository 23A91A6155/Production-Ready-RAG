"""
query.py
========
QUERY PHASE — Orchestrator

Run this script to interactively query the RAG pipeline.
The ingestion phase (ingest.py) must have been run at least once beforehand.

Architecture flow implemented here:
  User Query
  → Query Embedding       (src/embedding_generator.py)
  → Similarity Search     (src/vector_store.py)
  → Top-K Chunk Retrieval (src/retrieval_pipeline.py)
  → Context Builder       (src/prompt_builder.py)
  → LLM Generation        (src/llm_generator.py)
  → Answer with Citations

Usage:
  python query.py          # Enter interactive query loop
  python query.py --once   # Ask a single question from CLI arg
"""

import argparse
import sys
from pathlib import Path

# Load .env before any src imports that need API keys
from dotenv import load_dotenv
load_dotenv()

import yaml

from src.vector_store import VectorStore
from src.retrieval_pipeline import retrieve_top_k_chunks
from src.prompt_builder import build_prompt
from src.llm_generator import generate_answer


def load_config(config_path: str = "config.yaml") -> dict:
    """Load and return the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        print(f"[ERROR] Config file '{config_path}' not found.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_query(query: str, config: dict, store: VectorStore) -> None:
    """
    Execute a single query through the full RAG pipeline and display results.

    Args:
        query:  The user's question string.
        config: Configuration dictionary from config.yaml.
        store:  Initialized VectorStore instance.
    """
    print("\n" + "=" * 60)
    print(f"  QUESTION: {query}")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Step 1 & 2: Embed query and retrieve Top-K chunks
    # ----------------------------------------------------------------
    retrieved_chunks = retrieve_top_k_chunks(
        query=query,
        vector_store=store,
        top_k=config["top_k"],
        embedding_model=config["embedding_model"],
    )

    if not retrieved_chunks:
        print("\n[WARN] No chunks retrieved. Cannot generate an answer.")
        return

    # ----------------------------------------------------------------
    # Step 3: Build prompt with retrieved context
    # ----------------------------------------------------------------
    prompt = build_prompt(query=query, retrieved_chunks=retrieved_chunks)

    # ----------------------------------------------------------------
    # Step 4: Generate answer via LLM
    # ----------------------------------------------------------------
    answer = generate_answer(prompt=prompt, model=config["llm_model"])

    # ----------------------------------------------------------------
    # Step 5: Display final output
    # ----------------------------------------------------------------
    _display_results(query=query, answer=answer, retrieved_chunks=retrieved_chunks)


def _display_results(
    query: str,
    answer: str,
    retrieved_chunks: list,
) -> None:
    """
    Pretty-print the generated answer and all source citations.

    Args:
        query:            The original user question.
        answer:           Generated answer from LLM.
        retrieved_chunks: List of chunk dicts used to generate the answer.
    """
    print("\n" + "=" * 60)
    print("  ANSWER")
    print("=" * 60)
    print(f"\n{answer}\n")

    print("=" * 60)
    print("  SOURCES USED")
    print("=" * 60)
    for i, chunk in enumerate(retrieved_chunks, start=1):
        print(f"\n  [CHUNK {i}]")
        print(f"  Document : {chunk['source']}")
        print(f"  Chunk #  : {chunk['chunk_index']}")
        print(f"  Similarity: {chunk['similarity']:.4f}")
        print(f"  Preview  : {chunk['text'][:200].strip()}{'...' if len(chunk['text']) > 200 else ''}")
    print("\n" + "=" * 60)


def interactive_loop(config: dict, store: VectorStore) -> None:
    """
    Run an interactive CLI query loop until the user types 'exit' or 'quit'.

    Args:
        config: Configuration dictionary from config.yaml.
        store:  Initialized VectorStore instance.
    """
    print("\n" + "=" * 60)
    print("  RAG PIPELINE - QUERY PHASE")
    print("=" * 60)
    print(f"  Vector store: {store.count()} chunks available.")
    print("  Type your question and press Enter.")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    while True:
        try:
            query = input("\n  Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  [INFO] Exiting query session.")
            break

        if not query:
            print("  [WARN] Empty question -- please type something.")
            continue

        if query.lower() in {"exit", "quit", "q"}:
            print("  [INFO] Exiting query session.")
            break

        try:
            run_query(query=query, config=config, store=store)
        except RuntimeError as e:
            print(f"\n  [ERROR] {e}")
        except EnvironmentError as e:
            print(f"\n  [ERROR] {e}")
            break
        except Exception as e:
            print(f"\n  [UNEXPECTED ERROR] {e}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG Pipeline — Query Phase: ask questions against ingested documents."
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        default=None,
        help="Ask a single question and exit (non-interactive mode).",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the configuration YAML file (default: config.yaml).",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Initialize vector store once; share across all queries in the session
    try:
        store = VectorStore(
            persist_dir=config["vector_store_dir"],
            collection_name=config["collection_name"],
        )
    except Exception as e:
        print(f"\n[ERROR] Could not open vector store: {e}")
        sys.exit(1)

    if store.is_empty():
        print(
            "\n[ERROR] Vector store is empty. Run ingestion first:\n"
            "  python ingest.py"
        )
        sys.exit(1)

    if args.question:
        # Non-interactive: answer one question and exit
        try:
            run_query(query=args.question, config=config, store=store)
        except Exception as e:
            print(f"\n[ERROR] {e}")
            sys.exit(1)
    else:
        # Interactive loop
        interactive_loop(config=config, store=store)
