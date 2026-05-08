"""
retrieval_pipeline.py
=====================
QUERY PHASE — Steps 1 & 2

Responsibilities:
  - Accept a user query string
  - Embed the query using the SAME model used during ingestion
  - Perform cosine similarity search against the vector store
  - Return Top-K most relevant chunks with their text and metadata

This module is the bridge between the user's question and the context
that will be injected into the LLM prompt.
"""

from typing import List, Dict, Any
from src.embedding_generator import get_query_embedding
from src.vector_store import VectorStore


def retrieve_top_k_chunks(
    query: str,
    vector_store: VectorStore,
    top_k: int = 4,
    embedding_model: str = "text-embedding-3-small",
) -> List[Dict[str, Any]]:
    """
    Full retrieval pipeline: embed query → cosine search → return Top-K chunks.

    Args:
        query:           The user's question string.
        vector_store:    Initialized VectorStore instance.
        top_k:           Number of chunks to retrieve (3–5 recommended).
        embedding_model: Must be the same model used during ingestion.

    Returns:
        List of up to *top_k* chunk dicts, each containing:
          - text:        the chunk text
          - source:      source filename
          - chunk_index: position in source document
          - distance:    cosine distance
          - similarity:  1 - distance (higher = more relevant)

    Raises:
        ValueError: If the query is empty.
        RuntimeError: If the vector store is empty (ingestion not run yet).
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if vector_store.is_empty():
        raise RuntimeError(
            "The vector store is empty. "
            "Please run the ingestion phase first:\n"
            "  python ingest.py"
        )

    print(f"\n  [RETRIEVE] Embedding query...")
    query_embedding = get_query_embedding(query, model=embedding_model)

    print(f"  [RETRIEVE] Searching for Top-{top_k} similar chunks...")
    results = vector_store.similarity_search(
        query_embedding=query_embedding,
        top_k=top_k,
    )

    print(f"  [RETRIEVE] Found {len(results)} matching chunks.")
    for i, chunk in enumerate(results, start=1):
        print(
            f"    Chunk {i}: source='{chunk['source']}', "
            f"index={chunk['chunk_index']}, "
            f"similarity={chunk['similarity']:.4f}"
        )

    return results
