"""
vector_store.py
===============
INGESTION PHASE — Step 4  |  QUERY PHASE — Step 2

Responsibilities:
  - Wrap ChromaDB for local, persistent vector storage
  - Upsert chunk embeddings with text and metadata during ingestion
  - Perform cosine similarity search to retrieve Top-K chunks during query

ChromaDB is chosen because:
  - Stores data locally on disk (no external server needed)
  - Uses cosine similarity natively (via "cosine" space)
  - Simple Python API with no extra infrastructure

Collection schema per chunk document:
  id:        unique string ID (e.g. "sample_txt__chunk_0")
  embedding: List[float]  — the embedding vector
  document:  str          — the chunk text (ChromaDB calls this "document")
  metadata:
    source:      str  — source filename
    chunk_index: int  — 0-based chunk position within the document
"""

import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings


class VectorStore:
    """
    Local ChromaDB vector store for RAG chunk storage and retrieval.

    Usage:
        store = VectorStore(persist_dir="./vector_store_db", collection_name="rag_chunks")
        store.upsert_chunks(chunks, embeddings)
        results = store.similarity_search(query_embedding, top_k=4)
    """

    def __init__(self, persist_dir: str, collection_name: str):
        """
        Initialize the ChromaDB client and get (or create) the collection.

        Args:
            persist_dir:     Directory path where ChromaDB persists data.
            collection_name: Name of the ChromaDB collection.
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # Ensure the persist directory exists
        os.makedirs(persist_dir, exist_ok=True)

        # Persistent client — data survives across process restarts
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Get or create collection with cosine similarity space
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        print(f"  [STORE] ChromaDB collection '{collection_name}' ready "
              f"at '{persist_dir}' -- {self.collection.count()} existing chunks.")

    # ------------------------------------------------------------------
    # Ingestion methods
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: List[Dict],
        embeddings: List[List[float]],
    ) -> None:
        """
        Insert or update chunks with their embeddings into the collection.
        Uses upsert so re-running ingestion on unchanged files is idempotent.

        Args:
            chunks:     List of chunk dicts (text, source, chunk_index).
            embeddings: Corresponding list of embedding vectors.

        Raises:
            ValueError: If lengths of chunks and embeddings do not match.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings."
            )

        ids = []
        documents = []
        metadatas = []
        embedding_list = []

        for chunk, embedding in zip(chunks, embeddings):
            # Build a deterministic, unique ID from source + chunk index
            chunk_id = _build_chunk_id(chunk["source"], chunk["chunk_index"])

            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append({
                "source": chunk["source"],
                "chunk_index": int(chunk["chunk_index"]),
            })
            embedding_list.append(embedding)

        # ChromaDB upsert in one call (handles batching internally)
        self.collection.upsert(
            ids=ids,
            embeddings=embedding_list,
            documents=documents,
            metadatas=metadatas,
        )

        print(f"  [STORE] Upserted {len(ids)} chunks into '{self.collection_name}'.")

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Find the Top-K most similar chunks to the query embedding using cosine similarity.

        Args:
            query_embedding: Embedding vector for the user's query.
            top_k:           Number of results to return.

        Returns:
            List of result dicts, each containing:
              - text:        chunk text
              - source:      source filename
              - chunk_index: position in source document
              - distance:    cosine distance (lower = more similar)
              - similarity:  1 - distance (higher = more similar)
        """
        total_chunks = self.collection.count()
        if total_chunks == 0:
            return []

        # Cap top_k to the number of available chunks
        actual_k = min(top_k, total_chunks)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
        )

        # Unpack ChromaDB's nested response format
        retrieved: List[Dict[str, Any]] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "distance": round(dist, 6),
                "similarity": round(1.0 - dist, 6),
            })

        return retrieved

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of chunks currently stored in the collection."""
        return self.collection.count()

    def is_empty(self) -> bool:
        """Return True if the collection contains no chunks."""
        return self.count() == 0


def _build_chunk_id(source: str, chunk_index: int) -> str:
    """
    Build a deterministic unique ID for a chunk.
    Sanitizes the source filename to be safe as a ChromaDB ID component.

    Args:
        source:      Source filename (e.g. "sample.txt").
        chunk_index: 0-based chunk position.

    Returns:
        String ID like "sample_txt__chunk_0".
    """
    safe_source = source.replace(".", "_").replace(" ", "_").replace("/", "_")
    return f"{safe_source}__chunk_{chunk_index}"
