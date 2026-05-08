"""
embedding_generator.py
======================
INGESTION PHASE — Step 3  |  QUERY PHASE — Step 1

Uses sentence-transformers (all-MiniLM-L6-v2) to generate embeddings locally.
- No API key required
- Runs fully on CPU
- Produces 384-dimensional vectors optimized for semantic similarity
- Same model is used for BOTH ingestion and query (guaranteed consistency)
"""

from typing import List
from sentence_transformers import SentenceTransformer

# Singleton model instance — loaded once, reused across all calls
_model: SentenceTransformer | None = None
_loaded_model_name: str = ""


def get_embeddings(
    texts: List[str],
    model: str = "all-MiniLM-L6-v2",
) -> List[List[float]]:
    """
    Convert a list of text strings into embedding vectors using a local model.

    Args:
        texts:  List of strings to embed.
        model:  sentence-transformers model name (must match between ingest and query).

    Returns:
        List of embedding vectors (each is a list of floats).
    """
    if not texts:
        raise ValueError("Cannot embed an empty list of texts.")

    st_model = _load_model(model)

    print(f"  [EMBED] Encoding {len(texts)} text(s) locally with '{model}'...")
    vectors = st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    return [v.tolist() for v in vectors]


def get_query_embedding(
    query: str,
    model: str = "all-MiniLM-L6-v2",
) -> List[float]:
    """
    Embed a single query string.

    Args:
        query:  The user question string.
        model:  sentence-transformers model name (must match ingestion model).

    Returns:
        Single embedding vector as list of floats.
    """
    vectors = get_embeddings([query], model=model)
    return vectors[0]


def _load_model(model_name: str) -> SentenceTransformer:
    """
    Load and cache the sentence-transformers model.
    Downloads on first use (~90 MB), cached locally afterward.
    """
    global _model, _loaded_model_name

    if _model is None or _loaded_model_name != model_name:
        print(f"  [EMBED] Loading local embedding model '{model_name}' ...")
        _model = SentenceTransformer(model_name)
        _loaded_model_name = model_name
        print(f"  [EMBED] Model ready. Embedding dimension: {_model.get_sentence_embedding_dimension()}")

    return _model
