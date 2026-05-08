"""
text_chunker.py
===============
INGESTION PHASE — Step 2

Responsibilities:
  - Split cleaned document text into fixed-size, overlapping chunks
  - Chunk size and overlap are measured in TOKENS (via tiktoken)
  - Maintain chunk metadata:
      * source: filename of the originating document
      * chunk_index: zero-based position of this chunk within the document

Chunking strategy: Fixed-size with overlap
  - Tokenize the full document text
  - Slide a window of size=chunk_size across the token list
  - Advance the window by (chunk_size - chunk_overlap) tokens each step
  - Decode each window back to a text string

Output schema per chunk:
  {
    "text":        str,   # Decoded chunk text
    "source":      str,   # Source filename
    "chunk_index": int,   # 0-based position of this chunk in its document
  }
"""

import tiktoken
from typing import List, Dict


def chunk_documents(
    documents: List[Dict[str, str]],
    chunk_size: int = 400,
    chunk_overlap: int = 80,
    encoding_name: str = "cl100k_base",
) -> List[Dict]:
    """
    Split each document into fixed-size, overlapping token chunks.

    Args:
        documents:      List of document dicts from document_loader
                        (each has "filename" and "content" keys).
        chunk_size:     Target number of tokens per chunk (300–500).
        chunk_overlap:  Number of overlapping tokens between consecutive chunks.
        encoding_name:  tiktoken encoding to use (cl100k_base matches text-embedding-3-small).

    Returns:
        Flat list of chunk dicts across all documents.

    Raises:
        ValueError: If chunk_overlap >= chunk_size (overlap must be smaller).
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})."
        )

    enc = tiktoken.get_encoding(encoding_name)
    stride = chunk_size - chunk_overlap  # how many tokens we advance each step

    all_chunks: List[Dict] = []

    for doc in documents:
        filename = doc["filename"]
        content = doc["content"]

        # Tokenize the entire document
        token_ids: List[int] = enc.encode(content)
        total_tokens = len(token_ids)

        if total_tokens == 0:
            print(f"  [WARN] '{filename}' produced 0 tokens -- skipped.")
            continue

        doc_chunks = _create_chunks(
            token_ids=token_ids,
            chunk_size=chunk_size,
            stride=stride,
            source=filename,
            enc=enc,
        )

        all_chunks.extend(doc_chunks)
        print(
            f"  [CHUNK] '{filename}' -- {total_tokens} tokens -> "
            f"{len(doc_chunks)} chunks "
            f"(size={chunk_size}, overlap={chunk_overlap})"
        )

    return all_chunks


def _create_chunks(
    token_ids: List[int],
    chunk_size: int,
    stride: int,
    source: str,
    enc: tiktoken.Encoding,
) -> List[Dict]:
    """
    Slide a fixed window across *token_ids* and decode each window to text.

    Args:
        token_ids:  Full list of token IDs for one document.
        chunk_size: Number of tokens per chunk.
        stride:     Number of tokens to advance between chunks.
        source:     Source filename for metadata.
        enc:        tiktoken encoding object for decoding.

    Returns:
        List of chunk dicts for this document.
    """
    chunks: List[Dict] = []
    start = 0
    chunk_index = 0

    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunk_token_ids = token_ids[start:end]

        chunk_text = enc.decode(chunk_token_ids).strip()

        if chunk_text:  # skip empty chunks
            chunks.append({
                "text": chunk_text,
                "source": source,
                "chunk_index": chunk_index,
            })
            chunk_index += 1

        # If we've reached the end of the document, stop
        if end == len(token_ids):
            break

        start += stride

    return chunks
