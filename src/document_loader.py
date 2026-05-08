"""
document_loader.py
==================
INGESTION PHASE — Step 1

Responsibilities:
  - Scan the documents directory for supported file types (.txt, .md)
  - Read file contents with UTF-8 encoding
  - Perform basic cleaning:
      * Normalize whitespace (collapse multiple spaces/newlines)
      * Strip non-printable / control characters
  - Return a list of document dicts ready for chunking

Output schema per document:
  {
    "filename": str,   # basename of the source file (e.g. "sample.txt")
    "content":  str,   # cleaned text content
  }
"""

import os
import re
from pathlib import Path
from typing import List, Dict


# File extensions this loader supports
SUPPORTED_EXTENSIONS = {".txt", ".md"}


def load_documents(documents_dir: str) -> List[Dict[str, str]]:
    """
    Scan *documents_dir* and load all .txt / .md files.

    Args:
        documents_dir: Path (string) to the directory containing raw documents.

    Returns:
        List of dicts with keys "filename" and "content".

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If no supported documents are found in the directory.
    """
    doc_path = Path(documents_dir)

    if not doc_path.exists():
        raise FileNotFoundError(
            f"Documents directory not found: '{documents_dir}'\n"
            f"Please create the directory and place .txt or .md files in it."
        )

    documents: List[Dict[str, str]] = []

    for file_path in sorted(doc_path.iterdir()):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            raw_text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback for files with non-UTF-8 characters
            raw_text = file_path.read_text(encoding="latin-1")
            print(f"  [WARN] '{file_path.name}' read with latin-1 fallback encoding.")

        cleaned = _clean_text(raw_text)

        if not cleaned.strip():
            print(f"  [WARN] '{file_path.name}' is empty after cleaning -- skipped.")
            continue

        documents.append({
            "filename": file_path.name,
            "content": cleaned,
        })
        print(f"  [LOAD] '{file_path.name}' -- {len(cleaned)} characters loaded.")

    if not documents:
        raise ValueError(
            f"No supported documents found in '{documents_dir}'.\n"
            f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    return documents


def _clean_text(text: str) -> str:
    """
    Perform basic text cleaning:
      1. Remove non-printable / control characters (except normal whitespace).
      2. Normalize all whitespace sequences (tabs, multiple spaces) to single space.
      3. Normalize multiple consecutive blank lines to at most two newlines.

    Args:
        text: Raw string from file.

    Returns:
        Cleaned string.
    """
    # 1. Remove control characters except newline (\n) and carriage return (\r)
    text = re.sub(r"[^\S\n\r ]+", " ", text)  # replace odd whitespace with space
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)  # strip control chars

    # 2. Collapse multiple spaces on the same line to a single space
    text = re.sub(r"[ \t]+", " ", text)

    # 3. Normalize excessive blank lines (more than 2 consecutive newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 4. Strip leading/trailing whitespace
    text = text.strip()

    return text
