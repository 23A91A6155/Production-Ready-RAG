"""
prompt_builder.py
=================
QUERY PHASE — Step 3

Responsibilities:
  - Accept retrieved chunks from the retrieval pipeline
  - Format each chunk with its metadata
  - Assemble the exact prompt template specified in the requirements
  - Return the final prompt string ready to be sent to the LLM

Prompt template (exact, from requirements):
  - Instructs LLM to answer ONLY from provided context
  - Instructs LLM to say "I don't know." if answer is not in context
  - Injects context chunks with their source metadata
  - Appends the user's question
"""

from typing import List, Dict, Any


# -----------------------------------------------------------------------
# Prompt template constants
# -----------------------------------------------------------------------

SYSTEM_INSTRUCTIONS = """\
You are a helpful assistant designed to answer questions accurately based on a provided context. Your task is to use ONLY the information in the 'Context' section below.

Do not use any external knowledge you might have. If the answer to the question cannot be found within the provided context, you must respond with the exact phrase: "I don't know.\""""

CONTEXT_HEADER = "Context:\n---"
CONTEXT_FOOTER = "---"
QUESTION_LABEL = "Question:"
ANSWER_LABEL = "Answer:"


def build_prompt(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> str:
    """
    Build the full RAG prompt by combining instructions, formatted context chunks,
    and the user's question.

    Args:
        query:            The user's original question string.
        retrieved_chunks: List of chunk dicts from retrieval_pipeline.
                          Each dict must have keys: "text", "source", "chunk_index".

    Returns:
        Complete prompt string to be sent to the LLM.

    Raises:
        ValueError: If retrieved_chunks is empty.
    """
    if not retrieved_chunks:
        raise ValueError(
            "Cannot build a prompt with no retrieved chunks. "
            "Ensure the retrieval pipeline returned results."
        )

    # Format the context block
    context_lines = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        chunk_label = f"[CHUNK {i}]"
        chunk_text = chunk["text"].strip()
        source_meta = _format_metadata(chunk)

        context_lines.append(f"{chunk_label}: {chunk_text}")
        context_lines.append(f"Source: {source_meta}")
        context_lines.append("")  # blank line between chunks

    # Remove trailing blank line
    if context_lines and context_lines[-1] == "":
        context_lines.pop()

    context_block = "\n".join(context_lines)

    # Assemble the full prompt
    prompt = (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"{CONTEXT_HEADER}\n"
        f"{context_block}\n"
        f"{CONTEXT_FOOTER}\n\n"
        f"{QUESTION_LABEL} {query.strip()}\n\n"
        f"{ANSWER_LABEL}"
    )

    return prompt


def _format_metadata(chunk: Dict[str, Any]) -> str:
    """
    Format chunk metadata as a human-readable source citation string.

    Args:
        chunk: Chunk dict with "source" and "chunk_index" keys.

    Returns:
        Formatted metadata string, e.g.:
        "Document: sample.txt | Chunk Index: 3"
    """
    source = chunk.get("source", "unknown")
    chunk_index = chunk.get("chunk_index", "?")
    return f"Document: {source} | Chunk Index: {chunk_index}"
