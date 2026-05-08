"""
llm_generator.py
================
QUERY PHASE — Step 4

Uses xAI Grok API (OpenAI-compatible) for answer generation.
- Base URL: https://api.x.ai/v1
- Model: grok-3-fast
- Reads XAI_API_KEY from environment
- Same interface as the original OpenAI version
"""

import os
from openai import OpenAI


def generate_answer(
    prompt: str,
    model: str = "grok-3-fast",
    temperature: float = 0.0,
) -> str:
    """
    Send the RAG prompt to the Grok LLM and return the generated answer.

    Args:
        prompt:      The fully constructed prompt from prompt_builder.
        model:       xAI Grok model name.
        temperature: 0.0 = deterministic, factual answers (recommended for RAG).

    Returns:
        The generated answer string.
    """
    client = _get_client()

    print(f"  [LLM] Sending prompt to Grok '{model}'...")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=temperature,
    )

    answer = response.choices[0].message.content.strip()
    print(f"  [LLM] Answer generated ({len(answer)} characters).")
    return answer


def _get_client() -> OpenAI:
    """
    Build and return an OpenAI-compatible client pointing to Groq's API.

    Raises:
        EnvironmentError: If GROQ_API_KEY is not set in .env.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set.\n"
            "Add your Groq API key to .env:\n"
            "  GROQ_API_KEY=gsk_your-key-here\n"
            "Get your free key at: https://console.groq.com"
        )
    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )
