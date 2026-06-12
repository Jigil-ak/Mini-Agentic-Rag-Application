"""
Fallback LLM — Groq API with llama-3.1-8b-instant via groq SDK.

Serves as the backup when the primary Gemini model is unavailable or
returns an unusable response. All failures raise LLMException
(imported from primary_llm to keep a single exception hierarchy).
"""

import logging

from groq import Groq

from backend.config import GROQ_API_KEY, FALLBACK_MODEL
from backend.llm.primary_llm import LLMException

logger = logging.getLogger(__name__)


# ── Singleton Groq client ─────────────────────────────────────────────────
_client: Groq | None = None


def _get_client() -> Groq:
    """Lazily create and return the Groq client."""
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise LLMException("GROQ_API_KEY is not configured")
        _client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialised for model: %s", FALLBACK_MODEL)
    return _client


# ── Public API ─────────────────────────────────────────────────────────────

def generate(prompt: str) -> str:
    """Generate a response from Groq (llama-3.1-8b-instant).

    Args:
        prompt: The full prompt text to send to the model.

    Returns:
        The generated text response.

    Raises:
        LLMException: On API errors, empty responses, missing API key,
                       or network failures.
    """
    try:
        client = _get_client()

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=FALLBACK_MODEL,
            temperature=0.7,
            max_tokens=2048,
        )

        # Extract response text
        if not chat_completion.choices:
            raise LLMException("Groq returned no choices in response")

        text = chat_completion.choices[0].message.content

        if not text or not text.strip():
            raise LLMException("Groq returned an empty response")

        return text.strip()

    except LLMException:
        # Re-raise our own exceptions as-is
        raise
    except Exception as exc:
        raise LLMException(f"Groq API error: {exc}") from exc
