"""
Primary LLM — Google Gemini 2.0 Flash via google-generativeai SDK.

Configures the API key at module load time and exposes a single
generate() function. All failures are wrapped in LLMException.
"""

import logging

import google.generativeai as genai

from backend.config import GEMINI_API_KEY, PRIMARY_MODEL

logger = logging.getLogger(__name__)


# ── Custom exception ──────────────────────────────────────────────────────

class LLMException(Exception):
    """Raised when LLM generation fails for any reason."""
    pass


# ── Configure Gemini once at import time ──────────────────────────────────

def _configure_gemini() -> None:
    """Set the Gemini API key. Called once at module level."""
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY is not set — primary LLM calls will fail")
        return
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini API configured for model: %s", PRIMARY_MODEL)


_configure_gemini()


# ── Public API ─────────────────────────────────────────────────────────────

def generate(prompt: str) -> str:
    """Generate a response from Gemini 2.0 Flash.

    Args:
        prompt: The full prompt text to send to the model.

    Returns:
        The generated text response.

    Raises:
        LLMException: On API errors, empty responses, content filtering
                       blocks, timeouts, or missing API key.
    """
    if not GEMINI_API_KEY:
        raise LLMException("GEMINI_API_KEY is not configured")

    try:
        model = genai.GenerativeModel(PRIMARY_MODEL)
        response = model.generate_content(prompt)

        # Check for content filtering / safety blocks
        if not response.candidates:
            raise LLMException(
                "Gemini returned no candidates — content may have been filtered"
            )

        candidate = response.candidates[0]

        # Check finish reason for blocks
        if hasattr(candidate, "finish_reason"):
            finish_reason = str(candidate.finish_reason)
            if "SAFETY" in finish_reason.upper() or "BLOCK" in finish_reason.upper():
                raise LLMException(
                    f"Gemini content blocked by safety filter: {finish_reason}"
                )

        # Extract text
        text = response.text
        if not text or not text.strip():
            raise LLMException("Gemini returned an empty response")

        return text.strip()

    except LLMException:
        # Re-raise our own exceptions as-is
        raise
    except Exception as exc:
        raise LLMException(f"Gemini API error: {exc}") from exc
