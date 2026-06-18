"""
Fallback orchestration — try primary LLM, fall back to secondary.

Implements the core resilience pattern: attempt Gemini first, and if it
fails or returns an empty/whitespace-only response, transparently retry
with the Groq fallback model.
"""

import logging
from typing import Dict

from backend.llm import primary_llm, fallback_llm
from backend.config import PRIMARY_MODEL, FALLBACK_MODEL

logger = logging.getLogger(__name__)


def call_with_fallback(
    prompt: str,
    bypass_gemini: bool = False,
    initial_gemini_status: str = "not_attempted",
) -> Dict[str, object]:
    """Generate a response using the primary LLM, falling back if needed.

    Fallback is triggered when the primary model:
      - Raises any exception
      - Returns None
      - Returns an empty or whitespace-only string
      - Or when bypass_gemini is True

    Args:
        prompt: The full prompt to send to the LLM.
        bypass_gemini: If True, skips calling the primary model.
        initial_gemini_status: The previous gemini status in this request (e.g. from routing).

    Returns:
        Dict with keys:
            response (str): The generated text.
            model_used (str): Which model produced the response.
            fallback_triggered (bool): Whether the fallback was used.
            gemini_status (str): The status of Gemini ("success", "rate_limited", "failed", "not_attempted").

    Raises:
        RuntimeError: If both primary and fallback LLMs fail.
    """
    gemini_status = initial_gemini_status

    # ── Try primary (Gemini) ───────────────────────────────────────────
    if not bypass_gemini:
        try:
            response = primary_llm.generate(prompt)
            if response and response.strip():
                return {
                    "response": response,
                    "model_used": PRIMARY_MODEL,
                    "fallback_triggered": False,
                    "gemini_status": "success",
                }
            # Empty/whitespace response — trigger fallback
            print(f"[Fallback] Primary LLM ({PRIMARY_MODEL}) returned empty response — triggering fallback to {FALLBACK_MODEL}")
            logger.warning(
                "Primary LLM (%s) returned empty response — triggering fallback",
                PRIMARY_MODEL,
            )
            gemini_status = "failed"
        except Exception as exc:
            err_msg = str(exc)
            if "429" in err_msg or "ResourceExhausted" in err_msg or "resource_exhausted" in err_msg.lower():
                print(f"[Fallback] Gemini rate limit hit (ResourceExhausted 429). Triggering intentional failover to Groq ({FALLBACK_MODEL}).")
                logger.warning(
                    "[Fallback] Gemini rate limit hit (ResourceExhausted 429). Triggering intentional failover to Groq (%s).",
                    FALLBACK_MODEL,
                )
                gemini_status = "rate_limited"
            else:
                print(f"[Fallback] Primary LLM ({PRIMARY_MODEL}) failed: {exc}. Triggering fallback to Groq ({FALLBACK_MODEL}).")
                logger.warning(
                    "Primary LLM (%s) failed: %s — triggering fallback",
                    PRIMARY_MODEL,
                    exc,
                )
                gemini_status = "failed"
    else:
        print(f"[Fallback] Gemini bypassed per request instruction. Status remains '{gemini_status}'. Going directly to Groq ({FALLBACK_MODEL}).")

    # ── Try fallback (Groq) ────────────────────────────────────────────
    try:
        response = fallback_llm.generate(prompt)
        if response and response.strip():
            return {
                "response": response,
                "model_used": FALLBACK_MODEL,
                "fallback_triggered": True,
                "gemini_status": gemini_status,
            }
        raise RuntimeError(
            f"Fallback LLM ({FALLBACK_MODEL}) returned empty response"
        )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Both primary and fallback LLMs failed. "
            f"Primary: {PRIMARY_MODEL}, Fallback: {FALLBACK_MODEL}. "
            f"Last error: {exc}"
        ) from exc
