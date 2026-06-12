"""LLM package — primary (Gemini) and fallback (Groq) generation."""

from backend.llm.primary_llm import generate as primary_generate
from backend.llm.fallback_llm import generate as fallback_generate

__all__ = ["primary_generate", "fallback_generate"]
