"""
Tests for the fallback orchestration logic.

Mocks both primary and fallback LLM generate() functions to test
the retry behaviour without making real API calls.
"""

import pytest
from unittest.mock import patch

from backend.agent.fallback import call_with_fallback
from backend.llm.primary_llm import LLMException


class TestPrimarySuccess:
    """Test cases where the primary LLM succeeds."""

    @patch("backend.agent.fallback.fallback_llm")
    @patch("backend.agent.fallback.primary_llm")
    def test_primary_succeeds_no_fallback(self, mock_primary, mock_fallback):
        mock_primary.generate.return_value = "Primary response"

        result = call_with_fallback("test prompt")

        assert result["response"] == "Primary response"
        assert result["fallback_triggered"] is False
        mock_fallback.generate.assert_not_called()


class TestFallbackTriggered:
    """Test cases where the primary fails and fallback is used."""

    @patch("backend.agent.fallback.fallback_llm")
    @patch("backend.agent.fallback.primary_llm")
    def test_primary_exception_triggers_fallback(self, mock_primary, mock_fallback):
        mock_primary.generate.side_effect = LLMException("API error")
        mock_fallback.generate.return_value = "Fallback response"

        result = call_with_fallback("test prompt")

        assert result["response"] == "Fallback response"
        assert result["fallback_triggered"] is True

    @patch("backend.agent.fallback.fallback_llm")
    @patch("backend.agent.fallback.primary_llm")
    def test_primary_empty_string_triggers_fallback(self, mock_primary, mock_fallback):
        mock_primary.generate.return_value = ""
        mock_fallback.generate.return_value = "Fallback response"

        result = call_with_fallback("test prompt")

        assert result["response"] == "Fallback response"
        assert result["fallback_triggered"] is True

    @patch("backend.agent.fallback.fallback_llm")
    @patch("backend.agent.fallback.primary_llm")
    def test_primary_whitespace_triggers_fallback(self, mock_primary, mock_fallback):
        mock_primary.generate.return_value = "   "
        mock_fallback.generate.return_value = "Fallback response"

        result = call_with_fallback("test prompt")

        assert result["response"] == "Fallback response"
        assert result["fallback_triggered"] is True

    @patch("backend.agent.fallback.fallback_llm")
    @patch("backend.agent.fallback.primary_llm")
    def test_primary_none_triggers_fallback(self, mock_primary, mock_fallback):
        mock_primary.generate.return_value = None
        mock_fallback.generate.return_value = "Fallback response"

        result = call_with_fallback("test prompt")

        assert result["response"] == "Fallback response"
        assert result["fallback_triggered"] is True

    @patch("backend.agent.fallback.fallback_llm")
    @patch("backend.agent.fallback.primary_llm")
    def test_primary_fails_fallback_succeeds_returns_fallback_response(
        self, mock_primary, mock_fallback
    ):
        mock_primary.generate.side_effect = RuntimeError("Network timeout")
        mock_fallback.generate.return_value = "Groq saved the day"

        result = call_with_fallback("test prompt")

        assert result["response"] == "Groq saved the day"
        assert result["fallback_triggered"] is True
        mock_primary.generate.assert_called_once_with("test prompt")
        mock_fallback.generate.assert_called_once_with("test prompt")


class TestBothFail:
    """Test cases where both primary and fallback fail."""

    @patch("backend.agent.fallback.fallback_llm")
    @patch("backend.agent.fallback.primary_llm")
    def test_both_fail_raises_runtime_error(self, mock_primary, mock_fallback):
        mock_primary.generate.side_effect = LLMException("Gemini down")
        mock_fallback.generate.side_effect = LLMException("Groq down")

        with pytest.raises(RuntimeError, match="Both primary and fallback LLMs failed"):
            call_with_fallback("test prompt")

    @patch("backend.agent.fallback.fallback_llm")
    @patch("backend.agent.fallback.primary_llm")
    def test_primary_fails_fallback_empty_raises_runtime_error(
        self, mock_primary, mock_fallback
    ):
        mock_primary.generate.side_effect = LLMException("Gemini error")
        mock_fallback.generate.return_value = ""

        with pytest.raises(RuntimeError):
            call_with_fallback("test prompt")


class TestModelUsedReporting:
    """Test that the correct model name is reported in results."""

    @patch("backend.agent.fallback.PRIMARY_MODEL", "gemini-2.0-flash")
    @patch("backend.agent.fallback.FALLBACK_MODEL", "llama-3.1-8b-instant")
    @patch("backend.agent.fallback.fallback_llm")
    @patch("backend.agent.fallback.primary_llm")
    def test_primary_reports_correct_model(self, mock_primary, mock_fallback):
        mock_primary.generate.return_value = "Response"

        result = call_with_fallback("test")
        assert result["model_used"] == "gemini-2.0-flash"

    @patch("backend.agent.fallback.PRIMARY_MODEL", "gemini-2.0-flash")
    @patch("backend.agent.fallback.FALLBACK_MODEL", "llama-3.1-8b-instant")
    @patch("backend.agent.fallback.fallback_llm")
    @patch("backend.agent.fallback.primary_llm")
    def test_fallback_reports_correct_model(self, mock_primary, mock_fallback):
        mock_primary.generate.side_effect = LLMException("down")
        mock_fallback.generate.return_value = "Fallback"

        result = call_with_fallback("test")
        assert result["model_used"] == "llama-3.1-8b-instant"
