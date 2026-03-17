"""Tests for fallback chain logic."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from tokenrouter.fallback import (
    _is_retryable_error,
    chat_with_fallback,
)
from tokenrouter.models import get_model
from tokenrouter.types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    Usage,
)


class TestIsRetryableError:
    def test_429_is_retryable(self):
        assert _is_retryable_error(Exception("API error 429: rate limited"))

    def test_500_is_retryable(self):
        assert _is_retryable_error(Exception("API error 500: internal server error"))

    def test_502_is_retryable(self):
        assert _is_retryable_error(Exception("502 Bad Gateway"))

    def test_503_is_retryable(self):
        assert _is_retryable_error(Exception("503 Service Unavailable"))

    def test_504_is_retryable(self):
        assert _is_retryable_error(Exception("504 Gateway Timeout"))

    def test_timeout_is_retryable(self):
        assert _is_retryable_error(Exception("Connection timeout"))

    def test_not_configured_is_retryable(self):
        assert _is_retryable_error(Exception("Provider not configured"))

    def test_400_not_retryable(self):
        assert not _is_retryable_error(Exception("API error 400: bad request"))

    def test_401_not_retryable(self):
        assert not _is_retryable_error(Exception("API error 401: unauthorized"))

    def test_generic_error_not_retryable(self):
        assert not _is_retryable_error(Exception("Something went wrong"))


class TestChatWithFallback:
    @pytest.mark.asyncio
    async def test_primary_succeeds(self):
        model = get_model("gpt-5.2")
        assert model is not None

        mock_response = ChatCompletionResponse(
            id="test-1",
            model="gpt-5.2",
            choices=[
                ChatCompletionChoice(index=0, message={"role": "assistant", "content": "Hello"}, finish_reason="stop")
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        mock_provider = MagicMock()
        mock_provider.chat = AsyncMock(return_value=mock_response)

        with patch("tokenrouter.fallback._resolve_provider", return_value=mock_provider):
            result = await chat_with_fallback(
                ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}]),
                model,
                ["claude-sonnet-4"],
                {"openai": "sk-test"},
            )

        assert result.model_used.id == "gpt-5.2"
        assert not result.is_fallback
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_fallback_on_retryable_error(self):
        model = get_model("gpt-5.2")
        assert model is not None

        mock_response = ChatCompletionResponse(
            id="test-2",
            model="claude-sonnet-4",
            choices=[
                ChatCompletionChoice(index=0, message={"role": "assistant", "content": "Hi"}, finish_reason="stop")
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        call_count = 0

        async def mock_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("API error 429: rate limited")
            return mock_response

        mock_provider = MagicMock()
        mock_provider.chat = mock_chat

        with patch("tokenrouter.fallback._resolve_provider", return_value=mock_provider):
            result = await chat_with_fallback(
                ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}]),
                model,
                ["claude-sonnet-4"],
                {"openai": "sk-test", "anthropic": "sk-ant-test"},
            )

        assert result.is_fallback
        assert result.fallback_from == "gpt-5.2"
        assert result.attempts == 2

    @pytest.mark.asyncio
    async def test_non_retryable_error_raises(self):
        model = get_model("gpt-5.2")
        assert model is not None

        mock_provider = MagicMock()
        mock_provider.chat = AsyncMock(side_effect=Exception("API error 401: unauthorized"))

        with patch("tokenrouter.fallback._resolve_provider", return_value=mock_provider):
            with pytest.raises(Exception, match="401"):
                await chat_with_fallback(
                    ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}]),
                    model,
                    ["claude-sonnet-4"],
                    {"openai": "sk-test"},
                )
