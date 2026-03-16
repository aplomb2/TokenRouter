"""Fallback chain — automatic model switching on failure."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from tokenrouter.models import ModelConfig, get_model
from tokenrouter.providers import create_provider
from tokenrouter.types import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ProviderType,
)

logger = logging.getLogger("tokenrouter.fallback")

MAX_RETRIES = 2
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_retryable_error(error: Exception) -> bool:
    msg = str(error).lower()
    for code in RETRYABLE_STATUS_CODES:
        if str(code) in msg:
            return True
    if "timeout" in msg or "econnreset" in msg or "not configured" in msg:
        return True
    return False


def _resolve_provider(model_config: ModelConfig, keys: dict[str, str]):
    """Resolve provider adapter using user keys."""
    # Map provider types to key names
    provider_key_map: dict[str, list[str]] = {
        "openai": ["openai"],
        "anthropic": ["anthropic"],
        "google": ["google"],
        "deepseek": ["deepseek"],
        "moonshot": ["moonshot"],
        "qwen": ["qwen", "dashscope"],
        "doubao": ["doubao"],
        "zhipu": ["zhipu"],
    }

    possible_names = provider_key_map.get(model_config.provider, [model_config.provider])
    for name in possible_names:
        if name in keys:
            return create_provider(model_config.provider, keys[name])

    raise Exception(f"No API key for provider {model_config.provider} - not configured")


@dataclass
class FallbackResult:
    response: ChatCompletionResponse
    model_used: ModelConfig
    is_fallback: bool
    fallback_from: str | None
    attempts: int


async def chat_with_fallback(
    request: ChatCompletionRequest,
    primary_model: ModelConfig,
    fallback_chain: list[str],
    keys: dict[str, str],
) -> FallbackResult:
    """Non-streaming chat with automatic fallback on failure."""
    models = [primary_model.id] + fallback_chain[:MAX_RETRIES]

    for i, model_id in enumerate(models):
        model_config = get_model(model_id)
        if not model_config:
            continue

        try:
            provider = _resolve_provider(model_config, keys)
            response = await provider.chat(request, model_config.provider_model_id)
            return FallbackResult(
                response=response,
                model_used=model_config,
                is_fallback=i > 0,
                fallback_from=primary_model.id if i > 0 else None,
                attempts=i + 1,
            )
        except Exception as err:
            logger.warning("[fallback] %s failed (attempt %d): %s", model_config.id, i + 1, err)

            if not _is_retryable_error(err) or i == len(models) - 1:
                raise

    raise Exception("All models in fallback chain failed")


async def chat_stream_with_fallback(
    request: ChatCompletionRequest,
    primary_model: ModelConfig,
    fallback_chain: list[str],
    keys: dict[str, str],
) -> AsyncIterator[tuple[ChatCompletionChunk, ModelConfig, bool, str | None]]:
    """Streaming chat with automatic fallback. Yields (chunk, model_used, is_fallback, fallback_from)."""
    models = [primary_model.id] + fallback_chain[:MAX_RETRIES]

    for i, model_id in enumerate(models):
        model_config = get_model(model_id)
        if not model_config:
            continue

        has_yielded = False
        try:
            provider = _resolve_provider(model_config, keys)
            async for chunk in provider.chat_stream(request, model_config.provider_model_id):
                has_yielded = True
                yield (
                    chunk,
                    model_config,
                    i > 0,
                    primary_model.id if i > 0 else None,
                )
            return
        except Exception as err:
            logger.warning("[fallback-stream] %s failed: %s", model_config.id, err)

            if has_yielded:
                raise

            if not _is_retryable_error(err) or i == len(models) - 1:
                raise

    raise Exception("All models in fallback chain failed")
