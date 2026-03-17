"""TokenRouter — Lightweight AI model intelligent routing SDK."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

from tokenrouter.classifier import ClassificationResult, classify_async, classify_sync
from tokenrouter.config import TokenRouterConfig
from tokenrouter.fallback import chat_stream_with_fallback, chat_with_fallback
from tokenrouter.providers import PROVIDER_KEY_MAP
from tokenrouter.models import MODEL_REGISTRY, ModelConfig, calculate_cost, get_model
from tokenrouter.types import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CustomRule,
    RoutingStrategy,
    TokenRouterMetadata,
)

__version__ = "0.2.0"

__all__ = [
    "TokenRouter",
    "__version__",
    "ClassificationResult",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionChunk",
    "TokenRouterMetadata",
    "RoutingStrategy",
    "CustomRule",
    "TokenRouterConfig",
    "ModelConfig",
]

logger = logging.getLogger("tokenrouter")


class TokenRouter:
    """Intelligent AI model router — auto-selects optimal model based on prompt content.

    Usage:
        router = TokenRouter(keys={"openai": "sk-...", "anthropic": "sk-ant-..."})
        response = router.chat([{"role": "user", "content": "Write a sort function"}])
        print(response.model_used)
    """

    def __init__(
        self,
        keys: dict[str, str] | None = None,
        strategy: RoutingStrategy = "balanced",
        rules: list[CustomRule] | list[dict[str, str]] | None = None,
        exclude_models: list[str] | None = None,
        config: TokenRouterConfig | None = None,
    ) -> None:
        if config:
            self._keys = config.keys
            self._strategy = config.strategy
            self._rules = config.rules
            self._exclude_models = config.exclude_models
        else:
            self._keys = keys or {}
            self._strategy = strategy
            self._rules = [
                CustomRule(task=r["task"], model=r["model"]) if isinstance(r, dict) else r for r in (rules or [])
            ]
            self._exclude_models = exclude_models or []

    @classmethod
    def from_config(cls, path: str) -> TokenRouter:
        """Create a TokenRouter from a YAML config file."""
        config = TokenRouterConfig.from_yaml(path)
        return cls(config=config)

    @property
    def available_models(self) -> list[ModelConfig]:
        """Models available given the configured keys."""
        provider_models = []
        for model in MODEL_REGISTRY:
            possible = PROVIDER_KEY_MAP.get(model.provider, [model.provider])
            if any(k in self._keys for k in possible):
                if model.id not in self._exclude_models:
                    provider_models.append(model)
        return provider_models

    # === Async API ===

    async def achat(
        self,
        messages: list[dict[str, Any]],
        strategy: RoutingStrategy | None = None,
        model: str | None = None,
    ) -> ChatCompletionResponse:
        """Async chat — auto-routes to optimal model."""
        start = time.time()
        request_id = str(uuid.uuid4())
        effective_strategy = strategy or self._strategy

        request = ChatCompletionRequest(messages=messages, model=model or "auto")

        if model and model != "auto":
            # Explicit model
            model_config = get_model(model)
            if not model_config:
                raise ValueError(f"Model '{model}' not found. Available: {[m.id for m in MODEL_REGISTRY]}")
            classification = None
            fallback_chain = []
        else:
            classification = await classify_async(
                messages,
                strategy=effective_strategy,
                custom_rules=self._rules or None,
                keys=self._keys,
                exclude_models=self._exclude_models or None,
            )
            model_config = classification.selected_model
            fallback_chain = classification.fallback_chain

        result = await chat_with_fallback(request, model_config, fallback_chain, self._keys)

        latency_ms = int((time.time() - start) * 1000)
        result.response._tokenrouter = TokenRouterMetadata(
            request_id=request_id,
            model_requested=model or "auto",
            model_used=result.model_used.id,
            task_type=classification.task_type if classification else "explicit",
            complexity=classification.complexity if classification else "explicit",
            confidence=classification.confidence if classification else 1.0,
            cost=calculate_cost(
                result.model_used,
                result.response.usage.prompt_tokens,
                result.response.usage.completion_tokens,
            ),
            latency_ms=latency_ms,
            strategy=effective_strategy,
            fallback=result.is_fallback,
        )
        result.response.model = result.model_used.id
        return result.response

    async def achat_stream(
        self,
        messages: list[dict[str, Any]],
        strategy: RoutingStrategy | None = None,
        model: str | None = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Async streaming chat — auto-routes to optimal model."""
        effective_strategy = strategy or self._strategy

        request = ChatCompletionRequest(messages=messages, model=model or "auto", stream=True)

        if model and model != "auto":
            model_config = get_model(model)
            if not model_config:
                raise ValueError(f"Model '{model}' not found.")
            fallback_chain = []
        else:
            classification = await classify_async(
                messages,
                strategy=effective_strategy,
                custom_rules=self._rules or None,
                keys=self._keys,
                exclude_models=self._exclude_models or None,
            )
            model_config = classification.selected_model
            fallback_chain = classification.fallback_chain

        async for chunk, model_used, is_fallback, fallback_from in chat_stream_with_fallback(
            request, model_config, fallback_chain, self._keys
        ):
            chunk.model = model_used.id
            yield chunk

    # === Sync API (wrappers) ===

    def chat(
        self,
        messages: list[dict[str, Any]],
        strategy: RoutingStrategy | None = None,
        model: str | None = None,
    ) -> ChatCompletionResponse:
        """Synchronous chat — auto-routes to optimal model."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.achat(messages, strategy, model)).result()
        return asyncio.run(self.achat(messages, strategy, model))

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        strategy: RoutingStrategy | None = None,
        model: str | None = None,
    ) -> Iterator[ChatCompletionChunk]:
        """Synchronous streaming chat — yields chunks."""

        async def _collect():
            chunks = []
            async for chunk in self.achat_stream(messages, strategy, model):
                chunks.append(chunk)
            return chunks

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                chunks = pool.submit(asyncio.run, _collect()).result()
        else:
            chunks = asyncio.run(_collect())

        yield from chunks

    # === Utility ===

    def classify(self, messages: list[dict[str, Any]]) -> ClassificationResult:
        """Classify a prompt without making an API call (L1 only, sync)."""
        return classify_sync(
            messages,
            strategy=self._strategy,
            custom_rules=self._rules or None,
            exclude_models=self._exclude_models or None,
        )
