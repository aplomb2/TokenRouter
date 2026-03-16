"""OpenAI-compatible provider adapter for Chinese AI providers (DeepSeek, Moonshot, Qwen, etc.)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from tokenrouter.providers.base import parse_sse_stream
from tokenrouter.types import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
)


class OpenAICompatibleAdapter:
    def __init__(self, api_key: str, base_url: str, provider_name: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.provider_name = provider_name

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_body(self, request: ChatCompletionRequest, model_id: str, stream: bool) -> dict:
        body: dict = {
            "model": model_id,
            "messages": request.messages,
            "stream": stream,
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.stop is not None:
            body["stop"] = request.stop
        if request.presence_penalty is not None:
            body["presence_penalty"] = request.presence_penalty
        if request.frequency_penalty is not None:
            body["frequency_penalty"] = request.frequency_penalty
        return body

    async def chat(self, request: ChatCompletionRequest, model_id: str) -> ChatCompletionResponse:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self._build_body(request, model_id, stream=False),
            )
            if resp.status_code != 200:
                raise Exception(f"{self.provider_name} API error {resp.status_code}: {resp.text}")
            return ChatCompletionResponse.from_dict(resp.json())

    async def chat_stream(
        self, request: ChatCompletionRequest, model_id: str
    ) -> AsyncIterator[ChatCompletionChunk]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self._build_body(request, model_id, stream=True),
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise Exception(f"{self.provider_name} API error {resp.status_code}: {body.decode()}")
                async for data in parse_sse_stream(resp):
                    yield ChatCompletionChunk.from_dict(data)
