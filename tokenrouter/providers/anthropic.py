"""Anthropic provider adapter — converts between Anthropic and OpenAI formats."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

import httpx

from tokenrouter.exceptions import ProviderError
from tokenrouter.types import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    Usage,
    extract_text,
)


class AnthropicAdapter:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _convert_messages(self, request: ChatCompletionRequest) -> tuple[str | None, list[dict]]:
        """Separate system message and convert to Anthropic format."""
        system_text = None
        messages = []
        for msg in request.messages:
            if msg.get("role") == "system":
                system_text = extract_text(msg.get("content", ""))
            else:
                messages.append(
                    {
                        "role": msg.get("role", "user"),
                        "content": extract_text(msg.get("content", "")),
                    }
                )
        return system_text, messages

    async def chat(self, request: ChatCompletionRequest, model_id: str) -> ChatCompletionResponse:
        system_text, messages = self._convert_messages(request)

        body: dict = {
            "model": model_id,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
        }
        if system_text:
            body["system"] = system_text
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.stop is not None:
            body["stop_sequences"] = request.stop if isinstance(request.stop, list) else [request.stop]

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.base_url}/messages",
                headers=self._headers(),
                json=body,
            )
            if resp.status_code != 200:
                raise ProviderError("Anthropic", resp.status_code, resp.text)

            data = resp.json()

            # Convert Anthropic response to OpenAI format
            content = "".join(c["text"] for c in data.get("content", []) if c.get("type") == "text")
            usage_data = data.get("usage", {})
            stop_reason = data.get("stop_reason", "end_turn")

            return ChatCompletionResponse(
                id=f"chatcmpl-{data.get('id', '')}",
                created=int(time.time()),
                model=data.get("model", model_id),
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message={"role": "assistant", "content": content},
                        finish_reason="stop" if stop_reason == "end_turn" else stop_reason,
                    )
                ],
                usage=Usage(
                    prompt_tokens=usage_data.get("input_tokens", 0),
                    completion_tokens=usage_data.get("output_tokens", 0),
                    total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
                ),
            )

    async def chat_stream(self, request: ChatCompletionRequest, model_id: str) -> AsyncIterator[ChatCompletionChunk]:
        system_text, messages = self._convert_messages(request)

        body: dict = {
            "model": model_id,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
            "stream": True,
        }
        if system_text:
            body["system"] = system_text
        if request.temperature is not None:
            body["temperature"] = request.temperature

        chunk_id = f"chatcmpl-{int(time.time())}"
        created = int(time.time())

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/messages",
                headers=self._headers(),
                json=body,
            ) as resp:
                if resp.status_code != 200:
                    body_bytes = await resp.aread()
                    raise ProviderError("Anthropic", resp.status_code, body_bytes.decode())

                buffer = ""
                async for text in resp.aiter_text():
                    buffer += text
                    lines = buffer.split("\n")
                    buffer = lines.pop()

                    for line in lines:
                        trimmed = line.strip()
                        if not trimmed.startswith("data: "):
                            continue
                        try:
                            event = json.loads(trimmed[6:])
                            if event.get("type") == "content_block_delta" and event.get("delta", {}).get("text"):
                                yield ChatCompletionChunk(
                                    id=chunk_id,
                                    created=created,
                                    model=model_id,
                                    choices=[
                                        {
                                            "index": 0,
                                            "delta": {"content": event["delta"]["text"]},
                                            "finish_reason": None,
                                        }
                                    ],
                                )
                            elif event.get("type") == "message_stop":
                                yield ChatCompletionChunk(
                                    id=chunk_id,
                                    created=created,
                                    model=model_id,
                                    choices=[
                                        {
                                            "index": 0,
                                            "delta": {},
                                            "finish_reason": "stop",
                                        }
                                    ],
                                )
                        except json.JSONDecodeError:
                            pass
