"""Google Gemini provider adapter."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

import httpx

from tokenrouter.types import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    Usage,
    extract_text,
)


class GoogleAdapter:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    def _convert_messages(self, request: ChatCompletionRequest) -> tuple[str, list[dict]]:
        """Convert messages to Google Gemini format."""
        system_instruction = "\n".join(
            extract_text(m.get("content", ""))
            for m in request.messages
            if m.get("role") == "system"
        )
        contents = [
            {
                "role": "model" if m.get("role") == "assistant" else "user",
                "parts": [{"text": extract_text(m.get("content", ""))}],
            }
            for m in request.messages
            if m.get("role") != "system"
        ]
        return system_instruction, contents

    async def chat(self, request: ChatCompletionRequest, model_id: str) -> ChatCompletionResponse:
        system_instruction, contents = self._convert_messages(request)

        body: dict = {
            "contents": contents,
            "generationConfig": {},
        }
        gen_config = body["generationConfig"]
        if request.temperature is not None:
            gen_config["temperature"] = request.temperature
        if request.top_p is not None:
            gen_config["topP"] = request.top_p
        gen_config["maxOutputTokens"] = request.max_tokens or 4096
        if request.stop is not None:
            gen_config["stopSequences"] = request.stop if isinstance(request.stop, list) else [request.stop]

        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        url = f"{self.base_url}/models/{model_id}:generateContent?key={self.api_key}"

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=body, headers={"Content-Type": "application/json"})
            if resp.status_code != 200:
                raise Exception(f"Google API error {resp.status_code}: {resp.text}")

            data = resp.json()
            candidate = (data.get("candidates") or [{}])[0]
            text = "".join(
                p.get("text", "") for p in (candidate.get("content", {}).get("parts") or [])
            )
            usage_meta = data.get("usageMetadata", {})

            return ChatCompletionResponse(
                id=f"chatcmpl-{int(time.time())}",
                created=int(time.time()),
                model=model_id,
                choices=[
                    ChatCompletionChoice(index=0, message={"role": "assistant", "content": text}, finish_reason="stop")
                ],
                usage=Usage(
                    prompt_tokens=usage_meta.get("promptTokenCount", 0),
                    completion_tokens=usage_meta.get("candidatesTokenCount", 0),
                    total_tokens=usage_meta.get("totalTokenCount", 0),
                ),
            )

    async def chat_stream(
        self, request: ChatCompletionRequest, model_id: str
    ) -> AsyncIterator[ChatCompletionChunk]:
        system_instruction, contents = self._convert_messages(request)

        body: dict = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": request.max_tokens or 4096},
        }
        if request.temperature is not None:
            body["generationConfig"]["temperature"] = request.temperature
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        url = f"{self.base_url}/models/{model_id}:streamGenerateContent?alt=sse&key={self.api_key}"

        chunk_id = f"chatcmpl-{int(time.time())}"
        created = int(time.time())

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", url, json=body, headers={"Content-Type": "application/json"}
            ) as resp:
                if resp.status_code != 200:
                    body_bytes = await resp.aread()
                    raise Exception(f"Google API error {resp.status_code}: {body_bytes.decode()}")

                buffer = ""
                async for text_chunk in resp.aiter_text():
                    buffer += text_chunk
                    lines = buffer.split("\n")
                    buffer = lines.pop()

                    for line in lines:
                        trimmed = line.strip()
                        if not trimmed.startswith("data: "):
                            continue
                        try:
                            event = json.loads(trimmed[6:])
                            text = (
                                event.get("candidates", [{}])[0]
                                .get("content", {})
                                .get("parts", [{}])[0]
                                .get("text")
                            )
                            if text:
                                yield ChatCompletionChunk(
                                    id=chunk_id,
                                    created=created,
                                    model=model_id,
                                    choices=[{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                                )
                        except (json.JSONDecodeError, IndexError):
                            pass

        # Final stop chunk
        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_id,
            choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
        )
