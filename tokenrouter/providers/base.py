"""Base utilities for provider adapters."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx


async def parse_sse_stream(response: httpx.Response) -> AsyncIterator[dict]:
    """Parse Server-Sent Events from an httpx streaming response."""
    buffer = ""
    async for chunk in response.aiter_text():
        buffer += chunk
        lines = buffer.split("\n")
        buffer = lines.pop()

        for line in lines:
            trimmed = line.strip()
            if trimmed == "" or trimmed == "data: [DONE]":
                continue
            if trimmed.startswith("data: "):
                try:
                    yield json.loads(trimmed[6:])
                except json.JSONDecodeError:
                    pass
