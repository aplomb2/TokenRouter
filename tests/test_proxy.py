"""Tests for the OpenAI-compatible proxy server."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from tokenrouter.config import TokenRouterConfig
from tokenrouter.types import (
    ChatCompletionResponse,
    ChatCompletionChoice,
    Usage,
)

try:
    from fastapi.testclient import TestClient
    from tokenrouter.proxy import create_app
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")


@pytest.fixture
def client():
    config = TokenRouterConfig(keys={"openai": "sk-test"}, strategy="balanced")
    app = create_app(config=config)
    return TestClient(app)


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestModelsEndpoint:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        # Should include OpenAI models since we have openai key
        ids = [m["id"] for m in data["data"]]
        assert "gpt-5.2" in ids


class TestChatCompletions:
    def test_missing_messages(self, client):
        resp = client.post("/v1/chat/completions", json={})
        assert resp.status_code == 400

    def test_invalid_model(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 400

    def test_non_streaming(self, client):
        mock_response = ChatCompletionResponse(
            id="test-1",
            model="gpt-5.2",
            choices=[ChatCompletionChoice(index=0, message={"role": "assistant", "content": "Hello!"}, finish_reason="stop")],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        from tokenrouter.fallback import FallbackResult
        from tokenrouter.models import get_model
        mock_result = FallbackResult(
            response=mock_response,
            model_used=get_model("gpt-5.2"),
            is_fallback=False,
            fallback_from=None,
            attempts=1,
        )

        with patch("tokenrouter.proxy.chat_with_fallback", new_callable=AsyncMock, return_value=mock_result):
            resp = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "Hello"}],
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["choices"][0]["message"]["content"] == "Hello!"
            assert "X-TokenRouter-Model" in resp.headers
