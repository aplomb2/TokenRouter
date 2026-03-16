"""Tests for proxy key authentication and management API."""

from __future__ import annotations

import os
import pytest
from unittest.mock import AsyncMock, patch

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

ADMIN_TOKEN = "test-admin-secret-123"


@pytest.fixture
def app_with_auth(tmp_path):
    db_path = str(tmp_path / "test_auth.db")
    config = TokenRouterConfig(
        keys={"openai": "sk-test"},
        strategy="balanced",
        admin_token=ADMIN_TOKEN,
        database=db_path,
    )
    return create_app(config=config)


@pytest.fixture
def client(app_with_auth):
    return TestClient(app_with_auth)


@pytest.fixture
def admin_headers():
    return {"Authorization": f"Bearer {ADMIN_TOKEN}"}


class TestDashboard:
    def test_dashboard_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "TokenRouter" in resp.text


class TestKeyManagement:
    def test_create_key(self, client, admin_headers):
        resp = client.post("/v1/keys", json={"name": "test-key", "strategy": "balanced"}, headers=admin_headers)
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "test-key"
        assert data["api_key"].startswith("tr-")
        assert data["strategy"] == "balanced"

    def test_create_key_no_auth(self, client):
        resp = client.post("/v1/keys", json={"name": "test"})
        assert resp.status_code == 401

    def test_list_keys(self, client, admin_headers):
        client.post("/v1/keys", json={"name": "k1"}, headers=admin_headers)
        client.post("/v1/keys", json={"name": "k2"}, headers=admin_headers)
        resp = client.get("/v1/keys", headers=admin_headers)
        assert resp.status_code == 200
        assert len(resp.json()["keys"]) == 2

    def test_list_keys_no_auth(self, client):
        resp = client.get("/v1/keys")
        assert resp.status_code == 401

    def test_delete_key(self, client, admin_headers):
        resp = client.post("/v1/keys", json={"name": "to-delete"}, headers=admin_headers)
        key_id = resp.json()["id"]
        resp = client.delete(f"/v1/keys/{key_id}", headers=admin_headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_delete_nonexistent(self, client, admin_headers):
        resp = client.delete("/v1/keys/99999", headers=admin_headers)
        assert resp.status_code == 404

    def test_add_provider(self, client, admin_headers):
        resp = client.post("/v1/keys", json={"name": "prov-test"}, headers=admin_headers)
        key_id = resp.json()["id"]
        resp = client.post(
            f"/v1/keys/{key_id}/providers",
            json={"provider": "openai", "api_key": "sk-test-key"},
            headers=admin_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["status"] == "added"

    def test_add_provider_missing_fields(self, client, admin_headers):
        resp = client.post("/v1/keys", json={"name": "prov-test2"}, headers=admin_headers)
        key_id = resp.json()["id"]
        resp = client.post(
            f"/v1/keys/{key_id}/providers",
            json={"provider": "openai"},
            headers=admin_headers,
        )
        assert resp.status_code == 400

    def test_remove_provider(self, client, admin_headers):
        resp = client.post("/v1/keys", json={"name": "rm-prov"}, headers=admin_headers)
        key_id = resp.json()["id"]
        client.post(
            f"/v1/keys/{key_id}/providers",
            json={"provider": "anthropic", "api_key": "sk-ant-xxx"},
            headers=admin_headers,
        )
        resp = client.delete(f"/v1/keys/{key_id}/providers/anthropic", headers=admin_headers)
        assert resp.status_code == 200

    def test_remove_nonexistent_provider(self, client, admin_headers):
        resp = client.post("/v1/keys", json={"name": "no-prov"}, headers=admin_headers)
        key_id = resp.json()["id"]
        resp = client.delete(f"/v1/keys/{key_id}/providers/nonexistent", headers=admin_headers)
        assert resp.status_code == 404


class TestAuthenticatedChat:
    def test_tr_key_auth(self, client, admin_headers):
        """Create a tr-key with providers, then use it to make a chat request."""
        # Create key
        resp = client.post("/v1/keys", json={"name": "chat-key"}, headers=admin_headers)
        tr_key = resp.json()["api_key"]
        key_id = resp.json()["id"]

        # Add provider
        client.post(
            f"/v1/keys/{key_id}/providers",
            json={"provider": "openai", "api_key": "sk-real-key"},
            headers=admin_headers,
        )

        # Mock the chat call
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
            resp = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}], "model": "gpt-5.2"},
                headers={"Authorization": f"Bearer {tr_key}"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["choices"][0]["message"]["content"] == "Hello!"
            # Should have cost headers
            assert "X-TokenRouter-Cost" in resp.headers
            assert "X-TokenRouter-Saved" in resp.headers

    def test_invalid_tr_key(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer tr-invalidkey12345678901234567890"},
        )
        assert resp.status_code == 401

    def test_legacy_mode_no_auth(self):
        """Legacy mode (keys in config, no admin_token) should work without auth."""
        config = TokenRouterConfig(keys={"openai": "sk-test"}, strategy="balanced")
        app = create_app(config=config)
        client = TestClient(app)

        mock_response = ChatCompletionResponse(
            id="test-1",
            model="gpt-5.2",
            choices=[ChatCompletionChoice(index=0, message={"role": "assistant", "content": "Hi"}, finish_reason="stop")],
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
            resp = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}], "model": "gpt-5.2"},
            )
            assert resp.status_code == 200


class TestUsageAPI:
    def test_get_usage(self, client, admin_headers):
        # Create a key first
        resp = client.post("/v1/keys", json={"name": "usage-key"}, headers=admin_headers)
        key_id = resp.json()["id"]
        resp = client.get(f"/v1/usage/{key_id}", headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "total_requests" in data
        assert "total_cost" in data

    def test_get_savings(self, client, admin_headers):
        resp = client.post("/v1/keys", json={"name": "savings-key"}, headers=admin_headers)
        key_id = resp.json()["id"]
        resp = client.get(f"/v1/usage/{key_id}/savings", headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "saved" in data
        assert "saved_pct" in data

    def test_usage_no_auth(self, client):
        resp = client.get("/v1/usage/1")
        assert resp.status_code == 401


class TestDashboardAPI:
    def test_dashboard_stats(self, client, admin_headers):
        resp = client.get("/v1/dashboard/stats", headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "total_requests" in data

    def test_dashboard_daily(self, client, admin_headers):
        resp = client.get("/v1/dashboard/daily", headers=admin_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_dashboard_models(self, client, admin_headers):
        resp = client.get("/v1/dashboard/models", headers=admin_headers)
        assert resp.status_code == 200

    def test_dashboard_logs(self, client, admin_headers):
        resp = client.get("/v1/dashboard/logs", headers=admin_headers)
        assert resp.status_code == 200

    def test_dashboard_no_auth(self, client):
        resp = client.get("/v1/dashboard/stats")
        assert resp.status_code == 401
