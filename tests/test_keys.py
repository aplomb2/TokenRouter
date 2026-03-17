"""Tests for API key management (keys.py)."""

from __future__ import annotations

import pytest

from tokenrouter.keys import KeyStore, AsyncKeyStore, generate_api_key


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_keys.db")


@pytest.fixture
def store(db_path):
    s = KeyStore(db_path)
    yield s
    s.close()


class TestGenerateApiKey:
    def test_format(self):
        key = generate_api_key()
        assert key.startswith("tr-")
        assert len(key) == 35  # "tr-" + 32 hex chars

    def test_unique(self):
        keys = {generate_api_key() for _ in range(100)}
        assert len(keys) == 100


class TestKeyStore:
    def test_create_key(self, store: KeyStore):
        key = store.create_key(name="test-key", strategy="balanced")
        assert key.id > 0
        assert key.name == "test-key"
        assert key.api_key.startswith("tr-")
        assert key.strategy == "balanced"
        assert key.rate_limit == 0
        assert key.created_at > 0

    def test_get_key(self, store: KeyStore):
        created = store.create_key(name="lookup-test")
        fetched = store.get_key(created.api_key)
        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.name == "lookup-test"

    def test_get_key_not_found(self, store: KeyStore):
        assert store.get_key("tr-nonexistent") is None

    def test_get_key_by_id(self, store: KeyStore):
        created = store.create_key(name="by-id-test")
        fetched = store.get_key_by_id(created.id)
        assert fetched is not None
        assert fetched.api_key == created.api_key

    def test_list_keys(self, store: KeyStore):
        store.create_key(name="key-1")
        store.create_key(name="key-2")
        store.create_key(name="key-3")
        keys = store.list_keys()
        assert len(keys) == 3
        names = [k.name for k in keys]
        assert "key-1" in names
        assert "key-2" in names
        assert "key-3" in names

    def test_delete_key(self, store: KeyStore):
        key = store.create_key(name="to-delete")
        assert store.delete_key(key.id)
        assert store.get_key(key.api_key) is None

    def test_delete_nonexistent(self, store: KeyStore):
        assert not store.delete_key(99999)

    def test_update_key(self, store: KeyStore):
        key = store.create_key(name="original")
        store.update_key(key.id, name="updated", strategy="cheapest")
        fetched = store.get_key_by_id(key.id)
        assert fetched is not None
        assert fetched.name == "updated"
        assert fetched.strategy == "cheapest"

    def test_add_provider(self, store: KeyStore):
        key = store.create_key(name="with-providers")
        ok = store.add_provider(key.id, "openai", "sk-test-123")
        assert ok
        fetched = store.get_key(key.api_key)
        assert fetched is not None
        assert "openai" in fetched.providers
        assert fetched.providers["openai"] == "sk-test-123"

    def test_add_multiple_providers(self, store: KeyStore):
        key = store.create_key(name="multi-prov")
        store.add_provider(key.id, "openai", "sk-openai")
        store.add_provider(key.id, "anthropic", "sk-ant-xxx")
        store.add_provider(key.id, "deepseek", "sk-deep")
        fetched = store.get_key(key.api_key)
        assert fetched is not None
        assert len(fetched.providers) == 3

    def test_remove_provider(self, store: KeyStore):
        key = store.create_key(name="rm-prov")
        store.add_provider(key.id, "openai", "sk-test")
        assert store.remove_provider(key.id, "openai")
        fetched = store.get_key(key.api_key)
        assert fetched is not None
        assert "openai" not in fetched.providers

    def test_remove_nonexistent_provider(self, store: KeyStore):
        key = store.create_key(name="no-prov")
        assert not store.remove_provider(key.id, "nonexistent")

    def test_validate_key(self, store: KeyStore):
        key = store.create_key(name="valid")
        assert store.validate_key(key.api_key) is not None
        assert store.validate_key("tr-invalid") is None
        assert store.validate_key("not-a-tr-key") is None

    def test_cascade_delete(self, store: KeyStore):
        """Deleting a key should also delete its providers."""
        key = store.create_key(name="cascade")
        store.add_provider(key.id, "openai", "sk-test")
        store.delete_key(key.id)
        # Provider should be gone
        rows = store._conn.execute("SELECT COUNT(*) FROM provider_keys WHERE tr_key_id = ?", (key.id,)).fetchone()
        assert rows[0] == 0

    def test_to_dict_masks_keys(self, store: KeyStore):
        key = store.create_key(name="dict-test")
        store.add_provider(key.id, "openai", "sk-very-long-key-here")
        fetched = store.get_key(key.api_key)
        assert fetched is not None
        d = fetched.to_dict(mask_keys=True)
        assert d["providers"]["openai"].endswith("...")
        d_unmasked = fetched.to_dict(mask_keys=False)
        assert d_unmasked["providers"]["openai"] == "sk-very-long-key-here"


class TestAsyncKeyStore:
    @pytest.fixture
    def async_store(self, db_path):
        s = AsyncKeyStore(db_path)
        yield s
        s.close()

    @pytest.mark.asyncio
    async def test_create_and_get(self, async_store: AsyncKeyStore):
        key = await async_store.create_key(name="async-test")
        assert key.api_key.startswith("tr-")
        fetched = await async_store.get_key(key.api_key)
        assert fetched is not None
        assert fetched.name == "async-test"

    @pytest.mark.asyncio
    async def test_list_keys(self, async_store: AsyncKeyStore):
        await async_store.create_key(name="a1")
        await async_store.create_key(name="a2")
        keys = await async_store.list_keys()
        assert len(keys) == 2

    @pytest.mark.asyncio
    async def test_validate(self, async_store: AsyncKeyStore):
        key = await async_store.create_key(name="val")
        assert await async_store.validate_key(key.api_key) is not None
        assert await async_store.validate_key("tr-bad") is None

    @pytest.mark.asyncio
    async def test_log_request(self, async_store: AsyncKeyStore):
        key = await async_store.create_key(name="log-test")
        await async_store.log_request(
            tr_key_id=key.id,
            model_requested="auto",
            model_used="deepseek-chat",
            task_type="coding",
            complexity="medium",
            input_tokens=100,
            output_tokens=200,
            actual_cost=0.001,
            baseline_cost=0.05,
            latency_ms=500,
            success=True,
        )
        # Verify it was logged
        store = async_store._get_store()
        row = store._conn.execute("SELECT COUNT(*) FROM request_logs WHERE tr_key_id = ?", (key.id,)).fetchone()
        assert row[0] == 1
