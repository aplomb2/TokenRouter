"""Unified API key management with SQLite storage."""

from __future__ import annotations

import os
import secrets
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = os.path.join(Path.home(), ".tokenrouter", "keys.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tr_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    api_key TEXT NOT NULL UNIQUE,
    strategy TEXT NOT NULL DEFAULT 'balanced',
    rate_limit INTEGER DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS provider_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tr_key_id INTEGER NOT NULL,
    provider TEXT NOT NULL,
    api_key TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (tr_key_id) REFERENCES tr_keys(id) ON DELETE CASCADE,
    UNIQUE(tr_key_id, provider)
);

CREATE TABLE IF NOT EXISTS request_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tr_key_id INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    model_requested TEXT NOT NULL DEFAULT '',
    model_used TEXT NOT NULL DEFAULT '',
    task_type TEXT NOT NULL DEFAULT '',
    complexity TEXT NOT NULL DEFAULT '',
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    actual_cost REAL NOT NULL DEFAULT 0,
    baseline_cost REAL NOT NULL DEFAULT 0,
    saved REAL NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    success INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (tr_key_id) REFERENCES tr_keys(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_request_logs_tr_key_id ON request_logs(tr_key_id);
CREATE INDEX IF NOT EXISTS idx_request_logs_timestamp ON request_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_tr_keys_api_key ON tr_keys(api_key);
"""


def generate_api_key() -> str:
    """Generate a `tr-` prefixed API key (tr- + 32 hex chars)."""
    return "tr-" + secrets.token_hex(16)


@dataclass
class TRKey:
    id: int
    name: str
    api_key: str
    strategy: str
    rate_limit: int
    created_at: float
    providers: dict[str, str] = field(default_factory=dict)  # provider -> api_key

    def to_dict(self, mask_keys: bool = True) -> dict[str, Any]:
        providers = {}
        for p, k in self.providers.items():
            providers[p] = k[:8] + "..." if mask_keys and len(k) > 8 else k
        return {
            "id": self.id,
            "name": self.name,
            "api_key": self.api_key,
            "strategy": self.strategy,
            "rate_limit": self.rate_limit,
            "created_at": self.created_at,
            "providers": providers,
        }


class KeyStore:
    """Synchronous SQLite key store."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def create_key(
        self,
        name: str,
        strategy: str = "balanced",
        rate_limit: int = 0,
    ) -> TRKey:
        api_key = generate_api_key()
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO tr_keys (name, api_key, strategy, rate_limit, created_at) VALUES (?, ?, ?, ?, ?)",
            (name, api_key, strategy, rate_limit, now),
        )
        self._conn.commit()
        return TRKey(
            id=cur.lastrowid or 0,
            name=name,
            api_key=api_key,
            strategy=strategy,
            rate_limit=rate_limit,
            created_at=now,
        )

    def get_key(self, api_key: str) -> TRKey | None:
        """Look up a key by its `tr-xxx` value."""
        row = self._conn.execute(
            "SELECT id, name, api_key, strategy, rate_limit, created_at FROM tr_keys WHERE api_key = ?",
            (api_key,),
        ).fetchone()
        if not row:
            return None
        tr_key = TRKey(id=row[0], name=row[1], api_key=row[2], strategy=row[3], rate_limit=row[4], created_at=row[5])
        tr_key.providers = self._load_providers(tr_key.id)
        return tr_key

    def get_key_by_id(self, key_id: int) -> TRKey | None:
        row = self._conn.execute(
            "SELECT id, name, api_key, strategy, rate_limit, created_at FROM tr_keys WHERE id = ?",
            (key_id,),
        ).fetchone()
        if not row:
            return None
        tr_key = TRKey(id=row[0], name=row[1], api_key=row[2], strategy=row[3], rate_limit=row[4], created_at=row[5])
        tr_key.providers = self._load_providers(tr_key.id)
        return tr_key

    def list_keys(self) -> list[TRKey]:
        rows = self._conn.execute(
            "SELECT id, name, api_key, strategy, rate_limit, created_at FROM tr_keys ORDER BY created_at DESC"
        ).fetchall()
        keys: list[TRKey] = []
        for row in rows:
            tr_key = TRKey(
                id=row[0], name=row[1], api_key=row[2], strategy=row[3], rate_limit=row[4], created_at=row[5]
            )
            tr_key.providers = self._load_providers(tr_key.id)
            keys.append(tr_key)
        return keys

    def delete_key(self, key_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM tr_keys WHERE id = ?", (key_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def update_key(
        self, key_id: int, name: str | None = None, strategy: str | None = None, rate_limit: int | None = None
    ) -> bool:
        updates: list[str] = []
        params: list[Any] = []
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if strategy is not None:
            updates.append("strategy = ?")
            params.append(strategy)
        if rate_limit is not None:
            updates.append("rate_limit = ?")
            params.append(rate_limit)
        if not updates:
            return False
        params.append(key_id)
        cur = self._conn.execute(f"UPDATE tr_keys SET {', '.join(updates)} WHERE id = ?", params)
        self._conn.commit()
        return cur.rowcount > 0

    def add_provider(self, key_id: int, provider: str, api_key: str) -> bool:
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO provider_keys (tr_key_id, provider, api_key, active) VALUES (?, ?, ?, 1)",
                (key_id, provider, api_key),
            )
            self._conn.commit()
            return True
        except sqlite3.Error:
            return False

    def remove_provider(self, key_id: int, provider: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM provider_keys WHERE tr_key_id = ? AND provider = ?",
            (key_id, provider),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def _load_providers(self, key_id: int) -> dict[str, str]:
        rows = self._conn.execute(
            "SELECT provider, api_key FROM provider_keys WHERE tr_key_id = ? AND active = 1",
            (key_id,),
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def validate_key(self, api_key: str) -> TRKey | None:
        """Validate a tr-xxx key and return it with providers loaded."""
        if not api_key.startswith("tr-"):
            return None
        return self.get_key(api_key)


class AsyncKeyStore:
    """Async wrapper around KeyStore using aiosqlite."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self._store: KeyStore | None = None

    def _get_store(self) -> KeyStore:
        if self._store is None:
            self._store = KeyStore(self.db_path)
        return self._store

    async def create_key(self, name: str, strategy: str = "balanced", rate_limit: int = 0) -> TRKey:
        return self._get_store().create_key(name, strategy, rate_limit)

    async def get_key(self, api_key: str) -> TRKey | None:
        return self._get_store().get_key(api_key)

    async def get_key_by_id(self, key_id: int) -> TRKey | None:
        return self._get_store().get_key_by_id(key_id)

    async def list_keys(self) -> list[TRKey]:
        return self._get_store().list_keys()

    async def delete_key(self, key_id: int) -> bool:
        return self._get_store().delete_key(key_id)

    async def update_key(self, key_id: int, **kwargs: Any) -> bool:
        return self._get_store().update_key(key_id, **kwargs)

    async def add_provider(self, key_id: int, provider: str, api_key: str) -> bool:
        return self._get_store().add_provider(key_id, provider, api_key)

    async def remove_provider(self, key_id: int, provider: str) -> bool:
        return self._get_store().remove_provider(key_id, provider)

    async def validate_key(self, api_key: str) -> TRKey | None:
        return self._get_store().validate_key(api_key)

    async def log_request(
        self,
        tr_key_id: int,
        model_requested: str,
        model_used: str,
        task_type: str,
        complexity: str,
        input_tokens: int,
        output_tokens: int,
        actual_cost: float,
        baseline_cost: float,
        latency_ms: int,
        success: bool,
    ) -> None:
        store = self._get_store()
        saved = baseline_cost - actual_cost
        store._conn.execute(
            """INSERT INTO request_logs
               (tr_key_id, timestamp, model_requested, model_used, task_type, complexity,
                input_tokens, output_tokens, actual_cost, baseline_cost, saved, latency_ms, success)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                tr_key_id,
                time.time(),
                model_requested,
                model_used,
                task_type,
                complexity,
                input_tokens,
                output_tokens,
                actual_cost,
                baseline_cost,
                saved,
                latency_ms,
                1 if success else 0,
            ),
        )
        store._conn.commit()

    def close(self) -> None:
        if self._store:
            self._store.close()
