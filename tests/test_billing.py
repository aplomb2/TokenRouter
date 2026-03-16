"""Tests for billing engine (billing.py)."""

from __future__ import annotations

import time
import pytest

from tokenrouter.billing import (
    BillingEngine,
    calculate_baseline_cost,
    DailyCost,
    SavingsReport,
    UsageSummary,
)
from tokenrouter.keys import KeyStore


@pytest.fixture
def store(tmp_path):
    s = KeyStore(str(tmp_path / "test_billing.db"))
    yield s
    s.close()


@pytest.fixture
def billing(store: KeyStore) -> BillingEngine:
    return BillingEngine(store._conn)


@pytest.fixture
def key_with_logs(store: KeyStore, billing: BillingEngine):
    key = store.create_key(name="billing-test")
    now = time.time()
    # Insert some test logs
    for i in range(5):
        store._conn.execute(
            """INSERT INTO request_logs
               (tr_key_id, timestamp, model_requested, model_used, task_type, complexity,
                input_tokens, output_tokens, actual_cost, baseline_cost, saved, latency_ms, success)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (key.id, now - i * 3600, "auto", "deepseek-chat", "coding", "medium",
             1000, 500, 0.001, 0.05, 0.049, 200 + i * 10, 1),
        )
    # Add one failed request
    store._conn.execute(
        """INSERT INTO request_logs
           (tr_key_id, timestamp, model_requested, model_used, task_type, complexity,
            input_tokens, output_tokens, actual_cost, baseline_cost, saved, latency_ms, success)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (key.id, now - 100, "auto", "gpt-5.2", "coding", "high",
         2000, 1000, 0.02, 0.12, 0.10, 500, 0),
    )
    store._conn.commit()
    return key


class TestCalculateBaselineCost:
    def test_default_baseline(self):
        cost = calculate_baseline_cost(1000, 500)
        # claude-opus-4: 15/1M input + 75/1M output
        expected = (1000 / 1_000_000) * 15 + (500 / 1_000_000) * 75
        assert abs(cost - expected) < 1e-8

    def test_unknown_baseline(self):
        cost = calculate_baseline_cost(1000, 500, "nonexistent-model")
        assert cost == 0.0


class TestBillingEngine:
    def test_usage_summary_all(self, billing: BillingEngine, key_with_logs):
        summary = billing.get_usage_summary(key_with_logs.id, "all")
        assert summary.total_requests == 6  # 5 success + 1 fail
        assert summary.successful_requests == 5
        assert summary.total_input_tokens == 7000  # 5*1000 + 2000
        assert summary.total_output_tokens == 3500  # 5*500 + 1000
        assert summary.total_cost > 0
        assert summary.total_baseline_cost > 0
        assert summary.total_saved > 0
        assert summary.saved_pct > 0

    def test_usage_summary_today(self, billing: BillingEngine, key_with_logs):
        summary = billing.get_usage_summary(key_with_logs.id, "today")
        # All logs are within today (within last few hours)
        assert summary.total_requests > 0

    def test_usage_summary_empty_key(self, billing: BillingEngine):
        summary = billing.get_usage_summary(99999, "all")
        assert summary.total_requests == 0
        assert summary.total_cost == 0

    def test_models_used(self, billing: BillingEngine, key_with_logs):
        summary = billing.get_usage_summary(key_with_logs.id, "all")
        assert "deepseek-chat" in summary.models_used
        assert "gpt-5.2" in summary.models_used
        assert summary.models_used["deepseek-chat"] == 5

    def test_savings_report(self, billing: BillingEngine, key_with_logs):
        report = billing.get_savings_report(key_with_logs.id)
        assert report.request_count == 6
        assert report.total_cost > 0
        assert report.baseline_cost > report.total_cost
        assert report.saved > 0
        assert report.saved_pct > 0

    def test_savings_report_empty(self, billing: BillingEngine):
        report = billing.get_savings_report(99999)
        assert report.request_count == 0
        assert report.total_cost == 0
        assert report.saved_pct == 0

    def test_daily_costs(self, billing: BillingEngine, key_with_logs):
        costs = billing.get_daily_costs(key_with_logs.id, days=7)
        assert len(costs) >= 1
        for c in costs:
            assert isinstance(c, DailyCost)
            assert c.actual_cost >= 0

    def test_recent_logs(self, billing: BillingEngine, key_with_logs):
        logs = billing.get_recent_logs(key_with_logs.id, limit=10)
        assert len(logs) == 6
        # Most recent first
        assert logs[0]["timestamp"] >= logs[-1]["timestamp"]
        # Check structure
        assert "model_used" in logs[0]
        assert "actual_cost" in logs[0]
        assert "saved" in logs[0]

    def test_model_distribution(self, billing: BillingEngine, key_with_logs):
        dist = billing.get_model_distribution(key_with_logs.id)
        assert len(dist) == 2
        models = {d["model"] for d in dist}
        assert "deepseek-chat" in models

    def test_global_stats(self, billing: BillingEngine, key_with_logs):
        stats = billing.get_global_stats()
        assert stats["total_requests"] == 6
        assert stats["total_cost"] > 0

    def test_to_dict(self, billing: BillingEngine, key_with_logs):
        summary = billing.get_usage_summary(key_with_logs.id, "all")
        d = summary.to_dict()
        assert "total_requests" in d
        assert "saved_pct" in d
        assert isinstance(d["total_cost"], float)

        report = billing.get_savings_report(key_with_logs.id)
        d2 = report.to_dict()
        assert "saved" in d2
        assert "request_count" in d2
