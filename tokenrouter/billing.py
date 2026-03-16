"""Cost tracking engine — actual cost, baseline comparison, and savings reports."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from tokenrouter.models import MODEL_REGISTRY, ModelConfig, calculate_cost, get_model

# Default baseline model for "what would it cost on the premium model"
DEFAULT_BASELINE_MODEL = "claude-opus-4"


def get_baseline_model(baseline_id: str = DEFAULT_BASELINE_MODEL) -> ModelConfig | None:
    return get_model(baseline_id)


def calculate_baseline_cost(
    input_tokens: int,
    output_tokens: int,
    baseline_model_id: str = DEFAULT_BASELINE_MODEL,
) -> float:
    """Calculate what the request would cost on the baseline (premium) model."""
    model = get_baseline_model(baseline_model_id)
    if not model:
        return 0.0
    return calculate_cost(model, input_tokens, output_tokens)


@dataclass
class UsageSummary:
    period: str
    total_requests: int
    successful_requests: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    total_baseline_cost: float
    total_saved: float
    saved_pct: float
    models_used: dict[str, int]  # model_id -> request_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": round(self.total_cost, 6),
            "total_baseline_cost": round(self.total_baseline_cost, 6),
            "total_saved": round(self.total_saved, 6),
            "saved_pct": round(self.saved_pct, 2),
            "models_used": self.models_used,
        }


@dataclass
class SavingsReport:
    total_cost: float
    baseline_cost: float
    saved: float
    saved_pct: float
    request_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cost": round(self.total_cost, 6),
            "baseline_cost": round(self.baseline_cost, 6),
            "saved": round(self.saved, 6),
            "saved_pct": round(self.saved_pct, 2),
            "request_count": self.request_count,
        }


@dataclass
class DailyCost:
    date: str
    actual_cost: float
    baseline_cost: float


def _period_to_timestamp(period: str) -> float:
    """Convert period string to a start timestamp."""
    now = time.time()
    if period == "today":
        # Start of today (UTC)
        import datetime
        today = datetime.datetime.now(datetime.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        return today.timestamp()
    elif period == "week":
        return now - 7 * 86400
    elif period == "month":
        return now - 30 * 86400
    else:  # "all"
        return 0.0


class BillingEngine:
    """Query request_logs for usage and savings stats."""

    def __init__(self, db_conn: Any) -> None:
        """Takes a sqlite3 connection (from KeyStore._conn)."""
        self._conn = db_conn

    def get_usage_summary(self, tr_key_id: int, period: str = "all") -> UsageSummary:
        start_ts = _period_to_timestamp(period)
        rows = self._conn.execute(
            """SELECT
                COUNT(*) as total,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                COALESCE(SUM(input_tokens), 0) as input_tokens,
                COALESCE(SUM(output_tokens), 0) as output_tokens,
                COALESCE(SUM(actual_cost), 0) as cost,
                COALESCE(SUM(baseline_cost), 0) as baseline,
                COALESCE(SUM(saved), 0) as saved
            FROM request_logs
            WHERE tr_key_id = ? AND timestamp >= ?""",
            (tr_key_id, start_ts),
        ).fetchone()

        total, successful, input_tokens, output_tokens, cost, baseline, saved = rows

        # Models used
        model_rows = self._conn.execute(
            """SELECT model_used, COUNT(*) as cnt
            FROM request_logs
            WHERE tr_key_id = ? AND timestamp >= ?
            GROUP BY model_used""",
            (tr_key_id, start_ts),
        ).fetchall()
        models_used = {row[0]: row[1] for row in model_rows}

        saved_pct = (saved / baseline * 100) if baseline > 0 else 0.0

        return UsageSummary(
            period=period,
            total_requests=total,
            successful_requests=successful,
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_cost=cost,
            total_baseline_cost=baseline,
            total_saved=saved,
            saved_pct=saved_pct,
            models_used=models_used,
        )

    def get_savings_report(self, tr_key_id: int) -> SavingsReport:
        row = self._conn.execute(
            """SELECT
                COALESCE(SUM(actual_cost), 0),
                COALESCE(SUM(baseline_cost), 0),
                COALESCE(SUM(saved), 0),
                COUNT(*)
            FROM request_logs
            WHERE tr_key_id = ?""",
            (tr_key_id,),
        ).fetchone()

        total_cost, baseline_cost, saved, count = row
        saved_pct = (saved / baseline_cost * 100) if baseline_cost > 0 else 0.0

        return SavingsReport(
            total_cost=total_cost,
            baseline_cost=baseline_cost,
            saved=saved,
            saved_pct=saved_pct,
            request_count=count,
        )

    def get_daily_costs(self, tr_key_id: int, days: int = 30) -> list[DailyCost]:
        start_ts = time.time() - days * 86400
        rows = self._conn.execute(
            """SELECT
                DATE(timestamp, 'unixepoch') as day,
                COALESCE(SUM(actual_cost), 0),
                COALESCE(SUM(baseline_cost), 0)
            FROM request_logs
            WHERE tr_key_id = ? AND timestamp >= ?
            GROUP BY day
            ORDER BY day""",
            (tr_key_id, start_ts),
        ).fetchall()
        return [DailyCost(date=row[0], actual_cost=row[1], baseline_cost=row[2]) for row in rows]

    def get_recent_logs(self, tr_key_id: int, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT timestamp, model_requested, model_used, task_type, complexity,
                      input_tokens, output_tokens, actual_cost, baseline_cost, saved,
                      latency_ms, success
            FROM request_logs
            WHERE tr_key_id = ?
            ORDER BY timestamp DESC
            LIMIT ?""",
            (tr_key_id, limit),
        ).fetchall()
        return [
            {
                "timestamp": row[0],
                "model_requested": row[1],
                "model_used": row[2],
                "task_type": row[3],
                "complexity": row[4],
                "input_tokens": row[5],
                "output_tokens": row[6],
                "actual_cost": row[7],
                "baseline_cost": row[8],
                "saved": row[9],
                "latency_ms": row[10],
                "success": bool(row[11]),
            }
            for row in rows
        ]

    def get_model_distribution(self, tr_key_id: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT model_used, COUNT(*) as cnt, COALESCE(SUM(actual_cost), 0) as cost
            FROM request_logs
            WHERE tr_key_id = ?
            GROUP BY model_used
            ORDER BY cnt DESC""",
            (tr_key_id,),
        ).fetchall()
        return [{"model": row[0], "count": row[1], "cost": row[2]} for row in rows]

    def get_global_stats(self) -> dict[str, Any]:
        """Get stats across all keys (for dashboard overview)."""
        row = self._conn.execute(
            """SELECT
                COUNT(*),
                COALESCE(SUM(actual_cost), 0),
                COALESCE(SUM(baseline_cost), 0),
                COALESCE(SUM(saved), 0)
            FROM request_logs"""
        ).fetchone()
        total, cost, baseline, saved = row
        return {
            "total_requests": total,
            "total_cost": round(cost, 6),
            "total_baseline_cost": round(baseline, 6),
            "total_saved": round(saved, 6),
            "saved_pct": round(saved / baseline * 100, 2) if baseline > 0 else 0.0,
        }

    def get_global_daily_costs(self, days: int = 30) -> list[DailyCost]:
        start_ts = time.time() - days * 86400
        rows = self._conn.execute(
            """SELECT
                DATE(timestamp, 'unixepoch') as day,
                COALESCE(SUM(actual_cost), 0),
                COALESCE(SUM(baseline_cost), 0)
            FROM request_logs
            WHERE timestamp >= ?
            GROUP BY day
            ORDER BY day""",
            (start_ts,),
        ).fetchall()
        return [DailyCost(date=row[0], actual_cost=row[1], baseline_cost=row[2]) for row in rows]

    def get_global_model_distribution(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT model_used, COUNT(*) as cnt, COALESCE(SUM(actual_cost), 0) as cost
            FROM request_logs
            GROUP BY model_used
            ORDER BY cnt DESC"""
        ).fetchall()
        return [{"model": row[0], "count": row[1], "cost": row[2]} for row in rows]

    def get_global_recent_logs(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT r.timestamp, r.model_requested, r.model_used, r.task_type, r.complexity,
                      r.input_tokens, r.output_tokens, r.actual_cost, r.baseline_cost, r.saved,
                      r.latency_ms, r.success, t.name as key_name
            FROM request_logs r
            LEFT JOIN tr_keys t ON r.tr_key_id = t.id
            ORDER BY r.timestamp DESC
            LIMIT ?""",
            (limit,),
        ).fetchall()
        return [
            {
                "timestamp": row[0],
                "model_requested": row[1],
                "model_used": row[2],
                "task_type": row[3],
                "complexity": row[4],
                "input_tokens": row[5],
                "output_tokens": row[6],
                "actual_cost": row[7],
                "baseline_cost": row[8],
                "saved": row[9],
                "latency_ms": row[10],
                "success": bool(row[11]),
                "key_name": row[12] or "unknown",
            }
            for row in rows
        ]
