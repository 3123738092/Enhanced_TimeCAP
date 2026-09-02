"""Append-only per-call token/cost log (JSONL) plus aggregation.

One line = one LLM call. We store RAW token counts (not only cost_usd) so cost
can be recomputed if the price table changes. Aggregation reports Total /
per-instance average / median, following the CiK reporting style in the spec.
"""
from __future__ import annotations

import json
import os
import statistics
import threading
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

# Canonical column order for one logged call.
FIELDS = (
    "ts", "run_id", "channel", "model", "vendor", "task", "dataset", "idx",
    "input_tokens", "output_tokens", "reasoning_tokens",
    "cached_read_tokens", "cache_write_tokens", "total_tokens",
    "token_source", "price_snapshot_id", "fx_rate", "cost_usd",
    "latency_s", "attempt", "ok", "error",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class TokenLogger:
    """Appends one JSON object per line to ``<log_dir>/<run_id>.jsonl``."""

    def __init__(self, run_id: str, log_dir: str = "llm_runs"):
        self.run_id = run_id
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, f"{run_id}.jsonl")
        self._lock = threading.Lock()

    def log(self, record: dict[str, Any]) -> dict[str, Any]:
        row = {k: record.get(k) for k in FIELDS}
        row["ts"] = row["ts"] or _utc_now()
        row["run_id"] = row["run_id"] or self.run_id
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self._lock:  # safe for concurrent workers
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)
        return row

    def read(self) -> list[dict[str, Any]]:
        return list(read_jsonl(self.path))

    def summary(self) -> dict[str, Any]:
        return summarize(self.read())


def read_jsonl(path: str) -> Iterable[dict[str, Any]]:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _num(x: Any) -> float:
    return float(x) if isinstance(x, (int, float)) else 0.0


def summarize(rows: Iterable[dict[str, Any]], group_by: Optional[str] = None) -> dict[str, Any]:
    """Aggregate calls into total / per-call average & median of tokens and cost."""
    rows = list(rows)
    ok_rows = [r for r in rows if r.get("ok", True)]
    costs = [_num(r.get("cost_usd")) for r in ok_rows if r.get("cost_usd") is not None]
    out: dict[str, Any] = {
        "calls": len(rows),
        "ok_calls": len(ok_rows),
        "failed_calls": len(rows) - len(ok_rows),
        "input_tokens": sum(_num(r.get("input_tokens")) for r in ok_rows),
        "output_tokens": sum(_num(r.get("output_tokens")) for r in ok_rows),
        "reasoning_tokens": sum(_num(r.get("reasoning_tokens")) for r in ok_rows),
        "total_tokens": sum(_num(r.get("total_tokens")) for r in ok_rows),
        "cost_usd_total": round(sum(costs), 6) if costs else None,
        "cost_usd_avg": round(statistics.mean(costs), 6) if costs else None,
        "cost_usd_median": round(statistics.median(costs), 6) if costs else None,
        "priced_calls": len(costs),
        "estimated_calls": sum(1 for r in ok_rows if r.get("token_source") == "estimated"),
    }
    if group_by:
        groups: dict[Any, list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault(r.get(group_by), []).append(r)
        out["by_" + group_by] = {str(k): summarize(v) for k, v in groups.items()}
    return out
