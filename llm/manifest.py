"""Checkpoint / resume support: skip work that is already done.

Uses an append-only JSONL of completed keys (crash-safe: a half-written last
line is simply ignored on reload). Makes re-running a batch idempotent, so an
interrupted run resumes without re-spending tokens on completed samples.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone


def make_key(task: str, dataset: str, idx) -> str:
    return f"{task}|{dataset}|{idx}"


class Manifest:
    def __init__(self, run_id: str, manifest_dir: str = "llm_runs"):
        self.run_id = run_id
        os.makedirs(manifest_dir, exist_ok=True)
        self.path = os.path.join(manifest_dir, f"{run_id}.manifest.jsonl")
        self._done: set[str] = set()
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._done.add(json.loads(line)["key"])
                except (json.JSONDecodeError, KeyError):
                    # tolerate a torn final line from a crash mid-write
                    continue

    def is_done(self, task: str, dataset: str, idx) -> bool:
        return make_key(task, dataset, idx) in self._done

    def mark_done(self, task: str, dataset: str, idx) -> None:
        key = make_key(task, dataset, idx)
        rec = {"key": key, "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with self._lock:  # safe for concurrent workers
            if key in self._done:
                return
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)
            self._done.add(key)

    def __len__(self) -> int:
        return len(self._done)
