"""Shared pipeline primitives: the work Unit, split logic, formatting, atomic IO.

A *Unit* is one LLM job: which sample, the prompt, and where its output goes.
Domains turn (task, data) into a stream of Units; the runner executes them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Unit:
    idx: int          # data index — used in filename, manifest key, and log
    system: Optional[str]
    user: str
    out_relpath: str  # output path relative to the run's output root


def compute_splits(n: int, seq_len_day: int = 1) -> dict[str, list[int]]:
    """Reproduce the notebooks' 60/20/20 train/val/test position split.

    Positions index into ``indices`` (not into the raw time series).
    """
    num_train = int(n * 0.6)
    num_test = int(n * 0.2)
    num_vali = n - num_train - num_test
    train = list(range(0, num_train - seq_len_day))
    valid = list(range(num_train - seq_len_day, num_train + num_vali - seq_len_day))
    test = list(range(num_train + num_vali - seq_len_day, num_train + num_vali + num_test - seq_len_day))
    return {"train": train, "valid": valid, "test": test}


def fmt_series(values, fmt: str = "{:.2f}", sep: str = "|") -> str:
    """Format a 1-D numeric sequence like the notebooks: '12.30|11.90|...'."""
    return sep.join(fmt.format(float(x)) for x in values)


def fmt_series_str(values, sep: str = "|") -> str:
    """Join values as-is (some datasets store integer counts as strings)."""
    return sep.join(str(x) for x in values)


def retrieve_topk(text_emb: dict, i: int, idx_train, indices, k: int) -> list[int]:
    """Return the top-k train POSITIONS (_j) most similar to sample i by cosine.

    Mirrors the notebooks' retrieval: rank by descending cosine over the training
    set. (The notebooks used ``norm``/``random`` without importing them — a latent
    NameError — so retrieval lives here, imported correctly, once.)
    """
    import numpy as np
    from numpy.linalg import norm

    def cos(a, b):
        return float(np.dot(a, b) / (norm(a) * norm(b)))

    sim = [-cos(text_emb[i], text_emb[indices[ii]]) for ii in idx_train]
    order = np.argsort(sim)
    return [idx_train[int(order[_k])] for _k in range(k)]


def atomic_write_text(path: str, text: str) -> None:
    """Write ``text`` to ``path`` atomically (tmp file + os.replace).

    Prevents half-written output files if the process is killed mid-write, so a
    resumed run never sees a truncated result it would mistake for complete.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
