"""Aggregate ACTUAL cost/tokens from the token logs into the paper-style table.

Reads every per-call log under --log-dir (skipping the *.manifest.jsonl resume
files), groups by (task, dataset), and prints USD cost + token totals laid out
with LLM tasks as rows and datasets as columns.

Run:  python -m pipeline.report [--log-dir llm_runs] [--channel openai] [--model gpt-4o]
"""
from __future__ import annotations

import argparse
import glob
import os

from llm.tokenlog import read_jsonl

DATASETS = ["weather_ny", "weather_sf", "weather_hs", "finance_sp500",
            "finance_nikkei", "healthcare_mortality", "healthcare_positive"]
COL = {"weather_ny": "NY", "weather_sf": "SF", "weather_hs": "HS", "finance_sp500": "SP500",
       "finance_nikkei": "Nikkei", "healthcare_mortality": "Mort", "healthcare_positive": "Pos"}
TASKS = ["contextualize", "predict_time", "predict_text", "predict_in_context"]


def _num(x):
    return float(x) if isinstance(x, (int, float)) else 0.0


def load_rows(log_dir: str, channel=None, model=None):
    rows = []
    for path in sorted(glob.glob(os.path.join(log_dir, "*.jsonl"))):
        if path.endswith(".manifest.jsonl"):
            continue
        for r in read_jsonl(path):
            if not r.get("ok", False):
                continue
            if channel and r.get("channel") != channel:
                continue
            if model and r.get("model") != model:
                continue
            rows.append(r)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="llm_runs")
    ap.add_argument("--channel", default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args(argv)

    rows = load_rows(args.log_dir, args.channel, args.model)
    if not rows:
        raise SystemExit(f"No successful calls found in {args.log_dir} (filters: "
                         f"channel={args.channel}, model={args.model}). Run the pipeline first.")

    cost = {t: {ds: 0.0 for ds in DATASETS} for t in TASKS}
    calls = {t: {ds: 0 for ds in DATASETS} for t in TASKS}
    toks = {t: {ds: 0.0 for ds in DATASETS} for t in TASKS}
    priced_missing = 0
    for r in rows:
        t, ds = r.get("task"), r.get("dataset")
        if t not in cost or ds not in cost[t]:
            continue
        calls[t][ds] += 1
        toks[t][ds] += _num(r.get("total_tokens"))
        if r.get("cost_usd") is None:
            priced_missing += 1
        else:
            cost[t][ds] += _num(r.get("cost_usd"))

    scope = f"channel={args.channel or 'ALL'} model={args.model or 'ALL'}"
    w = 9
    header = "task \\ dataset".ljust(20) + "".join(COL[ds].rjust(w) for ds in DATASETS) + "row$".rjust(w)
    print(f"=== ACTUAL COST (USD)  [{scope}] ===\n{header}\n{'-'*len(header)}")
    grand = 0.0
    for t in TASKS:
        row = sum(cost[t][ds] for ds in DATASETS)
        grand += row
        print(t.ljust(20) + "".join(f"{cost[t][ds]:{w}.2f}" for ds in DATASETS) + f"{row:{w}.2f}")
    print("-" * len(header))
    print("TOTAL".ljust(20) + "".join(f"{sum(cost[t][ds] for t in TASKS):{w}.2f}" for ds in DATASETS) + f"{grand:{w}.2f}")

    print("\n=== calls per cell ===\n" + "task \\ dataset".ljust(20) + "".join(COL[ds].rjust(w) for ds in DATASETS))
    for t in TASKS:
        print(t.ljust(20) + "".join(f"{calls[t][ds]:{w}d}" for ds in DATASETS))

    print(f"\ntotal successful calls: {sum(calls[t][ds] for t in TASKS for ds in DATASETS)}"
          f"  total cost: ${grand:.4f}")
    if priced_missing:
        print(f"WARNING: {priced_missing} calls had no price (model not in llm/pricing.py); "
              "tokens counted, cost not. Add the model's price to include it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
