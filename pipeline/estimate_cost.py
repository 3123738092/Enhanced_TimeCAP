"""Offline cost ESTIMATE for the full LLM experiment (no API calls, no spend).

For each (task, dataset) it samples a few prompts, measures input tokens exactly
with the model's tokenizer (tiktoken), assumes a per-task output-token estimate
(from the observed gpt-4o smoke), and multiplies by the real call count to give a
per-cell USD estimate — laid out like the paper's table (rows = LLM tasks,
columns = datasets).

Run:  python -m pipeline.estimate_cost [--model gpt-4o] [--sample 15]
Actual costs after a real run come from the token logs (see pipeline.report).
"""
from __future__ import annotations

import argparse
import itertools

from llm.estimate import count_tokens
from llm.pricing import lookup
from pipeline.domains import domain_for_dataset

DATASETS = ["weather_ny", "weather_sf", "weather_hs", "finance_sp500",
            "finance_nikkei", "healthcare_mortality", "healthcare_positive"]
COL = {"weather_ny": "NY", "weather_sf": "SF", "weather_hs": "HS", "finance_sp500": "SP500",
       "finance_nikkei": "Nikkei", "healthcare_mortality": "Mort", "healthcare_positive": "Pos"}
TASKS = ["contextualize", "predict_time", "predict_text", "predict_in_context"]
# output-token estimate per task (contextualize ~145 observed; predictions are a single label)
OUT_EST = {"contextualize": 150, "predict_time": 5, "predict_text": 5, "predict_in_context": 5}


def n_calls(mod, d, task: str) -> int:
    return len(d.indices) if task == "contextualize" else len(d.splits["test"])


def avg_input_tokens(mod, d, task: str, model: str, sample: int) -> float:
    units = list(itertools.islice(mod.build_units(task, d, encoder_root="encoder", k=5), sample))
    if not units:
        return 0.0
    toks = [count_tokens((u.system or "") + "\n" + u.user, model) for u in units]
    return sum(toks) / len(toks)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--sample", type=int, default=15)
    args = ap.parse_args(argv)

    price = lookup(args.model)
    if price is None:
        raise SystemExit(f"No price for model {args.model!r}; add it to llm/pricing.py")
    p_in, p_out = price["in"], price["out"]

    cost = {t: {} for t in TASKS}
    calls = {t: {} for t in TASKS}
    for ds in DATASETS:
        mod = domain_for_dataset(ds)
        d = mod.load(ds, "dataset")
        for t in TASKS:
            n = n_calls(mod, d, t)
            ain = avg_input_tokens(mod, d, t, args.model, args.sample)
            c = n * (ain * p_in + OUT_EST[t] * p_out) / 1_000_000.0
            calls[t][ds] = n
            cost[t][ds] = c
            print(f"  [{t:18s} {ds:20s}] calls={n:5d} avg_in_tok={ain:7.0f} -> ${c:8.3f}")

    w = 9
    header = "task \\ dataset".ljust(20) + "".join(COL[ds].rjust(w) for ds in DATASETS) + "row$".rjust(w)
    print(f"\n=== ESTIMATED COST (USD, model={args.model}) ===\n{header}\n{'-'*len(header)}")
    grand = 0.0
    for t in TASKS:
        row = sum(cost[t][ds] for ds in DATASETS)
        grand += row
        print(t.ljust(20) + "".join(f"{cost[t][ds]:{w}.2f}" for ds in DATASETS) + f"{row:{w}.2f}")
    coltot = "TOTAL".ljust(20) + "".join(f"{sum(cost[t][ds] for t in TASKS):{w}.2f}" for ds in DATASETS) + f"{grand:{w}.2f}"
    print("-" * len(header) + "\n" + coltot)

    total_calls = sum(calls[t][ds] for t in TASKS for ds in DATASETS)
    print(f"\ntotal calls (all tasks x datasets): {total_calls}")
    print("NOTE: contextualize for finance_sp500 & finance_nikkei is the SAME shared "
          "summary set — run it once (the other is skipped), so subtract one finance "
          "contextualize column from the real spend.")
    print("These are ESTIMATES (input tokens exact via tiktoken; output assumed). "
          "Real costs are logged per call and aggregated by pipeline.report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
