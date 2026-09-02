"""Score the saved LLM predictions into the paper-style F1 / AUC table.

Parses each prediction .txt into a label, aligns it with ground truth exactly as
the encoder does (weather: rain[pos+1]; finance/healthcare: labels[pos]), and
reports F1-micro (= accuracy), F1-macro and AUC per (task, dataset).

Run:  python -m pipeline.score --out-root pipeline_out/gpt-4o-mini
"""
from __future__ import annotations

import argparse
import os

from sklearn.metrics import f1_score, roc_auc_score

from pipeline.domains import domain_for_dataset, finance, healthcare, weather

DATASETS = ["weather_ny", "weather_sf", "weather_hs", "finance_sp500",
            "finance_nikkei", "healthcare_mortality", "healthcare_positive"]
COL = {"weather_ny": "NY", "weather_sf": "SF", "weather_hs": "HS", "finance_sp500": "SP500",
       "finance_nikkei": "Nikkei", "healthcare_mortality": "Mort", "healthcare_positive": "Pos"}
TASKS = ["predict_time", "predict_text", "predict_in_context"]


def _parse_binary_rain(text: str):
    t = text.lower()
    if "not rain" in t:
        return 0
    if "rain" in t:
        return 1
    return None


def _parse_binary_exceed(text: str):
    t = text.lower()
    if "not exceed" in t or "does not exceed" in t or "did not exceed" in t:
        return 0
    if "exceed" in t:
        return 1
    return None


def _parse_finance(text: str):
    t = text.lower()
    hits = [lab for kw, lab in (("decrease", 0), ("neutral", 1), ("increase", 2)) if kw in t]
    return hits[0] if len(hits) == 1 else None  # ambiguous -> None


def _relpath(mod, d, task: str, _i: int, k: int) -> str:
    i = d.indices[_i]
    if mod is weather:
        c = d.city
        return {"predict_time": f"gpt_predict/{c}_{i}.txt",
                "predict_text": f"gpt_predict_text/{c}_{i}.txt",
                "predict_in_context": f"gpt_predict_in-context/{c}_k{k}_{i}_ref.txt"}[task]
    ind = d.indicator
    suffix = "_ref_int" if mod is healthcare else "_ref"
    return {"predict_time": f"gpt_predict_time/{i}_{ind}.txt",
            "predict_text": f"gpt_predict_text/{i}_{ind}.txt",
            "predict_in_context": f"gpt_predict_in-context/k{k}_{i}_{ind}{suffix}.txt"}[task]


def _ground_truth(mod, d, _i: int) -> int:
    if mod is weather:
        return int(bool(d.rain[_i + 1]))   # encoder: rain[pos + pred_len], pred_len=1
    return int(d.labels[_i])               # finance/healthcare: labels[pos]


def _parser(mod):
    return _parse_binary_rain if mod is weather else (_parse_finance if mod is finance else _parse_binary_exceed)


def score_cell(ds: str, task: str, out_root: str, data_root: str, k: int):
    mod = domain_for_dataset(ds)
    d = mod.load(ds, data_root)
    parse = _parser(mod)
    domain_dir = os.path.join(out_root, mod.NAME)
    y_true, y_pred, missing, unparsed = [], [], 0, 0
    for _i in d.splits["test"]:
        path = os.path.join(domain_dir, _relpath(mod, d, task, _i, k))
        if not os.path.exists(path):
            missing += 1
            continue
        with open(path, "r", encoding="utf-8") as f:
            lab = parse(f.read())
        if lab is None:
            unparsed += 1
            continue
        y_true.append(_ground_truth(mod, d, _i))
        y_pred.append(lab)
    total = len(d.splits["test"])
    if not y_true:
        return None
    f1mi = f1_score(y_true, y_pred, average="micro")
    f1ma = f1_score(y_true, y_pred, average="macro")
    try:
        if mod is finance:
            import numpy as np
            oh = np.eye(3)[y_pred]
            auc = roc_auc_score(y_true, oh, multi_class="ovr", average="macro", labels=[0, 1, 2])
        else:
            auc = roc_auc_score(y_true, y_pred)
    except Exception:
        auc = None
    return {"f1_micro": f1mi, "f1_macro": f1ma, "auc": auc,
            "n": len(y_true), "total": total, "missing": missing, "unparsed": unparsed}


def _print_table(title, cells, key, fmt="{:.3f}"):
    w = 9
    header = "task \\ dataset".ljust(20) + "".join(COL[ds].rjust(w) for ds in DATASETS)
    print(f"\n=== {title} ===\n{header}\n{'-'*len(header)}")
    for t in TASKS:
        row = ""
        for ds in DATASETS:
            c = cells[t][ds]
            v = c.get(key) if c else None
            row += (fmt.format(v) if isinstance(v, float) else "-").rjust(w)
        print(t.ljust(20) + row)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="pipeline_out/gpt-4o-mini")
    ap.add_argument("--data-root", default="dataset")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args(argv)

    cells = {t: {} for t in TASKS}
    cov_issues = []
    for ds in DATASETS:
        for t in TASKS:
            c = score_cell(ds, t, args.out_root, args.data_root, args.k)
            cells[t][ds] = c
            if c and (c["missing"] or c["unparsed"]):
                cov_issues.append(f"{t}/{ds}: parsed {c['n']}/{c['total']} "
                                  f"(missing={c['missing']}, unparsed={c['unparsed']})")

    print(f"Scored predictions under: {args.out_root}")
    _print_table("F1-macro", cells, "f1_macro")
    _print_table("F1-micro (accuracy)", cells, "f1_micro")
    _print_table("AUC (from hard labels — approximate)", cells, "auc")
    if cov_issues:
        print("\nCoverage notes (cells with missing/unparsed predictions):")
        for line in cov_issues:
            print("  " + line)
    else:
        print("\nCoverage: 100% of test predictions found and parsed for every cell.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
