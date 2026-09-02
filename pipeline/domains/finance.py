"""Finance domain — faithful port of the four finance notebooks.

Datasets: finance_sp500 / finance_nikkei. Shared market time series (columns
1..9 below), per-indicator labels (0=decrease, 1=neutral, 2=increase). Window =
20 market days. Contextualize summaries are market-level (indicator-independent),
so they are written as gpt_summary/<i>.txt and shared across both indicators.

Output filename conventions match the notebooks:
  contextualize      -> gpt_summary/<i>.txt
  predict_time       -> gpt_predict_time/<i>_<indicator>.txt
  predict_text       -> gpt_predict_text/<i>_<indicator>.txt
  predict_in_context -> gpt_predict_in-context/k<k>_<i>_<indicator>_ref.txt
"""
from __future__ import annotations

import os
import pickle as pkl
import random
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from ..base import Unit, compute_splits, fmt_series, retrieve_topk

NAME = "finance"
WINDOW = 20
DATASETS = {"finance_sp500": "sp500", "finance_nikkei": "nikkei"}
IND2NAME = {"sp500": "S&P 500", "nikkei": "Nikkei 225"}
TASKS = ("contextualize", "predict_time", "predict_text", "predict_in_context")

_SYS_ANALYST = ("Your job is to act as a professional finance analyst. You will write a "
                "high-quality report that is informative and helps in understanding the "
                "current financial situation.")


def _sys_forecast_time(ind: str) -> str:
    return ("Your job is to act as a professional financial forecaster. You will be given a "
            f"time-series data from the past 20 market days. Based on this information, your task "
            f"is to predict whether the {IND2NAME[ind]} price will decrease by more than 1%, "
            "increase by more than 1%, or change minimally in the next market day.")


def _sys_forecast_text(ind: str) -> str:
    return ("Your job is to act as a professional financial forecaster. You will be given a "
            f"financial summary of the past 20 market days. Based on this information, your task "
            f"is to predict whether the {IND2NAME[ind]} price will decrease by more than 1%, "
            "increase by more than 1%, or change minimally in the next market day.")


def _sys_forecast_incontext(ind: str) -> str:
    return ("Your job is to act as a professional financial forecaster. You will be given a "
            f"summary of the financial situation of the past 20 market days. Based on this "
            f"information, your task is to predict whether the {IND2NAME[ind]} price will decrease "
            "by more than 1%, increase by more than 1%, or change minimally in the next market day.")


@dataclass
class FinanceData:
    dataset: str
    indicator: str
    data: np.ndarray
    indices: list
    labels: np.ndarray
    splits: dict
    data_dir: str


def _pkl(path: str):
    with open(path, "rb") as f:
        return pkl.load(f)


def load(dataset: str, data_root: str) -> FinanceData:
    ind = DATASETS[dataset]
    ddir = os.path.join(data_root, "finance")
    indices = list(_pkl(os.path.join(ddir, "indices.pkl")))
    data = _pkl(os.path.join(ddir, "time_series.pkl"))
    labels = np.asarray(_pkl(os.path.join(ddir, f"labels_{ind}.pkl")))
    splits = compute_splits(len(indices), seq_len_day=0)
    return FinanceData(dataset, ind, data, indices, labels, splits, ddir)


# time series columns (col 0 is the date):
_COLS = [
    ("S&P 500", 1), ("VIX (Volatility Index)", 2), ("Nikkei 225", 3),
    ("FTSE 100", 4), ("Gold Futures", 5), ("Crude Oil Futures", 6),
    ("Exchange rate for EUR/USD", 7), ("Exchange rate for USD/JYP", 8),  # "JYP" typo kept for fidelity
    ("Exchange rate for USD/CNY", 9),
]


def _series_block(data, i: int) -> str:
    w = data[i:i + WINDOW]
    return "".join(f"- {label}: {fmt_series(w[:, col])}\n" for label, col in _COLS) + "\n"


def _contextualize_prompt(fd: FinanceData, i: int) -> tuple[str, str]:
    u = f"Your task is to analyze key financial indicators over the last {WINDOW} market days."
    u += f"\n\nReview the time-series data provided for the last {WINDOW} market days. "
    u += "Each time-series consists of daily values separated by a '|' token for the following indicators:\n"
    u += _series_block(fd.data, i)
    u += "Based on this time-series data, write a concise report that provides insights crucial for understanding the current financial situation. "
    u += "Your report should be limited to five sentences, yet comprehensive, highlighting key trends and considering their potential impact on the market."
    u += "Do not write numerical values while writing the report."
    return _SYS_ANALYST, u


def _predict_time_prompt(fd: FinanceData, i: int) -> tuple[str, str]:
    name = IND2NAME[fd.indicator]
    u = f"Your task is to predict whether the {name} price will:\n"
    u += "(1) Decrease: decrease by more than 1%\n(2) Increase: increase by more than 1%\n(3) Neutral: change minimally, between -1% to 1%\nin the next market day. "
    u += f"Review the time-series data provided for the last {WINDOW} market days. "
    u += "Each time-series consists of daily values separated by a '|' token for the following indicators:\n\n"
    u += _series_block(fd.data, i)
    u += f"Based on this information, predict whether the {name} price will decrease by more than 1%, increase by more than 1%, or otherwise, in the next market day. "
    u += "Respond with either 'decrease', 'increase', or 'neutral'. Do not provide any other details. "
    return _sys_forecast_time(fd.indicator), u


def _predict_text_prompt(fd: FinanceData, i: int, summary: str) -> tuple[str, str]:
    name = IND2NAME[fd.indicator]
    u = f"Your task is to predict whether the {name} price will:\n"
    u += "(1) Decrease: decrease by more than 1%\n(2) Increase: increase by more than 1%\n(3) Neutral: change minimally, between -1% to 1%\nin the next market day. "
    u += f"The financial situation of the last {WINDOW} market days is summarized as follows:\n\n"
    u += f"{summary.replace(chr(10) + chr(10), ' ')}\n\n"
    u += f"Based on this information, predict whether the {name} price will decrease by more than 1%, increase by more than 1%, or otherwise (neutral), in the next market day. "
    u += "Respond with either 'decrease', 'increase', or 'neutral'. Do not provide any other details. "
    return _sys_forecast_text(fd.indicator), u


_FIN_OUTCOME = {0: "Decreased", 1: "Neutral", 2: "Increased"}


def _predict_in_context_prompt(fd: FinanceData, i: int, texts: dict, text_emb: dict, k: int) -> tuple[str, str]:
    name = IND2NAME[fd.indicator]
    idx_train = fd.splits["train"]
    u = f"Your task is to predict whether the {name} price will:\n"
    u += "(1) Decrease: decrease by more than 1%\n(2) Increase: increase by more than 1%\n(3) Neutral: change minimally, between -1% and 1%\nin the next market day. "
    u += f"First, review the following {k} examples of financial summaries and {name} outcomes so that you can refer to when making predictions.\n\n"
    for _k, _j in enumerate(retrieve_topk(text_emb, i, idx_train, fd.indices, k)):
        j = fd.indices[_j]
        u += f"Summary #{_k + 1}: {texts[j].replace(chr(10) + chr(10), ' ')}"
        u += f"\nOutcome #{_k + 1}: {_FIN_OUTCOME[int(fd.labels[_j])]}\n\n"
    u += f"The financial situation of the last {WINDOW} market days is summarized as follows:\n\n"
    u += f"Summary: {texts[i].replace(chr(10) + chr(10), ' ')}\n"
    u += "Outcome:\n\n"
    u += "Refer to the provided examples and predict the outcome of the current financial summary. "
    u += "Respond your prediction with either 'decrease', 'increase' or 'neutral'. "
    u += "Response should not include other terms."
    return _sys_forecast_incontext(fd.indicator), u


def _load_summaries(fd: FinanceData, summary_dir: str) -> dict:
    texts = {}
    for i in fd.indices:
        path = os.path.join(summary_dir, f"{i}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing summary {path}. Run 'contextualize' first or set --summary-dir.")
        with open(path, "r", encoding="utf-8") as f:
            texts[i] = f.read()
    return texts


def _load_embeddings(fd: FinanceData, encoder_root: str) -> dict:
    # NOTE: the notebook mapped indices[:-1], but finance embeddings have the full
    # length (len == len(indices)); [:-1] drops the last embedding and crashes on
    # the last test sample. Map all indices (train retrieval is unchanged).
    embs = _pkl(os.path.join(encoder_root, "embeddings", f"finance_{fd.indicator}.pkl"))
    return {fd.indices[_i]: embs[_i] for _i in range(len(fd.indices))}


def build_units(
    task: str,
    fd: FinanceData,
    *,
    summary_dir: Optional[str] = None,
    encoder_root: Optional[str] = None,
    k: int = 5,
) -> Iterator[Unit]:
    ind = fd.indicator
    if task == "contextualize":
        for i in fd.indices:
            sys, usr = _contextualize_prompt(fd, i)
            yield Unit(i, sys, usr, f"gpt_summary/{i}.txt")
    elif task == "predict_time":
        for _i in fd.splits["test"]:
            i = fd.indices[_i]
            sys, usr = _predict_time_prompt(fd, i)
            yield Unit(i, sys, usr, f"gpt_predict_time/{i}_{ind}.txt")
    elif task == "predict_text":
        texts = _load_summaries(fd, summary_dir or os.path.join(fd.data_dir, "gpt_summary"))
        for _i in fd.splits["test"]:
            i = fd.indices[_i]
            sys, usr = _predict_text_prompt(fd, i, texts[i])
            yield Unit(i, sys, usr, f"gpt_predict_text/{i}_{ind}.txt")
    elif task == "predict_in_context":
        random.seed(2024)
        if not encoder_root:
            raise ValueError("predict_in_context needs --encoder-root (for retrieval embeddings).")
        texts = _load_summaries(fd, summary_dir or os.path.join(fd.data_dir, "gpt_summary"))
        text_emb = _load_embeddings(fd, encoder_root)
        for _i in fd.splits["test"]:
            i = fd.indices[_i]
            sys, usr = _predict_in_context_prompt(fd, i, texts, text_emb, k)
            yield Unit(i, sys, usr, f"gpt_predict_in-context/k{k}_{i}_{ind}_ref.txt")
    else:
        raise ValueError(f"Unknown task {task!r}. Finance tasks: {TASKS}")
