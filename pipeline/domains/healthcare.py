"""Healthcare domain — faithful port of the four healthcare notebooks.

Datasets: healthcare_mortality / healthcare_positive. Window = 20 weeks.
Binary label (0 = did not exceed the average threshold, 1 = exceeded).
Thresholds: positive 6.26%, mortality 7.84%.

Count columns are stored as strings and emitted verbatim; rate columns are
formatted to 2 decimals — matching the notebooks exactly.

Bugfixes vs the original notebooks (all latent NameErrors that would crash):
  * predict_text checked ``indicator == 'posiive'`` (typo) -> 'positive'
  * in-context checked ``elif inidicator == 'mortality'`` (typo) -> 'indicator'
  * both used ``norm`` / ``random`` without importing them

Output filename conventions match the notebooks:
  contextualize      -> gpt_summary/<i>_<indicator>.txt
  predict_time       -> gpt_predict_time/<i>_<indicator>.txt
  predict_text       -> gpt_predict_text/<i>_<indicator>.txt
  predict_in_context -> gpt_predict_in-context/k<k>_<i>_<indicator>_ref_int.txt
"""
from __future__ import annotations

import os
import pickle as pkl
import random
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from ..base import Unit, compute_splits, fmt_series, fmt_series_str, retrieve_topk

NAME = "healthcare"
WINDOW = 20
DATASETS = {"healthcare_mortality": "mortality", "healthcare_positive": "positive"}
TASKS = ("contextualize", "predict_time", "predict_text", "predict_in_context")

AVG = {"positive": "6.26", "mortality": "7.84"}
METRIC = {
    "positive": "the percentage of respiratory specimens testing positive for influenza",
    "mortality": "the ratio of mortality from Influenza or Pneumonia to the total number of death",
}
_SYS_ANALYST = ("Your job is to act as a professional healthcare analyst. You will write a "
                "high-quality report that is informative and helps understand the current "
                "healthcare situation.")
# task -> indicator -> tail phrase of the forecaster system prompt
_SYS_TAIL = {
    "predict_time": {"positive": "exceed its average", "mortality": "exceed its average"},
    "predict_text": {"positive": "exceed the average threshold", "mortality": "exceed its average"},
    "predict_in_context": {"positive": "exceed the average threshold", "mortality": "exceed the average threshold"},
}


@dataclass
class HealthcareData:
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


def load(dataset: str, data_root: str) -> HealthcareData:
    ind = DATASETS[dataset]
    ddir = os.path.join(data_root, "healthcare")
    indices = list(_pkl(os.path.join(ddir, f"indices_{ind}.pkl")))
    data = _pkl(os.path.join(ddir, f"time_series_{ind}.pkl"))
    labels = np.asarray(_pkl(os.path.join(ddir, f"labels_{ind}.pkl")))
    splits = compute_splits(len(indices), seq_len_day=0)
    return HealthcareData(dataset, ind, data, indices, labels, splits, ddir)


def _sys_forecast(task: str, ind: str) -> str:
    given = "a time-series data from the past 20 weeks" if task == "predict_time" else "a healthcare summary of the past 20 weeks"
    return (f"Your job is to act as a professional healthcare forecaster. You will be given {given}. "
            f"Based on this information, your task is to predict whether {METRIC[ind]} will "
            f"{_SYS_TAIL[task][ind]} in the comming week.")


def _block(ind: str, data, i: int) -> str:
    w = data[i:i + WINDOW]
    if ind == "positive":
        return (f"- Number of specimens tested: {fmt_series_str(w[:, 1])}\n"
                f"- Number of positive specimens for Influenza A: {fmt_series_str(w[:, 2])}\n"
                f"- Number of positive specimens for Influenza B: {fmt_series_str(w[:, 3])}\n"
                f"- Ratio of positive specimens (%): {fmt_series(w[:, 4])}\n"
                f"- Ratio of positive specimens for Influenza A (%): {fmt_series(w[:, 5])}\n"
                f"- Ratio of positive specimens for Influenza B (%): {fmt_series(w[:, 6])}\n\n")
    return (f"- Total number of death: {fmt_series_str(w[:, 3])}\n"
            f"- Number of death from influenza: {fmt_series_str(w[:, 1])}\n"
            f"- Number of death from pneumonia: {fmt_series_str(w[:, 2])}\n"
            f"- Ratio of mortality from Influenza or Pneumonia (%): {fmt_series(w[:, 4])}\n\n")


def _contextualize_prompt(hd: HealthcareData, i: int) -> tuple[str, str]:
    ind = hd.indicator
    if ind == "positive":
        u = f"Your task is to analyze the respiratory specimens testing positive for influenza over the last {WINDOW} weeks. "
        u += "The average ratio of positive speciemens is 6.26%."
    else:
        u = f"Your task is to analyze the mortality from Influenza or Pneumonia over the last {WINDOW} weeks. "
        u += "The average ratio of mortality from Influenza or Pneumonia to the total number of death is 7.84%."
    u += f"\n\nReview the time-series data provided for the last {WINDOW} weeks. "
    u += "Each time-series consists of weekly values separated by a '|' token for the following indicators:\n"
    u += _block(ind, hd.data, i)
    u += "Based on this time-series data, write a concise report that provides insights crucial for understanding the current healthcare situation. "
    u += "Your report should be limited to five sentences, yet comprehensive, highlighting key trends and considering their potential impact on the healthcare system. "
    u += "Do not write redundant information."
    return _SYS_ANALYST, u


def _predict_time_prompt(hd: HealthcareData, i: int) -> tuple[str, str]:
    ind = hd.indicator
    avg = AVG[ind]
    not_line = "Does not exceed" if ind == "mortality" else "Not exceed"
    u = f"Your task is to predict whether {METRIC[ind]} will:\n"
    u += f"(1) Exceed its average of {avg}%\n(2) {not_line} its average of {avg}%\n"
    u += "in the coming week. "
    u += f"Review the time-series data provided for the last {WINDOW} weeks. "
    u += "Each time-series consists of weekly values separated by a '|' token for the following indicators:\n"
    u += _block(ind, hd.data, i)
    if ind == "positive":
        u += f"Based on this time-series data, predict whether {METRIC[ind]} will exceed its average of {avg}% or not in the comming week. "
    else:
        u += f"Based on this time-series data, predict whether {METRIC[ind]} will exceed {avg}% or not. "
    u += "Respond with either 'exceed' or 'not exceed'. Do not provide any other details."
    return _sys_forecast("predict_time", ind), u


def _predict_text_prompt(hd: HealthcareData, i: int, summary: str) -> tuple[str, str]:
    ind = hd.indicator
    avg = AVG[ind]
    u = f"Your task is to predict whether {METRIC[ind]} will:\n"
    u += f"(1) Exceed its average of {avg}%\n(2) Not exceed its average of {avg}%\n"
    u += "in the coming week. "
    u += f"The healthcare situation of the last {WINDOW} weeks is summarized as follows:\n\n"
    u += f"{summary}\n\n"
    u += f"Analyze this summary and predict whether {METRIC[ind]} will exceed the average of {avg}% or not. "
    u += "Respond with either 'exceed' or 'not exceed'. Do not provide any other details."
    return _sys_forecast("predict_text", ind), u


def _predict_in_context_prompt(hd: HealthcareData, i: int, texts: dict, text_emb: dict, k: int) -> tuple[str, str]:
    ind = hd.indicator
    avg = AVG[ind]
    idx_train = hd.splits["train"]
    u = f"Your task is to predict whether {METRIC[ind]} will:\n"
    u += f"(1) Exceed its average of {avg}%\n(2) Not exceed its average of {avg}%\n"
    u += "in the coming week. "
    u += f"First, review the following {k} examples of healthcare summaries and their outcomes so that you can refer to when making predictions.\n\n"
    for _k, _j in enumerate(retrieve_topk(text_emb, i, idx_train, hd.indices, k)):
        j = hd.indices[_j]
        u += f"Summary #{_k + 1}: {texts[j].replace(chr(10) + chr(10), ' ')}"
        label = int(hd.labels[_j])
        outcome = f"Did not exceed {avg}%" if label == 0 else f"Exceeded {avg}%"
        u += f"\nOutcome #{_k + 1}: {outcome}\n\n"
    u += f"The healthcare situation of the last {WINDOW} weeks is summarized as follows:\n\n"
    u += f"Summary: {texts[i].replace(chr(10) + chr(10), ' ')}\n"
    u += "Outcome:\n\n"
    u += "Refer to the provided examples and predict the outcome of the current healthcare summary. "
    u += "Respond with either 'exceed' or 'not exceed'. "
    u += "Response should not include other terms."
    return _sys_forecast("predict_in_context", ind), u


def _load_summaries(hd: HealthcareData, summary_dir: str) -> dict:
    texts = {}
    for i in hd.indices:
        path = os.path.join(summary_dir, f"{i}_{hd.indicator}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing summary {path}. Run 'contextualize' first or set --summary-dir.")
        with open(path, "r", encoding="utf-8") as f:
            texts[i] = f.read()
    return texts


def _load_embeddings(hd: HealthcareData, encoder_root: str) -> dict:
    embs = _pkl(os.path.join(encoder_root, "embeddings", f"healthcare_{hd.indicator}.pkl"))
    return {hd.indices[_i]: embs[_i] for _i in range(len(hd.indices))}  # healthcare uses ALL indices


def build_units(
    task: str,
    hd: HealthcareData,
    *,
    summary_dir: Optional[str] = None,
    encoder_root: Optional[str] = None,
    k: int = 5,
) -> Iterator[Unit]:
    ind = hd.indicator
    if task == "contextualize":
        for i in hd.indices:
            sys, usr = _contextualize_prompt(hd, i)
            yield Unit(i, sys, usr, f"gpt_summary/{i}_{ind}.txt")
    elif task == "predict_time":
        for _i in hd.splits["test"]:
            i = hd.indices[_i]
            sys, usr = _predict_time_prompt(hd, i)
            yield Unit(i, sys, usr, f"gpt_predict_time/{i}_{ind}.txt")
    elif task == "predict_text":
        texts = _load_summaries(hd, summary_dir or os.path.join(hd.data_dir, "gpt_summary"))
        for _i in hd.splits["test"]:
            i = hd.indices[_i]
            sys, usr = _predict_text_prompt(hd, i, texts[i])
            yield Unit(i, sys, usr, f"gpt_predict_text/{i}_{ind}.txt")
    elif task == "predict_in_context":
        random.seed(2024)
        if not encoder_root:
            raise ValueError("predict_in_context needs --encoder-root (for retrieval embeddings).")
        texts = _load_summaries(hd, summary_dir or os.path.join(hd.data_dir, "gpt_summary"))
        text_emb = _load_embeddings(hd, encoder_root)
        for _i in hd.splits["test"]:
            i = hd.indices[_i]
            sys, usr = _predict_in_context_prompt(hd, i, texts, text_emb, k)
            yield Unit(i, sys, usr, f"gpt_predict_in-context/k{k}_{i}_{ind}_ref_int.txt")
    else:
        raise ValueError(f"Unknown task {task!r}. Healthcare tasks: {TASKS}")
