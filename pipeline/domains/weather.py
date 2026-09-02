"""Weather domain — faithful port of the four weather notebooks.

Datasets: weather_ny / weather_sf / weather_hs (New York / San Francisco / Houston).
Time series columns: [humidity, pressure, temperature, wind_speed, wind_direction].
Window = 24 hours. Label = rain / not-rain in the next 24h (binary).

Output filename conventions match the notebooks so results are comparable:
  contextualize      -> gpt_summary/<city>_<i>.txt
  predict_time       -> gpt_predict/<city>_<i>.txt
  predict_text       -> gpt_predict_text/<city>_<i>.txt
  predict_in_context -> gpt_predict_in-context/<city>_k<k>_<i>_ref.txt

Bugfix vs the original notebook: the in-context notebook used ``norm`` and
``random`` without importing them (a latent NameError); both are imported here.
"""
from __future__ import annotations

import os
import pickle as pkl
import random
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from ..base import Unit, compute_splits, fmt_series, retrieve_topk

NAME = "weather"
WINDOW = 24
DATASETS = {"weather_ny": "ny", "weather_sf": "sf", "weather_hs": "hs"}
CITY_FULL = {"ny": "New York City", "hs": "Houston", "sf": "San Francisco"}
TASKS = ("contextualize", "predict_time", "predict_text", "predict_in_context")

_SYS_ANALYST = ("Your job is to act as a professional weather analyst. You will write a "
                "high-quality report that is informative and helps in understanding the "
                "current weather situation.")
_SYS_FORECAST_TS = ("Your job is to act as a professional weather forecaster. You will be given "
                    "a time-series data of the weather from the past 24 hours. Based on this "
                    "information, your task is to predict whether it will rain in the next 24 hours.")
_SYS_FORECAST_SUM = ("Your job is to act as a professional weather forecaster. You will be given "
                     "a summary of the weather from the past 24 hours. Based on this information, "
                     "your task is to predict whether it will rain in the next 24 hours.")


@dataclass
class WeatherData:
    dataset: str
    city: str
    data: np.ndarray
    indices: list
    rain: np.ndarray
    splits: dict
    data_dir: str


def _pkl(path: str):
    with open(path, "rb") as f:
        return pkl.load(f)


def load(dataset: str, data_root: str) -> WeatherData:
    city = DATASETS[dataset]
    ddir = os.path.join(data_root, "weather")
    indices = list(_pkl(os.path.join(ddir, "indices.pkl")))
    data = _pkl(os.path.join(ddir, f"time_series_{city}.pkl"))
    rain = np.asarray(_pkl(os.path.join(ddir, f"rain_{city}.pkl")))
    splits = compute_splits(len(indices), seq_len_day=1)
    return WeatherData(dataset, city, data, indices, rain, splits, ddir)


def _features(data, i: int) -> dict:
    w = data[i:i + WINDOW]
    return {
        "humidity": fmt_series(w[:, 0]),
        "pressure": fmt_series(w[:, 1]),
        "temperature": fmt_series(w[:, 2]),
        "wind_speed": fmt_series(w[:, 3]),
        "wind_direction": fmt_series(w[:, 4]),
    }


def _series_block(f: dict) -> str:
    return (f"- Temperature (Kelvin): {f['temperature']}\n"
            f"- Humidity (%): {f['humidity']}\n"
            f"- Air Pressure (hPa): {f['pressure']}\n"
            f"- Wind Speed (m/s): {f['wind_speed']}\n"
            f"- Wind Direction (degrees): {f['wind_direction']}\n\n")


def _contextualize_prompt(wd: WeatherData, i: int) -> tuple[str, str]:
    city = CITY_FULL[wd.city]
    f = _features(wd.data, i)
    u = f"Your task is to analyze key weather indicators in {city} over the last {WINDOW} hours."
    u += f"\n\nReview the time-series data provided for the last {WINDOW} hours. "
    u += "Each time-series consists of hourly values separated by a '|' token for the following indicators:\n"
    u += _series_block(f)
    u += "Based on this time-series data, write a concise report that provides insights crucial for understanding the current weather situation. "
    u += f"Your report should be limited to five sentences, yet comprehensive, highlighting key trends and considering their potential impact on the weather in {city}."
    u += "Do not write numerical values while writing the report."
    return _SYS_ANALYST, u


def _predict_time_prompt(wd: WeatherData, i: int) -> tuple[str, str]:
    city = CITY_FULL[wd.city]
    f = _features(wd.data, i)
    u = f"Your task is to predict whether it will rain or not in {city} in the next {WINDOW} hours. "
    u += f"Review the time-series data provided for the last {WINDOW} hours. "
    u += "Each time-series consists of hourly values separated by a '|' token for the following indicators:\n\n"
    u += _series_block(f)
    u += "Based on this information, respond with either 'rain' or 'not rain'. Do not provide any other details. "
    return _SYS_FORECAST_TS, u


def _predict_text_prompt(wd: WeatherData, i: int, summary: str) -> tuple[str, str]:
    city = CITY_FULL[wd.city]
    u = f"Your task is to predict whether it will rain or not in {city} in the next {WINDOW} hours."
    u += " The weather of the past 24 hours is summarized as follows:\n\n"
    u += f"{summary}\n\n"
    u += "Based on this information, respond with either 'rain' or 'not rain'. Do not provide any other details. "
    return _SYS_FORECAST_SUM, u


def _predict_in_context_prompt(wd: WeatherData, i: int, texts: dict, text_emb: dict, k: int) -> tuple[str, str]:
    city = CITY_FULL[wd.city]
    idx_train = wd.splits["train"]
    u = f"Your task is to predict whether it will rain or not in {city} in the next {WINDOW} hours. "
    u += f"First, review the following {k} examples of weather summaries and outcomes so that you can refer to when making predictions.\n\n"
    for _k, _j in enumerate(retrieve_topk(text_emb, i, idx_train, wd.indices, k)):
        j = wd.indices[_j]
        u += f"Summary #{_k + 1}: {texts[j]}"
        u += f"\nOutcome #{_k + 1}: It rained.\n\n" if wd.rain[_j + 1] else f"\nOutcome #{_k + 1}: It did not rain.\n\n"
    u += "The weather of the last 24 hours is summarized as follows:\n\n"
    u += f"Summary: {texts[i]}\n"
    u += "Outcome:\n\n"
    u += "Based on the understanding of the provided examples, predict the outcome of the current weather summary. "
    u += "Respond your prediction with either 'rain' or 'not rain'. "
    u += "Response should not include other terms."
    return _SYS_FORECAST_SUM, u


def _load_summaries(wd: WeatherData, summary_dir: str) -> dict:
    texts = {}
    for i in wd.indices:
        path = os.path.join(summary_dir, f"{wd.city}_{i}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing summary {path}. Run the 'contextualize' task first, or point "
                f"--summary-dir at an existing gpt_summary directory."
            )
        with open(path, "r", encoding="utf-8") as f:
            texts[i] = f.read()
    return texts


def _load_embeddings(wd: WeatherData, encoder_root: str) -> dict:
    # Weather embeddings have length len(indices) - 1, so indices[:-1] is correct
    # here (and the weather test split never reaches the last index).
    path = os.path.join(encoder_root, "embeddings", f"weather_{wd.city}.pkl")
    embs = _pkl(path)
    return {wd.indices[_i]: embs[_i] for _i in range(len(wd.indices) - 1)}


def build_units(
    task: str,
    wd: WeatherData,
    *,
    summary_dir: Optional[str] = None,
    encoder_root: Optional[str] = None,
    k: int = 5,
) -> Iterator[Unit]:
    city = wd.city
    if task == "contextualize":
        for i in wd.indices:
            sys, usr = _contextualize_prompt(wd, i)
            yield Unit(i, sys, usr, f"gpt_summary/{city}_{i}.txt")
    elif task == "predict_time":
        for _i in wd.splits["test"]:
            i = wd.indices[_i]
            sys, usr = _predict_time_prompt(wd, i)
            yield Unit(i, sys, usr, f"gpt_predict/{city}_{i}.txt")
    elif task == "predict_text":
        texts = _load_summaries(wd, summary_dir or os.path.join(wd.data_dir, "gpt_summary"))
        for _i in wd.splits["test"]:
            i = wd.indices[_i]
            sys, usr = _predict_text_prompt(wd, i, texts[i])
            yield Unit(i, sys, usr, f"gpt_predict_text/{city}_{i}.txt")
    elif task == "predict_in_context":
        random.seed(2024)  # match the notebook (retrieval itself is deterministic)
        if not encoder_root:
            raise ValueError("predict_in_context needs --encoder-root (for retrieval embeddings).")
        texts = _load_summaries(wd, summary_dir or os.path.join(wd.data_dir, "gpt_summary"))
        text_emb = _load_embeddings(wd, encoder_root)
        for _i in wd.splits["test"]:
            i = wd.indices[_i]
            sys, usr = _predict_in_context_prompt(wd, i, texts, text_emb, k)
            yield Unit(i, sys, usr, f"gpt_predict_in-context/{city}_k{k}_{i}_ref.txt")
    else:
        raise ValueError(f"Unknown task {task!r}. Weather tasks: {TASKS}")
