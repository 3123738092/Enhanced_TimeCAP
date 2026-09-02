"""Domain registry.

Each domain module exposes: NAME, DATASETS (dataset id -> short code), TASKS,
load(dataset, data_root) -> data, and build_units(task, data, ...) -> Iterator[Unit].
"""
from __future__ import annotations

from . import finance, healthcare, weather

DOMAINS = {weather.NAME: weather, finance.NAME: finance, healthcare.NAME: healthcare}

# dataset id (e.g. "weather_ny") -> domain module
DATASET_TO_DOMAIN = {}
for _mod in DOMAINS.values():
    for _ds in _mod.DATASETS:
        DATASET_TO_DOMAIN[_ds] = _mod


def get_domain(name: str):
    try:
        return DOMAINS[name]
    except KeyError:
        raise ValueError(f"Unknown domain {name!r}. Known: {sorted(DOMAINS)}")


def domain_for_dataset(dataset: str):
    try:
        return DATASET_TO_DOMAIN[dataset]
    except KeyError:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Known datasets: {sorted(DATASET_TO_DOMAIN)}"
        )
