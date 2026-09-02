"""Versioned price table + the single, uniform cost formula.

Because :mod:`llm.usage_norm` already splits cached tokens out of the fresh
input for every provider, ONE formula prices all channels:

    cost = (input * price_in
            + cached_read * price_cached_in
            + cache_write * price_cache_write
            + output * price_out) / 1e6

Notes / rules from the spec:
- Prices are USD per 1,000,000 tokens.
- ``output`` already includes reasoning tokens -> reasoning is NOT priced again.
- Prices change: every entry carries ``as_of`` + ``source``; the table has a
  ``PRICE_SNAPSHOT_ID`` recorded on every logged call so old runs stay reproducible.
- The numbers below were cross-checked against 2026-08 pricing aggregators
  (devtk.ai, aipricing.guru, finout.io, metacto.com, benchlm.ai; SiliconFlow via
  Artificial Analysis). Aggregators can lag/disagree — confirm on each vendor's
  official pricing page and bump ``as_of`` before publishing a paper.
- CNY-priced (SiliconFlow) entries were pre-converted at fx_usd_per_cny; the
  same rate is echoed back so each logged cost is reproducible.
"""
from __future__ import annotations

from typing import Optional, Tuple

from .models import canonical_model_key
from .types import Usage

PRICE_SNAPSHOT_ID = "2026-08-14"

_FX_USD_PER_CNY = 0.14837  # 1 USD = 6.74 CNY, verified 2026-08-14 (market consensus; corrected from stale 7.15 in spec doc §6.4)

# key -> USD per 1M tokens. cached_in/cache_write optional (defaults applied below).
# Verified against 2026-08 pricing aggregators (still confirm on official pages
# before publishing). SiliconFlow rows are USD-equivalent from Artificial Analysis.
PRICE_TABLE = {
    # ---- Anthropic (aipricing.guru / finout.io, 2026-08) ----
    "claude-opus-4-8":   {"in": 5.00, "out": 25.00, "cached_in": 0.50, "cache_write": 6.25},
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00, "cached_in": 0.30, "cache_write": 3.75},
    "claude-haiku-4-5":  {"in": 1.00, "out": 5.00,  "cached_in": 0.10, "cache_write": 1.25},
    "claude-fable-5":    {"in": 10.00, "out": 50.00, "cached_in": 1.00, "cache_write": 12.50},
    # ---- OpenAI (devtk.ai / aipricing.guru, 2026-08) ----
    "gpt-5.6-sol":   {"in": 5.00, "out": 30.00, "cached_in": 0.50},
    "gpt-5.6-terra": {"in": 2.50, "out": 15.00, "cached_in": 0.25},
    "gpt-5.6-luna":  {"in": 1.00, "out": 6.00,  "cached_in": 0.10},
    "gpt-4o":        {"in": 2.50, "out": 10.00, "cached_in": 1.25},
    "gpt-4o-mini":   {"in": 0.15, "out": 0.60,  "cached_in": 0.075},
    "gpt-4.1-nano":  {"in": 0.10, "out": 0.40},
    # gpt-5 family (OpenAI official pricing page, verified 2026-09-02; cached input = 10% of input)
    "gpt-5":         {"in": 1.25, "out": 10.00, "cached_in": 0.125},
    "gpt-5-mini":    {"in": 0.25, "out": 2.00,  "cached_in": 0.025},
    # ---- Google Gemini (metacto.com / benchlm.ai, 2026-08) ----
    # cached_in ≈ 25% of input (Gemini implicit context-cache discount); confirm
    # on the official pricing page before publishing.
    "gemini-3.1-pro":        {"in": 2.00, "out": 12.00, "cached_in": 0.50},
    "gemini-3.6-flash":      {"in": 1.50, "out": 7.50,  "cached_in": 0.375},
    "gemini-2.5-flash-lite": {"in": 0.10, "out": 0.40,  "cached_in": 0.025},
    # ---- SiliconFlow, USD-equivalent (Artificial Analysis, 2026-08) ----
    "deepseek-v4-pro":               {"in": 1.74, "out": 3.48, "cached_in": 0.145},
    "deepseek-v4-flash":             {"in": 0.14, "out": 0.28, "cached_in": 0.028},
    "qwen3-235b-a22b-instruct-2507": {"in": 0.09, "out": 0.60},
    "kimi-k2-instruct":              {"in": 0.58, "out": 2.29},
    # ---- SiliconFlow CNY-sourced (¥ price x _FX_USD_PER_CNY=1/6.74; unit prices confirmed on SiliconFlow 2026-08) ----
    "deepseek-v3.2": {"in": 0.5935, "out": 0.8902, "fx_usd_per_cny": _FX_USD_PER_CNY},  # ¥4 / ¥6
    "qwen3.6-27b":   {"in": 0.4451, "out": 2.6706, "fx_usd_per_cny": _FX_USD_PER_CNY},  # ¥3 / ¥18
    "kimi-k2.6":     {"in": 0.9644, "out": 4.0059, "fx_usd_per_cny": _FX_USD_PER_CNY},  # ¥6.5 / ¥27
    "qwen2.5-72b-instruct": {"in": 0.6128, "out": 0.6128, "fx_usd_per_cny": _FX_USD_PER_CNY},  # ¥4.13 unified
    "deepseek-v3":   {"in": 0.2967, "out": 1.1869, "fx_usd_per_cny": _FX_USD_PER_CNY},  # ¥2 / ¥8
    "deepseek-r1":   {"in": 0.5935, "out": 2.3739, "fx_usd_per_cny": _FX_USD_PER_CNY},  # ¥4 / ¥16
}


def lookup(model: str) -> Optional[dict]:
    """Find the price entry for a model id (vendor/tier prefix tolerant)."""
    return PRICE_TABLE.get(canonical_model_key(model))


def compute_cost(usage: Usage, model: str) -> Tuple[Optional[float], str, Optional[float]]:
    """Return (cost_usd, price_snapshot_id, fx_rate).

    cost_usd is ``None`` when the model is not in the table (the call is still
    logged with raw token counts, so cost can be back-filled later).
    """
    entry = lookup(model)
    if entry is None:
        return None, PRICE_SNAPSHOT_ID, None
    p_in = entry["in"]
    p_out = entry["out"]
    p_cached = entry.get("cached_in", p_in)          # default: no discount known
    p_write = entry.get("cache_write", p_in * 1.25)  # Anthropic-style default
    cost = (
        usage.input_tokens * p_in
        + usage.cached_read_tokens * p_cached
        + usage.cache_write_tokens * p_write
        + usage.output_tokens * p_out
    ) / 1_000_000.0
    return cost, PRICE_SNAPSHOT_ID, entry.get("fx_usd_per_cny")
