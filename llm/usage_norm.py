"""Normalize each provider's ``usage`` object into the canonical :class:`Usage`.

The whole point of this module is to absorb the two hard rules from the spec:

1. **cached-token attribution differs per vendor**
   - OpenAI-compatible: ``prompt_tokens`` INCLUDES cached -> subtract it out.
   - Anthropic:        ``input_tokens`` EXCLUDES cache; read/write are separate.
   - Gemini:           ``prompt_token_count`` INCLUDES cached -> subtract it out.
2. **reasoning tokens are never billed twice** -> they stay inside
   ``output_tokens`` and are only *also* surfaced as ``reasoning_tokens`` for
   analysis.

Every accessor is null-safe: aggregator/compatible endpoints frequently omit
the ``*_details`` fields (or return ``None``), and that must not crash.
"""
from __future__ import annotations

from typing import Any

from .types import Usage


def _get(obj: Any, key: str, default: int = 0) -> int:
    """Read ``key`` from an object attribute OR a dict, coercing None/missing to default."""
    if obj is None:
        return default
    val = obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _sub(obj: Any, key: str) -> Any:
    """Read a possibly-nested sub-object (e.g. prompt_tokens_details)."""
    if obj is None:
        return None
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def normalize_openai(usage: Any) -> Usage:
    """OpenAI & OpenAI-compatible (SiliconFlow, DeepSeek, Moonshot, vLLM, ...)."""
    prompt = _get(usage, "prompt_tokens")
    completion = _get(usage, "completion_tokens")
    cached = _get(_sub(usage, "prompt_tokens_details"), "cached_tokens")
    if cached == 0:
        # DeepSeek's native API (api.deepseek.com) reports cache hits at the TOP
        # level as ``prompt_cache_hit_tokens`` instead of inside
        # ``prompt_tokens_details`` (the OpenAI standard). Without this fallback
        # DeepSeek cache hits are billed at the full input rate.
        cached = _get(usage, "prompt_cache_hit_tokens")
    reasoning = _get(_sub(usage, "completion_tokens_details"), "reasoning_tokens")
    fresh_in = max(prompt - cached, 0)
    return Usage(
        input_tokens=fresh_in,
        output_tokens=completion,
        reasoning_tokens=reasoning,
        cached_read_tokens=cached,
        cache_write_tokens=0,
        total_tokens=fresh_in + cached + completion,
        token_source="api",
    )


def normalize_anthropic(usage: Any) -> Usage:
    """Anthropic Messages API: input_tokens already excludes cache."""
    fresh_in = _get(usage, "input_tokens")
    output = _get(usage, "output_tokens")
    cache_read = _get(usage, "cache_read_input_tokens")
    cache_write = _get(usage, "cache_creation_input_tokens")
    return Usage(
        input_tokens=fresh_in,
        output_tokens=output,
        reasoning_tokens=0,  # Anthropic bills thinking as output; no separate field
        cached_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        total_tokens=fresh_in + cache_read + cache_write + output,
        token_source="api",
    )


def normalize_gemini(usage: Any) -> Usage:
    """Google Gemini: prompt_token_count includes cache; thoughts are billed as output."""
    prompt = _get(usage, "prompt_token_count")
    candidates = _get(usage, "candidates_token_count")
    cached = _get(usage, "cached_content_token_count")
    thoughts = _get(usage, "thoughts_token_count")
    fresh_in = max(prompt - cached, 0)
    output = candidates + thoughts  # thinking tokens are billed at the output rate
    return Usage(
        input_tokens=fresh_in,
        output_tokens=output,
        reasoning_tokens=thoughts,
        cached_read_tokens=cached,
        cache_write_tokens=0,
        total_tokens=fresh_in + cached + output,
        token_source="api",
    )


NORMALIZERS = {
    "openai": normalize_openai,
    "anthropic": normalize_anthropic,
    "gemini": normalize_gemini,
}


def normalize(protocol: str, usage: Any) -> Usage:
    try:
        return NORMALIZERS[protocol](usage)
    except KeyError:
        raise ValueError(f"Unknown protocol for usage normalization: {protocol!r}")
