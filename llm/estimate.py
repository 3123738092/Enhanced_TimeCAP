"""Local token estimation — the fallback when a provider omits ``usage``.

Primary accounting always uses the API-reported ``usage`` (method A). This
module is method B: a client-side estimate, used ONLY when the response has no
usage, and always tagged ``token_source="estimated"`` so estimates are never
silently trusted as ground truth.

``tiktoken`` is used when available (best for OpenAI-family models); otherwise
we fall back to a coarse character heuristic. Either way it is an *estimate*.
"""
from __future__ import annotations

from typing import Optional

from .types import Usage

# Rough heuristic: ~4 characters per token for English-ish text.
_CHARS_PER_TOKEN = 4.0


def count_tokens(text: str, model: Optional[str] = None) -> int:
    """Best-effort token count for a single string."""
    if not text:
        return 0
    try:
        import tiktoken  # lazy: not required unless a provider omits usage

        try:
            enc = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, round(len(text) / _CHARS_PER_TOKEN))


def estimate_usage(prompt_text: str, completion_text: str, model: Optional[str] = None) -> Usage:
    """Build an ``estimated`` Usage from the raw prompt and completion strings."""
    pin = count_tokens(prompt_text, model)
    out = count_tokens(completion_text, model)
    return Usage(
        input_tokens=pin,
        output_tokens=out,
        reasoning_tokens=0,
        cached_read_tokens=0,
        cache_write_tokens=0,
        total_tokens=pin + out,
        token_source="estimated",
    )
