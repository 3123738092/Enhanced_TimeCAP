"""Anthropic Claude (native Messages API).

Differences from OpenAI handled here: ``system`` is a top-level parameter (not a
message), ``max_tokens`` is required, and the response ``content`` is a list of
blocks. Usage field names (input_tokens / output_tokens / cache_*) are
normalized later in :mod:`llm.usage_norm`.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

PROTOCOL = "anthropic"

_DEFAULT_MAX_TOKENS = 2048


def build_client(cfg):
    import anthropic  # lazy

    return anthropic.Anthropic(api_key=cfg.require_key())


def call(client, model: str, system: Optional[str], user: str, gen: dict) -> Any:
    gen = dict(gen)
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": gen.pop("max_tokens", _DEFAULT_MAX_TOKENS),
        "messages": [{"role": "user", "content": user}],
    }
    if system:
        params["system"] = system
    for k in ("temperature", "top_p", "top_k", "stop_sequences"):
        if k in gen:
            params[k] = gen[k]
    return client.messages.create(**params)


def extract(raw: Any) -> Tuple[str, Any, Optional[str]]:
    text = "".join(
        getattr(b, "text", "") for b in getattr(raw, "content", []) if getattr(b, "type", None) == "text"
    )
    usage = getattr(raw, "usage", None)
    finish = getattr(raw, "stop_reason", None)
    return text, usage, finish
