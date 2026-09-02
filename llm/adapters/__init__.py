"""Protocol adapters.

Each adapter exposes the same tiny surface so :class:`llm.client.LLMClient`
stays vendor-agnostic:

    PROTOCOL: str
    build_client(cfg)                 -> sdk client        (lazy SDK import)
    call(client, model, system, user, gen) -> raw response (the network call)
    extract(raw)                      -> (text, raw_usage, finish_reason)

``raw_usage`` is the provider's own usage object; normalization into the
canonical :class:`llm.types.Usage` happens centrally in :mod:`llm.usage_norm`.
"""
from __future__ import annotations

from . import anthropic_native, gemini_native, openai_compat

_ADAPTERS = {
    openai_compat.PROTOCOL: openai_compat,
    anthropic_native.PROTOCOL: anthropic_native,
    gemini_native.PROTOCOL: gemini_native,
}


def get_adapter(protocol: str):
    try:
        return _ADAPTERS[protocol]
    except KeyError:
        raise ValueError(f"No adapter for protocol {protocol!r}. Known: {sorted(_ADAPTERS)}")
