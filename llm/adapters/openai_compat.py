"""OpenAI + OpenAI-compatible channels (SiliconFlow, DeepSeek, Moonshot, vLLM...).

The only per-channel difference is ``base_url`` + ``api_key`` (from the channel
config). ``gen`` keys map 1:1 onto the OpenAI Chat Completions parameters.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

PROTOCOL = "openai"


def build_client(cfg):
    from openai import OpenAI  # lazy: only needed to make real calls

    # timeout so a hung/dead connection (e.g. after the machine sleeps) fails and
    # is retried by our own retry layer, instead of freezing forever.
    # max_retries=0 -> let llm.retry own the retry policy (no double retries).
    return OpenAI(api_key=cfg.require_key(), base_url=cfg.base_url, timeout=120.0, max_retries=0)


def _wants_max_completion_tokens(model: str) -> bool:
    """OpenAI's gpt-5 family and o-series reject ``max_tokens`` and require
    ``max_completion_tokens``. Older models (gpt-4o*) and OpenAI-compatible
    gateways (SiliconFlow/DeepSeek/Moonshot) still use ``max_tokens``."""
    m = model.lower().rsplit("/", 1)[-1]
    return m.startswith("gpt-5") or m.startswith(("o1", "o3", "o4"))


def call(client, model: str, system: Optional[str], user: str, gen: dict) -> Any:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    params = dict(gen)
    if _wants_max_completion_tokens(model):
        # gpt-5 / o-series: renamed token param, and only the default
        # temperature (1) / top_p are accepted -> drop custom sampling values.
        if "max_tokens" in params:
            params["max_completion_tokens"] = params.pop("max_tokens")
        params.pop("temperature", None)
        params.pop("top_p", None)
    return client.chat.completions.create(model=model, messages=messages, **params)


def extract(raw: Any) -> Tuple[str, Any, Optional[str], Optional[str]]:
    choice = raw.choices[0]
    text = choice.message.content or ""
    # DeepSeek-R1 (and some gateways for o-series) return the chain-of-thought in
    # a separate ``reasoning_content`` field. Surface it (4th element) instead of
    # dropping it; the client uses it in the estimate fallback for reasoning
    # models. It is intentionally NOT concatenated into ``text`` (the answer).
    reasoning = getattr(choice.message, "reasoning_content", None)
    finish = getattr(choice, "finish_reason", None)
    usage = getattr(raw, "usage", None)
    return text, usage, finish, reasoning
