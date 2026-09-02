"""Google Gemini (native ``google-genai`` SDK).

Differences handled here: system prompt goes into ``system_instruction`` on the
config, ``max_tokens`` -> ``max_output_tokens``, and usage lives on
``usage_metadata`` (prompt/candidates/cached_content/thoughts token counts),
normalized later in :mod:`llm.usage_norm`.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

PROTOCOL = "gemini"


def build_client(cfg):
    from google import genai  # lazy

    return genai.Client(api_key=cfg.require_key())


def call(client, model: str, system: Optional[str], user: str, gen: dict) -> Any:
    from google.genai import types

    cfg_kwargs: dict[str, Any] = {}
    if system:
        cfg_kwargs["system_instruction"] = system
    if "temperature" in gen:
        cfg_kwargs["temperature"] = gen["temperature"]
    if "top_p" in gen:
        cfg_kwargs["top_p"] = gen["top_p"]
    if "max_tokens" in gen:
        cfg_kwargs["max_output_tokens"] = gen["max_tokens"]
    config = types.GenerateContentConfig(**cfg_kwargs)
    return client.models.generate_content(model=model, contents=user, config=config)


def extract(raw: Any) -> Tuple[str, Any, Optional[str]]:
    text = getattr(raw, "text", "") or ""
    usage = getattr(raw, "usage_metadata", None)
    finish = None
    try:
        finish = raw.candidates[0].finish_reason
    except Exception:  # noqa: BLE001 - finish_reason is best-effort metadata
        finish = None
    return text, usage, finish
