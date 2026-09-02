"""Canonical data structures shared across the llm package.

All adapters (OpenAI-compatible / Anthropic / Gemini) normalize their
provider-specific responses into these types, so the rest of the pipeline
never sees vendor differences.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass(frozen=True)
class Usage:
    """Normalized token usage for a single call.

    Fields follow the accounting rules agreed in the project spec
    (D:/桌面/Agent_TokenUsage_与USD成本_对比方法整理.md):

    - ``input_tokens``       : fresh (non-cached) input tokens, billed at price_in
    - ``cached_read_tokens`` : cached input read, billed at the discounted cached price
    - ``cache_write_tokens`` : cache-creation tokens (mainly Anthropic), billed at write price
    - ``output_tokens``      : generated tokens; ALREADY INCLUDES reasoning tokens
    - ``reasoning_tokens``   : subset of output, kept for ANALYSIS ONLY (never billed twice)
    - ``total_tokens``       : input + cached_read + cache_write + output
    - ``token_source``       : "api" (from response.usage) or "estimated" (local tokenizer)
    """

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    token_source: str = "api"

    def identity_ok(self) -> bool:
        """input + cached_read + cache_write + output should equal total."""
        return (
            self.input_tokens
            + self.cached_read_tokens
            + self.cache_write_tokens
            + self.output_tokens
        ) == self.total_tokens

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChatResult:
    """Everything the pipeline needs from one LLM call."""

    text: str
    usage: Usage
    channel: str
    model: str
    vendor: str
    latency_s: float
    finish_reason: Optional[str] = None
    attempts: int = 1
    cost_usd: Optional[float] = None
    price_snapshot_id: Optional[str] = None
    fx_rate: Optional[float] = None
    raw: Any = field(default=None, repr=False)
