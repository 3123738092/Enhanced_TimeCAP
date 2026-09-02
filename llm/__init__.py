"""llm — a small, robust, multi-channel LLM calling layer with token/cost accounting.

Public surface:
    LLMClient           - make calls, auto-logs tokens/cost, resumable
    ChatResult, Usage   - normalized result/usage types
    TokenLogger, summarize - per-call JSONL log + aggregation
    compute_cost, PRICE_TABLE - versioned pricing
"""
from __future__ import annotations

from .client import LLMClient
from .config import CHANNELS, resolve_channel
from .pricing import PRICE_SNAPSHOT_ID, PRICE_TABLE, compute_cost
from .tokenlog import TokenLogger, summarize
from .types import ChatResult, Usage

__all__ = [
    "LLMClient",
    "ChatResult",
    "Usage",
    "TokenLogger",
    "summarize",
    "compute_cost",
    "PRICE_TABLE",
    "PRICE_SNAPSHOT_ID",
    "CHANNELS",
    "resolve_channel",
]
