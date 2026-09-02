"""Model-string helpers: derive the vendor and a canonical pricing key.

The same model can be reached through different channels (e.g. DeepSeek via
SiliconFlow vs via DeepSeek's own API), so ``model`` and ``channel`` are kept
separate. ``vendor`` is derived here purely from the model string for grouping
in analysis.
"""
from __future__ import annotations

# Prefix (before "/") on aggregator model ids -> canonical vendor.
_PREFIX_VENDOR = {
    "deepseek-ai": "deepseek",
    "moonshotai": "moonshot",
    "qwen": "qwen",
    "zai-org": "zhipu",
    "thudm": "zhipu",
    "minimaxai": "minimax",
    "meta-llama": "meta",
    "google": "google",
    "01-ai": "yi",
    "internlm": "internlm",
    "mistralai": "mistral",
}

# Fallback: match on the bare model name for non-prefixed ids.
_NAME_VENDOR_PREFIXES = (
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("chatgpt", "openai"),
    ("claude", "anthropic"),
    ("gemini", "google"),
    ("deepseek", "deepseek"),
    ("kimi", "moonshot"),
    ("qwen", "qwen"),
    ("glm", "zhipu"),
    ("minimax", "minimax"),
    ("llama", "meta"),
)

# SiliconFlow tier segments that are not the vendor.
_TIER_SEGMENTS = {"pro", "lora"}


def derive_vendor(model: str) -> str:
    """Best-effort vendor label from a model id. Returns 'unknown' if unsure."""
    if not model:
        return "unknown"
    m = model.strip()
    if "/" in m:
        parts = [p for p in m.split("/") if p]
        # Skip a leading tier segment like "Pro/deepseek-ai/DeepSeek-V3".
        head = parts[0].lower()
        if head in _TIER_SEGMENTS and len(parts) > 1:
            head = parts[1].lower()
        mapped = _PREFIX_VENDOR.get(head)
        if mapped:
            return mapped
    low = m.lower()
    for prefix, vendor in _NAME_VENDOR_PREFIXES:
        if low.startswith(prefix) or f"/{prefix}" in low:
            return vendor
    return "unknown"


def canonical_model_key(model: str) -> str:
    """Lowercased key used to look up the price table.

    Strips vendor/tier prefixes so "Pro/deepseek-ai/DeepSeek-V3" and
    "deepseek-ai/DeepSeek-V3" resolve to the same "deepseek-v3".
    """
    if not model:
        return ""
    tail = model.strip().split("/")[-1]
    return tail.lower()
