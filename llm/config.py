"""Channel registry: routing + auth + protocol for each endpoint.

A *channel* is where the request goes (base_url + which API key + which wire
protocol). It is deliberately separate from *model* and *vendor*: the same
model (e.g. DeepSeek-V3) may be reached through several channels (SiliconFlow,
DeepSeek's own API, ...), and we want to compare them.

Keys are read from environment variables — never hardcoded. A channel can be
constructed without a key present (so the code builds fine); the key is only
required when an actual call is made (see :meth:`require_key`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

PROTOCOLS = {"openai", "anthropic", "gemini"}


@dataclass(frozen=True)
class ChannelConfig:
    name: str
    protocol: str            # "openai" | "anthropic" | "gemini"
    key_env: str             # environment variable holding the API key
    base_url: Optional[str] = None  # None -> the SDK's default endpoint

    @property
    def api_key(self) -> Optional[str]:
        return os.environ.get(self.key_env)

    def require_key(self) -> str:
        key = self.api_key
        if not key:
            raise RuntimeError(
                f"Channel '{self.name}' needs an API key: set the environment "
                f"variable {self.key_env} (e.g. `export {self.key_env}=...`)."
            )
        return key


# All OpenAI-compatible channels share one client implementation; they differ
# only by base_url + key_env. Anthropic / Gemini use their native adapters.
CHANNELS: dict[str, ChannelConfig] = {
    "openai":      ChannelConfig("openai", "openai", "OPENAI_API_KEY", None),
    "siliconflow": ChannelConfig("siliconflow", "openai", "SILICONFLOW_API_KEY", "https://api.siliconflow.cn/v1"),
    "deepseek":    ChannelConfig("deepseek", "openai", "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1"),
    "moonshot":    ChannelConfig("moonshot", "openai", "MOONSHOT_API_KEY", "https://api.moonshot.cn/v1"),
    "anthropic":   ChannelConfig("anthropic", "anthropic", "ANTHROPIC_API_KEY", None),
    "gemini":      ChannelConfig("gemini", "gemini", "GEMINI_API_KEY", None),
}


def resolve_channel(name: str) -> ChannelConfig:
    try:
        return CHANNELS[name]
    except KeyError:
        raise ValueError(
            f"Unknown channel {name!r}. Known channels: {sorted(CHANNELS)}"
        )
