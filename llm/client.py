"""LLMClient — the one entry point the pipeline uses.

    llm = LLMClient(channel="siliconflow", model="deepseek-ai/DeepSeek-V3",
                    run_id="weather_ny_contextualize")
    res = llm.chat(system=sys, user=usr, temperature=0.7, max_tokens=2048,
                   meta={"task": "contextualize", "dataset": "weather_ny", "idx": 0})
    res.text        # response text
    res.usage       # normalized Usage
    res.cost_usd    # priced via the versioned table

Responsibilities: channel/adapter dispatch, retry, usage normalization (with a
local-estimate fallback), cost, and per-call JSONL logging. It does ONE call and
logs it. Resume/idempotency (the manifest) is owned by the pipeline runner, so
"done" is only recorded after the output is safely persisted. SDK clients are
built lazily so importing this module needs no API keys or provider SDKs.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from .adapters import get_adapter
from .config import resolve_channel
from .estimate import estimate_usage
from .models import derive_vendor
from .pricing import compute_cost
from .retry import retry_call
from .tokenlog import TokenLogger
from .types import ChatResult
from .usage_norm import normalize


class LLMClient:
    def __init__(
        self,
        channel: str,
        model: str,
        run_id: str,
        log_dir: str = "llm_runs",
        max_retries: int = 5,
        logger: Optional[TokenLogger] = None,
        adapter: Any = None,      # injectable for tests (a module-like object)
        sdk_client: Any = None,   # injectable for tests (skips build_client)
    ):
        self.cfg = resolve_channel(channel)
        self.channel = channel
        self.model = model
        self.vendor = derive_vendor(model)
        self.protocol = self.cfg.protocol
        self.adapter = adapter or get_adapter(self.protocol)
        self.run_id = run_id
        self.max_retries = max_retries
        self.logger = logger or TokenLogger(run_id, log_dir)
        self._sdk = sdk_client

    def ensure_sdk(self):
        """Build the SDK client up front (call once before concurrent use)."""
        if self._sdk is None:
            self._sdk = self.adapter.build_client(self.cfg)
        return self._sdk

    # -- main call -------------------------------------------------------------
    def chat(self, system: Optional[str], user: str, meta: Optional[dict] = None, **gen) -> ChatResult:
        meta = meta or {}
        t0 = time.time()
        try:
            if self._sdk is None:  # inside try so auth/build errors are logged too
                self._sdk = self.adapter.build_client(self.cfg)
            raw, attempts = retry_call(
                lambda: self.adapter.call(self._sdk, self.model, system, user, gen),
                max_retries=self.max_retries,
            )
            extracted = self.adapter.extract(raw)
            text, raw_usage, finish = extracted[0], extracted[1], extracted[2]
            reasoning = extracted[3] if len(extracted) > 3 else None  # thinking text, if any
            if raw_usage is None:
                prompt_text = f"{system}\n{user}" if system else user
                # include the model's reasoning/thinking text so reasoning models
                # are not under-counted on the local-estimate fallback path
                completion_text = f"{text}\n{reasoning}" if reasoning else text
                usage = estimate_usage(prompt_text, completion_text, self.model)
            else:
                usage = normalize(self.protocol, raw_usage)
            cost, snap, fx = compute_cost(usage, self.model)
            result = ChatResult(
                text=text, usage=usage, channel=self.channel, model=self.model,
                vendor=self.vendor, latency_s=time.time() - t0, finish_reason=finish,
                attempts=attempts, cost_usd=cost, price_snapshot_id=snap, fx_rate=fx, raw=raw,
            )
            self._log(result, meta, ok=True, error=None)
            return result
        except Exception as exc:  # log the failure, then propagate
            attempts = getattr(exc, "_llm_attempts", 1)  # real count from retry_call; 1 for pre-call errors
            self._log_failure(meta, time.time() - t0, repr(exc), attempts)
            raise

    # -- logging ---------------------------------------------------------------
    def _log(self, result: ChatResult, meta: dict, ok: bool, error: Optional[str]) -> None:
        u = result.usage
        self.logger.log({
            "channel": self.channel, "model": self.model, "vendor": self.vendor,
            "task": meta.get("task"), "dataset": meta.get("dataset"), "idx": meta.get("idx"),
            "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
            "reasoning_tokens": u.reasoning_tokens, "cached_read_tokens": u.cached_read_tokens,
            "cache_write_tokens": u.cache_write_tokens, "total_tokens": u.total_tokens,
            "token_source": u.token_source, "price_snapshot_id": result.price_snapshot_id,
            "fx_rate": result.fx_rate, "cost_usd": result.cost_usd,
            "latency_s": round(result.latency_s, 4), "attempt": result.attempts,
            "ok": ok, "error": error,
        })

    def _log_failure(self, meta: dict, latency: float, error: str, attempts: int = 1) -> None:
        self.logger.log({
            "channel": self.channel, "model": self.model, "vendor": self.vendor,
            "task": meta.get("task"), "dataset": meta.get("dataset"), "idx": meta.get("idx"),
            "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
            "cached_read_tokens": 0, "cache_write_tokens": 0, "total_tokens": 0,
            "token_source": "none", "price_snapshot_id": None, "fx_rate": None,
            "cost_usd": None, "latency_s": round(latency, 4),
            "attempt": attempts, "ok": False, "error": error,
        })
