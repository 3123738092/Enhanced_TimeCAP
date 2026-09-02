"""Small, SDK-agnostic retry-with-backoff helper.

Only *transient* failures are retried (rate limits, timeouts, 5xx, connection
resets). Deterministic failures (e.g. 401 invalid key, 400 bad request) are
raised immediately — retrying them just wastes time and quota. ``sleep`` is
injectable so tests run instantly.
"""
from __future__ import annotations

import time
from typing import Callable, Tuple, TypeVar

T = TypeVar("T")

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_RETRYABLE_NAME_HINTS = ("timeout", "connection", "ratelimit", "apiconnection", "overloaded", "unavailable")


def is_retryable(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int) and status in _RETRYABLE_STATUS:
        return True
    name = type(exc).__name__.lower()
    return any(hint in name for hint in _RETRYABLE_NAME_HINTS)


def retry_call(
    fn: Callable[[], T],
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable: Callable[[BaseException], bool] = is_retryable,
    sleep: Callable[[float], None] = time.sleep,
) -> Tuple[T, int]:
    """Call ``fn`` with exponential backoff. Returns (result, attempts_used)."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn(), attempt
        except BaseException as exc:  # noqa: BLE001 - re-raised below when appropriate
            if attempt > max_retries or not retryable(exc):
                setattr(exc, "_llm_attempts", attempt)  # let the caller log the real count
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            sleep(delay)
