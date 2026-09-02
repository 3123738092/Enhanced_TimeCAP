"""I/O + integration tests: token log, manifest resume, and LLMClient end-to-end.

The client is driven with a FAKE adapter (a SimpleNamespace exposing
build_client/call/extract) and an injected sdk client, so no network / keys /
SDKs are needed. All files are written under a throwaway TemporaryDirectory.
"""
from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace as NS

from llm.client import LLMClient
from llm.manifest import Manifest
from llm.tokenlog import TokenLogger


def _fake_adapter(raw=None, text="ok", usage=None, raise_exc=None):
    def call(client, model, system, user, gen):
        if raise_exc is not None:
            raise raise_exc
        return raw if raw is not None else NS()

    return NS(
        build_client=lambda cfg: object(),
        call=call,
        extract=lambda r: (text, usage, "stop"),
    )


def _openai_usage(prompt, completion, cached=0):
    return NS(
        prompt_tokens=prompt, completion_tokens=completion,
        prompt_tokens_details=NS(cached_tokens=cached),
        completion_tokens_details=None,
    )


class TestTokenLogger(unittest.TestCase):
    def test_log_and_summarize(self):
        with tempfile.TemporaryDirectory() as d:
            log = TokenLogger("run1", log_dir=d)
            log.log({"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                     "cost_usd": 0.001, "ok": True})
            log.log({"input_tokens": 20, "output_tokens": 10, "total_tokens": 30,
                     "cost_usd": 0.002, "ok": True})
            rows = log.read()
            self.assertEqual(len(rows), 2)
            s = log.summary()
            self.assertEqual(s["calls"], 2)
            self.assertEqual(s["total_tokens"], 45)
            self.assertAlmostEqual(s["cost_usd_total"], 0.003, places=6)
            self.assertEqual(s["failed_calls"], 0)


class TestManifest(unittest.TestCase):
    def test_idempotent_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            m = Manifest("run1", manifest_dir=d)
            self.assertFalse(m.is_done("contextualize", "weather_ny", 0))
            m.mark_done("contextualize", "weather_ny", 0)
            m.mark_done("contextualize", "weather_ny", 0)  # idempotent
            self.assertTrue(m.is_done("contextualize", "weather_ny", 0))
            self.assertEqual(len(m), 1)
            # a fresh instance (simulating a restart) reloads completed keys
            m2 = Manifest("run1", manifest_dir=d)
            self.assertTrue(m2.is_done("contextualize", "weather_ny", 0))


class TestClientEndToEnd(unittest.TestCase):
    def _client(self, d, adapter):
        return LLMClient(channel="openai", model="gpt-4o", run_id="r",
                         log_dir=d, adapter=adapter, sdk_client=object())

    def test_happy_path_logs_costs_and_marks_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            adapter = _fake_adapter(usage=_openai_usage(100, 50, cached=0), text="report")
            c = self._client(d, adapter)
            res = c.chat("sys", "user", meta={"task": "contextualize", "dataset": "weather_ny", "idx": 0},
                         temperature=0.7, max_tokens=64)
            self.assertEqual(res.text, "report")
            self.assertEqual(res.usage.input_tokens, 100)
            self.assertEqual(res.usage.total_tokens, 150)
            self.assertIsNotNone(res.cost_usd)          # gpt-4o is in the table
            rows = c.logger.read()
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["ok"])
            self.assertEqual(rows[0]["total_tokens"], 150)

    def test_estimate_fallback_when_no_usage(self):
        with tempfile.TemporaryDirectory() as d:
            adapter = _fake_adapter(usage=None, text="hello world hello")
            c = self._client(d, adapter)
            res = c.chat("sys", "the prompt", meta={"task": "t", "dataset": "ds", "idx": 1})
            self.assertEqual(res.usage.token_source, "estimated")
            self.assertGreater(res.usage.input_tokens, 0)
            self.assertGreater(res.usage.output_tokens, 0)

    def test_failure_is_logged_and_reraised(self):
        with tempfile.TemporaryDirectory() as d:
            adapter = _fake_adapter(raise_exc=ValueError("boom"))  # non-retryable
            c = self._client(d, adapter)
            with self.assertRaises(ValueError):
                c.chat("sys", "user", meta={"task": "t", "dataset": "ds", "idx": 2})
            rows = c.logger.read()
            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0]["ok"])
            self.assertIn("boom", rows[0]["error"])


if __name__ == "__main__":
    unittest.main()
