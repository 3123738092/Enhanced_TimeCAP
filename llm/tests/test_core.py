"""Pure-logic tests: usage normalization, pricing, retry, vendor derivation.

No network, no keys, no SDKs — provider responses are faked with SimpleNamespace.
Run:  python -m unittest discover -s llm/tests -t .
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace as NS

from llm import compute_cost
from llm.models import canonical_model_key, derive_vendor
from llm.retry import retry_call
from llm.types import Usage
from llm.usage_norm import normalize_anthropic, normalize_gemini, normalize_openai


class TestUsageNorm(unittest.TestCase):
    def test_openai_subtracts_cached_from_prompt(self):
        u = normalize_openai(NS(
            prompt_tokens=100, completion_tokens=50,
            prompt_tokens_details=NS(cached_tokens=30),
            completion_tokens_details=NS(reasoning_tokens=20),
        ))
        self.assertEqual(u.input_tokens, 70)       # 100 - 30 cached
        self.assertEqual(u.cached_read_tokens, 30)
        self.assertEqual(u.output_tokens, 50)
        self.assertEqual(u.reasoning_tokens, 20)   # subset of output, not added
        self.assertEqual(u.total_tokens, 150)
        self.assertTrue(u.identity_ok())

    def test_openai_null_safe_when_details_missing(self):
        u = normalize_openai(NS(prompt_tokens=10, completion_tokens=5))
        self.assertEqual((u.input_tokens, u.cached_read_tokens, u.reasoning_tokens), (10, 0, 0))
        self.assertEqual(u.total_tokens, 15)
        self.assertTrue(u.identity_ok())

    def test_openai_accepts_dict_and_none(self):
        u = normalize_openai({"prompt_tokens": 8, "completion_tokens": 2})
        self.assertEqual(u.total_tokens, 10)
        u2 = normalize_openai(None)  # provider returned no usage object at all
        self.assertEqual(u2.total_tokens, 0)

    def test_anthropic_cache_is_separate(self):
        u = normalize_anthropic(NS(
            input_tokens=100, output_tokens=40,
            cache_read_input_tokens=10, cache_creation_input_tokens=5,
        ))
        self.assertEqual(u.input_tokens, 100)      # already excludes cache
        self.assertEqual(u.cached_read_tokens, 10)
        self.assertEqual(u.cache_write_tokens, 5)
        self.assertEqual(u.total_tokens, 155)
        self.assertTrue(u.identity_ok())

    def test_gemini_thoughts_count_as_output(self):
        u = normalize_gemini(NS(
            prompt_token_count=100, candidates_token_count=40,
            cached_content_token_count=20, thoughts_token_count=15,
        ))
        self.assertEqual(u.input_tokens, 80)       # 100 - 20 cached
        self.assertEqual(u.output_tokens, 55)      # candidates + thoughts
        self.assertEqual(u.reasoning_tokens, 15)
        self.assertEqual(u.total_tokens, 155)
        self.assertTrue(u.identity_ok())


class TestPricing(unittest.TestCase):
    def test_known_model_input_output(self):
        u = Usage(input_tokens=1_000_000, output_tokens=1_000_000, total_tokens=2_000_000)
        cost, snap, fx = compute_cost(u, "gpt-4o")   # 2.50 in / 10.00 out
        self.assertAlmostEqual(cost, 12.5, places=6)
        self.assertTrue(snap)

    def test_cached_read_uses_discounted_price(self):
        u = Usage(cached_read_tokens=1_000_000, total_tokens=1_000_000)
        cost, _, _ = compute_cost(u, "gpt-4o")       # cached_in 1.25
        self.assertAlmostEqual(cost, 1.25, places=6)

    def test_reasoning_not_double_counted(self):
        # output already includes reasoning; cost must equal output*price only.
        u = Usage(output_tokens=100, reasoning_tokens=40, total_tokens=100)
        cost, _, _ = compute_cost(u, "gpt-4o")
        self.assertAlmostEqual(cost, 100 * 10.0 / 1e6, places=9)

    def test_prefix_tolerant_lookup_and_fx(self):
        u = Usage(input_tokens=1_000_000, total_tokens=1_000_000)
        cost, _, fx = compute_cost(u, "Pro/deepseek-ai/DeepSeek-V3.2")
        self.assertIsNotNone(cost)                    # resolved via canonical key
        self.assertIsNotNone(fx)                      # CNY-sourced -> fx echoed

    def test_unknown_model_returns_none_cost(self):
        cost, snap, fx = compute_cost(Usage(total_tokens=5), "totally-made-up-model")
        self.assertIsNone(cost)
        self.assertTrue(snap)


class TestModels(unittest.TestCase):
    def test_vendor_and_key(self):
        self.assertEqual(derive_vendor("deepseek-ai/DeepSeek-V3"), "deepseek")
        self.assertEqual(derive_vendor("Pro/moonshotai/Kimi-K2.6"), "moonshot")
        self.assertEqual(derive_vendor("gpt-4o"), "openai")
        self.assertEqual(derive_vendor("claude-sonnet-5"), "anthropic")
        self.assertEqual(derive_vendor("gemini-3-pro"), "google")
        self.assertEqual(canonical_model_key("Pro/deepseek-ai/DeepSeek-V3.2"), "deepseek-v3.2")


class TestRetry(unittest.TestCase):
    def test_retries_transient_then_succeeds(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise TimeoutError("transient")   # name contains 'timeout' -> retryable
            return "ok"

        result, attempts = retry_call(flaky, max_retries=5, sleep=lambda _s: None)
        self.assertEqual(result, "ok")
        self.assertEqual(attempts, 3)

    def test_non_retryable_raises_immediately(self):
        def bad():
            raise ValueError("deterministic")     # not retryable

        with self.assertRaises(ValueError):
            retry_call(bad, max_retries=5, sleep=lambda _s: None)


if __name__ == "__main__":
    unittest.main()
