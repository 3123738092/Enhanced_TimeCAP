"""Runner tests: resume, idempotency, atomic output, adopting pre-existing files.

Driven with a fake adapter + injected sdk client (no data, no key, no SDKs).
"""
from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace as NS

from llm.client import LLMClient
from llm.manifest import Manifest
from pipeline.base import Unit
from pipeline.runner import run


def _fake_adapter(text="output-text", raise_exc=None):
    usage = NS(prompt_tokens=5, completion_tokens=3,
               prompt_tokens_details=NS(cached_tokens=0), completion_tokens_details=None)

    def call(*a, **k):
        if raise_exc is not None:
            raise raise_exc
        return NS()

    return NS(
        build_client=lambda cfg: object(),
        call=call,
        extract=lambda r: (text, usage, "stop"),
    )


def _client(log_dir):
    return LLMClient("openai", "gpt-4o", "run", log_dir=log_dir,
                     adapter=_fake_adapter(), sdk_client=object())


def _units(idxs):
    return [Unit(i, "sys", f"user {i}", f"gpt_summary/ny_{i}.txt") for i in idxs]


class TestRunner(unittest.TestCase):
    def test_writes_marks_and_logs(self):
        with tempfile.TemporaryDirectory() as log_dir, tempfile.TemporaryDirectory() as out:
            client = _client(log_dir)
            man = Manifest("run", log_dir)
            stats = run(client, man, _units([0, 1, 2]), "contextualize", "weather_ny", out,
                        gen={"temperature": 0.7, "max_tokens": 32})
            self.assertEqual(stats, {"processed": 3, "skipped": 0, "failed": 0, "aborted": False})
            for i in (0, 1, 2):
                p = os.path.join(out, "gpt_summary", f"ny_{i}.txt")
                self.assertTrue(os.path.exists(p))
                with open(p) as f:
                    self.assertEqual(f.read(), "output-text")
            self.assertEqual(len(man), 3)
            self.assertEqual(len(client.logger.read()), 3)
            # no temp files left behind
            self.assertFalse([x for x in os.listdir(os.path.join(out, "gpt_summary")) if x.endswith(".tmp")])

    def test_resume_skips_completed(self):
        with tempfile.TemporaryDirectory() as log_dir, tempfile.TemporaryDirectory() as out:
            client = _client(log_dir)
            run(client, Manifest("run", log_dir), _units([0, 1, 2]), "contextualize", "weather_ny", out,
                gen={"temperature": 0.7})
            # a fresh run (new client + manifest, same run_id/dir) must skip everything
            client2 = _client(log_dir)
            stats = run(client2, Manifest("run", log_dir), _units([0, 1, 2]), "contextualize", "weather_ny", out,
                        gen={"temperature": 0.7})
            self.assertEqual(stats, {"processed": 0, "skipped": 3, "failed": 0, "aborted": False})
            # same run_id -> shared log file; resume appends nothing, so it stays at 3
            self.assertEqual(len(client2.logger.read()), 3)

    def test_adopts_preexisting_output_file(self):
        with tempfile.TemporaryDirectory() as log_dir, tempfile.TemporaryDirectory() as out:
            # simulate an output already produced by a previous notebook run
            os.makedirs(os.path.join(out, "gpt_summary"), exist_ok=True)
            with open(os.path.join(out, "gpt_summary", "ny_1.txt"), "w") as f:
                f.write("pre-existing")
            client = _client(log_dir)
            man = Manifest("run", log_dir)
            stats = run(client, man, _units([0, 1, 2]), "contextualize", "weather_ny", out,
                        gen={"temperature": 0.7})
            self.assertEqual(stats, {"processed": 2, "skipped": 1, "failed": 0, "aborted": False})
            self.assertTrue(man.is_done("contextualize", "weather_ny", 1))
            with open(os.path.join(out, "gpt_summary", "ny_1.txt")) as f:
                self.assertEqual(f.read(), "pre-existing")  # not overwritten

    def test_concurrent_processes_all(self):
        # workers>1 must process everything exactly once (locks prevent races)
        with tempfile.TemporaryDirectory() as log_dir, tempfile.TemporaryDirectory() as out:
            client = _client(log_dir)
            man = Manifest("run", log_dir)
            stats = run(client, man, _units(range(20)), "contextualize", "weather_ny", out,
                        gen={"temperature": 0.7}, workers=4)
            self.assertEqual(stats["processed"], 20)
            self.assertEqual(stats["failed"], 0)
            self.assertEqual(len(man), 20)                      # no double-marks
            self.assertEqual(len(client.logger.read()), 20)     # no lost/dup log lines
            files = [f for f in os.listdir(os.path.join(out, "gpt_summary")) if f.endswith(".txt")]
            self.assertEqual(len(files), 20)

    def test_aborts_on_consecutive_failures(self):
        # a systemic error (e.g. bad key) must not hammer the whole dataset
        with tempfile.TemporaryDirectory() as log_dir, tempfile.TemporaryDirectory() as out:
            adapter = _fake_adapter(raise_exc=RuntimeError("no key"))
            client = LLMClient("openai", "gpt-4o", "run", log_dir=log_dir,
                               adapter=adapter, sdk_client=object())
            stats = run(client, Manifest("run", log_dir), _units(range(100)),
                        "contextualize", "weather_ny", out, gen={"temperature": 0.7},
                        max_consecutive_failures=5)
            self.assertTrue(stats["aborted"])
            self.assertEqual(stats["failed"], 5)      # stopped at 5, not 100
            self.assertEqual(stats["processed"], 0)


if __name__ == "__main__":
    unittest.main()
