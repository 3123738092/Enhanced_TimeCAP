"""CLI entry point for the LLM-agent pipeline (contextualize / predict tasks).

Example (key filled in later via env var):
    export SILICONFLOW_API_KEY=sk-...
    python -m pipeline.run_llm --task contextualize --channel siliconflow \
        --model deepseek-ai/DeepSeek-V3 --dataset weather_ny --limit 3

Design: this replaces the batch loop of the notebooks with a resumable .py
driver. It never writes into the author's dataset dir — outputs go under
--out-root. Summaries for predict_text / predict_in_context default to the
author's gpt_summary (so predictions can run without regenerating them).
"""
from __future__ import annotations

import argparse
import json
import os

from llm.client import LLMClient
from llm.manifest import Manifest

from .domains import domain_for_dataset
from .runner import run

_TASKS = ("contextualize", "predict_time", "predict_text", "predict_in_context")


def _default_run_id(args) -> str:
    safe_model = args.model.replace("/", "-")
    return f"{args.dataset}_{args.task}_{args.channel}_{safe_model}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline.run_llm", description="TimeCAP LLM-agent pipeline")
    p.add_argument("--task", required=True, choices=_TASKS)
    p.add_argument("--dataset", required=True, help="e.g. weather_ny / weather_sf / weather_hs")
    p.add_argument("--channel", required=True, help="siliconflow | openai | deepseek | moonshot | anthropic | gemini")
    p.add_argument("--model", required=True, help="model id, e.g. deepseek-ai/DeepSeek-V3")
    p.add_argument("--data-root", default="dataset", help="root of the dataset/ dir")
    p.add_argument("--encoder-root", default="encoder", help="root of encoder/ (for in-context embeddings)")
    p.add_argument("--summary-dir", default=None, help="dir of gpt_summary txts (default: author's)")
    p.add_argument("--out-root", default="pipeline_out", help="where generated outputs go")
    p.add_argument("--log-dir", default="llm_runs", help="token logs + manifests")
    p.add_argument("--run-id", default=None, help="log/manifest id (default: dataset_task_channel_model)")
    p.add_argument("--k", type=int, default=5, help="in-context examples")
    p.add_argument("--limit", type=int, default=None, help="cap number of units examined (smoke test)")
    p.add_argument("--workers", type=int, default=1, help="concurrent API calls (thread pool)")
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--max-failures", type=int, default=100, help="abort after this many failures (rate-limit tolerant)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--top-p", type=float, default=1.0)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    domain = domain_for_dataset(args.dataset)
    if args.task not in domain.TASKS:
        raise SystemExit(f"Task {args.task!r} not supported by domain {domain.NAME!r}: {domain.TASKS}")

    data = domain.load(args.dataset, args.data_root)
    units = domain.build_units(
        args.task, data,
        summary_dir=args.summary_dir, encoder_root=args.encoder_root, k=args.k,
    )

    run_id = args.run_id or _default_run_id(args)
    client = LLMClient(channel=args.channel, model=args.model, run_id=run_id,
                       log_dir=args.log_dir, max_retries=args.max_retries)
    manifest = Manifest(run_id, manifest_dir=args.log_dir)
    out_root = os.path.join(args.out_root, domain.NAME)
    gen = {"temperature": args.temperature, "max_tokens": args.max_tokens, "top_p": args.top_p}

    print(f">>> {args.task} | {args.dataset} | channel={args.channel} model={args.model} -> {out_root}")
    stats = run(client, manifest, units, args.task, args.dataset, out_root,
                gen=gen, limit=args.limit, workers=args.workers,
                max_failures=args.max_failures, max_consecutive_failures=args.max_failures)
    print("run stats:", stats)
    print("token summary:", json.dumps(client.logger.summary(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
