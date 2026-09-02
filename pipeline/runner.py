"""The batch runner: execute Units with resume, atomic output, and progress.

Crash-safety ordering per unit:
    1. skip if manifest says done OR the output file already exists
    2. call the LLM (client logs tokens/cost)
    3. write the output file atomically
    4. ONLY THEN mark the manifest done

So an interruption can leave at worst an un-marked completed call (re-run
re-does it), never a "marked done but missing output".

``workers > 1`` runs the API calls concurrently (thread pool); the logger and
manifest are lock-protected, and the SDK client is built once up front.
"""
from __future__ import annotations

import concurrent.futures
import os
import threading
from typing import Iterable, Optional

from llm.client import LLMClient
from llm.manifest import Manifest

from .base import Unit, atomic_write_text


def _build_worklist(manifest, units, task, dataset, out_root, limit):
    """Filter to units that still need work; return (worklist, skipped)."""
    work, skipped, examined = [], 0, 0
    for u in units:
        if limit is not None and examined >= limit:
            break
        examined += 1
        out_path = os.path.join(out_root, u.out_relpath)
        # The output FILE is the source of truth: a manifest can say "done" while
        # the file is missing (e.g. a crash between the API call and the write, or
        # a lost file), which previously skipped it forever. Regenerate if missing.
        if os.path.exists(out_path):
            if not manifest.is_done(task, dataset, u.idx):
                manifest.mark_done(task, dataset, u.idx)  # adopt pre-existing output
            skipped += 1
            continue
        work.append((u, out_path))
    return work, skipped


def _run_sequential(client, manifest, work, task, dataset, gen, progress_every, max_consecutive_failures):
    processed = failed = 0
    consec = 0
    for u, out_path in work:
        try:
            res = client.chat(u.system, u.user, meta={"task": task, "dataset": dataset, "idx": u.idx}, **gen)
        except Exception as exc:  # already logged ok=False by the client
            failed += 1
            consec += 1
            if consec >= max_consecutive_failures:
                print(f"[{task}/{dataset}] ABORT after {consec} consecutive failures: {exc}")
                return processed, failed, True
            continue
        atomic_write_text(out_path, res.text)
        manifest.mark_done(task, dataset, u.idx)
        processed += 1
        consec = 0
        if progress_every and processed % progress_every == 0:
            print(f"[{task}/{dataset}] processed={processed} failed={failed}")
    return processed, failed, False


def _run_concurrent(client, manifest, work, task, dataset, gen, workers, max_failures, progress_every):
    try:
        client.ensure_sdk()  # build once in the main thread (avoids a build race)
    except Exception as exc:
        print(f"[{task}/{dataset}] ABORT: cannot build client: {exc}")
        return 0, 0, True
    lock = threading.Lock()
    state = {"processed": 0, "failed": 0}
    abort = threading.Event()

    def do(item):
        u, out_path = item
        if abort.is_set():
            return
        try:
            res = client.chat(u.system, u.user, meta={"task": task, "dataset": dataset, "idx": u.idx}, **gen)
        except Exception:  # already logged ok=False by the client
            with lock:
                state["failed"] += 1
                if state["failed"] >= max_failures:
                    abort.set()
            return
        atomic_write_text(out_path, res.text)
        manifest.mark_done(task, dataset, u.idx)
        with lock:
            state["processed"] += 1
            n = state["processed"]
        if progress_every and n % progress_every == 0:
            print(f"[{task}/{dataset}] processed={n} failed={state['failed']}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(do, work))
    return state["processed"], state["failed"], abort.is_set()


def run(
    client: LLMClient,
    manifest: Manifest,
    units: Iterable[Unit],
    task: str,
    dataset: str,
    out_root: str,
    gen: Optional[dict] = None,
    limit: Optional[int] = None,
    progress_every: int = 25,
    max_consecutive_failures: int = 10,
    workers: int = 1,
    max_failures: int = 20,
) -> dict:
    """Execute units. ``limit`` caps units EXAMINED (skipped or attempted) so a
    re-run smoke doesn't scan the whole set. Sequential runs abort after
    ``max_consecutive_failures`` in a row; concurrent runs (``workers`` > 1) abort
    after ``max_failures`` total — either way a systemic error can't hammer the
    whole dataset."""
    gen = gen or {}
    work, skipped = _build_worklist(manifest, units, task, dataset, out_root, limit)
    if workers <= 1:
        processed, failed, aborted = _run_sequential(
            client, manifest, work, task, dataset, gen, progress_every, max_consecutive_failures)
    else:
        processed, failed, aborted = _run_concurrent(
            client, manifest, work, task, dataset, gen, workers, max_failures, progress_every)
    return {"processed": processed, "skipped": skipped, "failed": failed, "aborted": aborted}
