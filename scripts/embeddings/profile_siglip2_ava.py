#!/usr/bin/env python3
"""Profile AVA SigLIP2 stages without changing the production cache contract.

``decode`` measures only ImageIO/Pillow preparation with bounded threaded work.
``infer`` prepares a deterministic sample first, warms one MLX model, then
measures only batched forward passes.  The full overlapped cache should be run
with ``cache_siglip2_ava.py``; its ``progress.json`` now exposes stage timing,
queue occupancy, and backpressure counters.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from image_aesthetics.embeddings.backend import SigLIP2Embedder
from image_aesthetics.embeddings.preprocess import load_manifest, prepare_image


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def decode_one(item):
    started = time.perf_counter()
    image, decoder = prepare_image(Path(item.source_path), 224)
    return image, decoder, time.perf_counter() - started


def run_decode(items, workers: int) -> dict:
    started = time.perf_counter()
    durations: list[float] = []
    decoders: dict[str, int] = {}
    # ``map`` preserves deterministic result order while native decoders run
    # concurrently.  The 3,000-image profile is intentionally bounded in size.
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="profile-decode") as pool:
        for _item, decoder, elapsed in pool.map(decode_one, items):
            durations.append(elapsed)
            decoders[decoder] = decoders.get(decoder, 0) + 1
    elapsed = time.perf_counter() - started
    return {
        "workers": workers,
        "images": len(items),
        "wall_seconds": elapsed,
        "images_per_second": len(items) / elapsed if elapsed else 0.0,
        "decode_p50_seconds": percentile(durations, 0.50),
        "decode_p95_seconds": percentile(durations, 0.95),
        "decoder_counts": decoders,
    }


def run_inference(images, model_revision: str, batch_size: int) -> dict:
    embedder = SigLIP2Embedder.production("google/siglip2-base-patch16-224", model_revision)
    # Force one representative forward pass before starting the clock.  This
    # excludes MLX graph compilation and model construction from throughput.
    warmup = images[: min(batch_size, len(images))]
    embedder.embed(__import__("numpy").stack(warmup, axis=0))
    timings: list[float] = []
    count = 0
    started = time.perf_counter()
    import numpy as np
    for offset in range(0, len(images), batch_size):
        batch = images[offset:offset + batch_size]
        batch_started = time.perf_counter()
        output = embedder.embed(np.stack(batch, axis=0))
        # Materialize/check output so timing includes the complete MLX result.
        if output.shape != (len(batch), 768):
            raise RuntimeError(f"unexpected embedding shape {output.shape}")
        timings.append(time.perf_counter() - batch_started)
        count += len(batch)
    elapsed = time.perf_counter() - started
    return {
        "batch_size": batch_size,
        "images": count,
        "forward_wall_seconds": elapsed,
        "forward_images_per_second": count / elapsed if elapsed else 0.0,
        "batch_p50_seconds": percentile(timings, 0.50),
        "batch_p95_seconds": percentile(timings, 0.95),
        "batch_count": len(timings),
    }


def current_rss_gb() -> float:
    try:
        value = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip())
        return value / (1024 ** 2)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def system_swap_snapshot() -> str | None:
    try:
        return subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def memory_pressure_snapshot() -> str | None:
    try:
        return subprocess.check_output(["memory_pressure", "-Q"], text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def stream_forward(embedder, images, batch_size: int) -> tuple[list, float]:
    import numpy as np
    outputs = []
    started = time.perf_counter()
    for offset in range(0, len(images), batch_size):
        outputs.append(embedder.embed(np.stack(images[offset:offset + batch_size], axis=0)))
    return outputs, time.perf_counter() - started


def run_owner_comparison(images, model_revision: str, batch_size: int) -> dict:
    import numpy as np
    single = SigLIP2Embedder.production("google/siglip2-base-patch16-224", model_revision)
    single.embed(np.stack(images[:batch_size], axis=0))
    baseline_runs = [stream_forward(single, images, batch_size) for _ in range(2)]
    baseline_outputs = np.concatenate(baseline_runs[0][0], axis=0)
    # Do not leave the baseline model resident while measuring two owners; the
    # comparison should represent one versus two model instances, not three.
    del single
    import gc
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except (ImportError, AttributeError):
        pass

    swap_before = system_swap_snapshot()
    pressure_before = memory_pressure_snapshot()
    owner_a = SigLIP2Embedder.production("google/siglip2-base-patch16-224", model_revision)
    owner_b = SigLIP2Embedder.production("google/siglip2-base-patch16-224", model_revision)
    owner_a.embed(np.stack(images[:batch_size], axis=0))
    owner_b.embed(np.stack(images[-batch_size:], axis=0))
    rss_after_warmup = current_rss_gb()
    halves = (images[: len(images) // 2], images[len(images) // 2:])
    concurrent_runs = []
    for _ in range(2):
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="mlx-owner") as pool:
            futures = [pool.submit(stream_forward, owner, part, batch_size)
                       for owner, part in zip((owner_a, owner_b), halves)]
            values = [future.result() for future in futures]
        concurrent_runs.append({
            "wall_seconds": time.perf_counter() - started,
            "stream_seconds": [value[1] for value in values],
            "outputs": np.concatenate([np.concatenate(value[0], axis=0) for value in values], axis=0),
        })
    paired = concurrent_runs[0]["outputs"]
    diff = np.abs(baseline_outputs - paired)
    cosine = (baseline_outputs * paired).sum(axis=1)
    return {
        "batch_size": batch_size,
        "images": len(images),
        "single_owner_runs": [{"wall_seconds": value[1],
                                "images_per_second": len(images) / value[1]}
                               for value in baseline_runs],
        "two_owner_runs": [{"wall_seconds": value["wall_seconds"],
                             "images_per_second": len(images) / value["wall_seconds"],
                             "stream_seconds": value["stream_seconds"]}
                            for value in concurrent_runs],
        "single_owner_mean_seconds": statistics.mean(value[1] for value in baseline_runs),
        "two_owner_mean_seconds": statistics.mean(value["wall_seconds"] for value in concurrent_runs),
        "speedup_two_over_single": statistics.mean(value[1] for value in baseline_runs) / statistics.mean(value["wall_seconds"] for value in concurrent_runs),
        "rss_after_two_warmups_gb": rss_after_warmup,
        "ru_maxrss_gb": __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss / (1024 ** 3),
        "swap_before": swap_before,
        "swap_after": system_swap_snapshot(),
        "memory_pressure_before": pressure_before,
        "memory_pressure_after": memory_pressure_snapshot(),
        "embedding_max_abs_diff": float(diff.max()),
        "embedding_mean_abs_diff": float(diff.mean()),
        "embedding_min_cosine": float(cosine.min()),
        "embedding_mean_cosine": float(cosine.mean()),
    }


def run_shared_owner(images, model_revision: str, batch_size: int) -> dict:
    """Probe two callers sharing one warmed model instance.

    The MLX vision forward is read-only after construction; this is isolated
    from the production path and reports an error rather than hiding one.
    """
    import numpy as np
    owner = SigLIP2Embedder.production("google/siglip2-base-patch16-224", model_revision)
    owner.embed(np.stack(images[:batch_size], axis=0))
    baseline_outputs = np.concatenate(stream_forward(owner, images, batch_size)[0], axis=0)
    halves = (images[: len(images) // 2], images[len(images) // 2:])
    runs = []
    for _ in range(2):
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="shared-mlx") as pool:
            futures = [pool.submit(stream_forward, owner, part, batch_size) for part in halves]
            values = [future.result() for future in futures]
        outputs = np.concatenate([np.concatenate(value[0], axis=0) for value in values], axis=0)
        runs.append({"wall_seconds": time.perf_counter() - started,
                     "stream_seconds": [value[1] for value in values],
                     "outputs": outputs})
    diff = np.abs(baseline_outputs - runs[0]["outputs"])
    cosine = (baseline_outputs * runs[0]["outputs"]).sum(axis=1)
    return {
        "batch_size": batch_size,
        "images": len(images),
        "runs": [{"wall_seconds": run["wall_seconds"],
                   "images_per_second": len(images) / run["wall_seconds"],
                   "stream_seconds": run["stream_seconds"]} for run in runs],
        "mean_seconds": statistics.mean(run["wall_seconds"] for run in runs),
        "embedding_max_abs_diff": float(diff.max()),
        "embedding_mean_abs_diff": float(diff.mean()),
        "embedding_min_cosine": float(cosine.min()),
        "embedding_mean_cosine": float(cosine.mean()),
        "rss_gb": current_rss_gb(),
        "swap": system_swap_snapshot(),
        "memory_pressure": memory_pressure_snapshot(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("decode", "infer", "owners", "shared-owner"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-revision", help="required for infer")
    parser.add_argument("--workers", default="4,8,12")
    parser.add_argument("--batch-sizes", default="32,48,64")
    args = parser.parse_args(argv)
    items = load_manifest(args.manifest)
    if args.mode == "decode":
        result = {"mode": "decode", "manifest": str(args.manifest),
                  "results": [run_decode(items, int(value)) for value in args.workers.split(",") if value]}
    elif args.mode == "infer":
        if not args.model_revision:
            parser.error("--model-revision is required for infer")
        print(json.dumps({"mode": "infer", "phase": "preparing", "images": len(items)}), flush=True)
        images = [prepare_image(Path(item.source_path), 224)[0] for item in items]
        result = {"mode": "infer", "manifest": str(args.manifest),
                  "results": [run_inference(images, args.model_revision, int(value))
                              for value in args.batch_sizes.split(",") if value]}
    elif args.mode == "owners":
        if not args.model_revision:
            parser.error("--model-revision is required for owners")
        print(json.dumps({"mode": "owners", "phase": "preparing", "images": len(items)}), flush=True)
        images = [prepare_image(Path(item.source_path), 224)[0] for item in items]
        result = {"mode": "owners", "manifest": str(args.manifest),
                  "results": [run_owner_comparison(images, args.model_revision, int(value))
                              for value in args.batch_sizes.split(",") if value]}
    else:
        if not args.model_revision:
            parser.error("--model-revision is required for shared-owner")
        print(json.dumps({"mode": "shared-owner", "phase": "preparing", "images": len(items)}), flush=True)
        images = [prepare_image(Path(item.source_path), 224)[0] for item in items]
        result = {"mode": "shared-owner", "manifest": str(args.manifest),
                  "results": [run_shared_owner(images, args.model_revision, int(value))
                              for value in args.batch_sizes.split(",") if value]}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
