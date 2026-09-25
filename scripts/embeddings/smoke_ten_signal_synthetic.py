#!/usr/bin/env python3
"""Run a labelled synthetic-only 1,000-image TenSignalRunner smoke.

This exercises decode, bounded scheduling, shards, ledger recovery, status and
verification.  It deliberately does *not* exercise production scoring: every
provider is a deterministic finite test double and the generated JPEGs contain
no research or user media.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from image_aesthetics.embeddings.feature_contract import DIMENSIONS, FEATURES
from image_aesthetics.embeddings.ledger import Ledger
from image_aesthetics.embeddings.preprocess import ManifestItem
from image_aesthetics.embeddings.ten_signal_runner import RunnerConfig, TenSignalRunner


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _contract() -> dict:
    features = {}
    for name in FEATURES:
        features[name] = {
            "dimension": DIMENSIONS.get(name, 1), "dtype": "float32",
            "range": [-1.0, 1.0] if name in DIMENSIONS else [0.0, 1.0],
            "direction": "higher_is_better", "model_revision": "SYNTHETIC-TEST-ONLY",
            "checkpoint_sha256": hashlib.sha256(name.encode()).hexdigest(),
            "preprocessing": {"fixture": "deterministic-rgb-jpeg", "image_size": 64},
            "normalization": "synthetic-finite-output",
        }
    return {"features": features}


def _providers(calls: dict[str, int]):
    factories = {}
    for name in FEATURES:
        dimension = DIMENSIONS.get(name, 1)

        def factory(name=name, dimension=dimension):
            class SyntheticProvider:
                def __init__(self):
                    self.name = name
                    self.dimension = dimension

                def extract(self, images):
                    calls[name] = calls.get(name, 0) + len(images)
                    # Deterministic but intentionally meaningless.  The shape
                    # and finite-value contract are what this smoke validates.
                    mean = np.asarray(images, dtype=np.float32).mean(axis=(1, 2, 3), keepdims=False)
                    return np.repeat((mean / 255.0).reshape(-1, 1), dimension, axis=1).astype(np.float32)

            return SyntheticProvider()

        factories[name] = factory
    return factories


def _fixtures(root: Path, count: int) -> list[ManifestItem]:
    image_root = root / "fixtures"
    image_root.mkdir()
    rng = np.random.default_rng(20260921)
    items: list[ManifestItem] = []
    manifest = root / "fixture-manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for position in range(count):
            path = image_root / f"{position:04d}.jpg"
            # Vary dimensions and pixels to exercise aspect-preserving decode.
            height, width = 40 + (position % 5) * 9, 48 + (position % 7) * 7
            pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
            Image.fromarray(pixels, "RGB").save(path, quality=85)
            stat = path.stat()
            item = ManifestItem(str(position), str(path), position, stat.st_size, stat.st_mtime_ns)
            items.append(item)
            handle.write(json.dumps({"image_id": item.image_id, "source_path": item.source_path,
                                     "manifest_position": position, "source_size": item.source_size,
                                     "source_mtime_ns": item.source_mtime_ns}, sort_keys=True) + "\n")
    return items


def _provider_inventory() -> dict:
    """Record the local-only readiness evidence that selected this smoke mode."""
    a1 = Path("/Users/rushivyas/datasets/image-aesthetics/ava/releases/siglip2-aesthetic/a1-v1/model.safetensors")
    siglip = Path("/Users/rushivyas/.cache/huggingface/hub/models--google--siglip2-base-patch16-224/snapshots/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2/model.safetensors")
    installed = {name: importlib.util.find_spec(name) is not None
                 for name in ("mlx", "transformers", "torch", "open_clip", "pyiqa", "cv2", "safetensors")}
    absent_factory = "no production SignalProvider factory is implemented or registered in this repository"
    signals = {
        "siglip_embedding": {"checkpoint": str(siglip), "checkpoint_present": siglip.is_file(),
                              "dependencies": {key: installed[key] for key in ("mlx", "transformers")}, "blocked_by": absent_factory},
        "clip_embedding": {"checkpoint_present": False, "dependencies": {"open_clip": installed["open_clip"]},
                           "blocked_by": "OpenAI CLIP ViT-L/14 checkpoint and open_clip dependency are absent; " + absent_factory},
        "laion_clip": {"checkpoint_present": False, "dependencies": {"open_clip": installed["open_clip"]},
                       "blocked_by": "depends on absent CLIP and LAION MLP checkpoint; " + absent_factory},
        "a1_siglip": {"checkpoint": str(a1), "checkpoint_present": a1.is_file(),
                       "checkpoint_sha256": hashlib.sha256(a1.read_bytes()).hexdigest() if a1.is_file() else None,
                       "blocked_by": "A1 checkpoint is present, but no production A1 SignalProvider loader/factory is implemented"},
        "nima": {"checkpoint_present": False, "blocked_by": "genuine AVA-trained NIMA checkpoint and provider are absent"},
        "musiq_aesthetic": {"checkpoint_present": False, "dependencies": {"pyiqa": installed["pyiqa"]},
                             "blocked_by": "pyiqa, MUSIQ-AVA checkpoint, and provider are absent"},
        "musiq_technical": {"checkpoint_present": False, "dependencies": {"pyiqa": installed["pyiqa"]},
                              "blocked_by": "pyiqa, MUSIQ-KonIQ checkpoint, and provider are absent"},
        "sharpness": {"dependencies": {"cv2": installed["cv2"]}, "blocked_by": "OpenCV and production provider are absent"},
        "contrast": {"dependencies": {"cv2": installed["cv2"]}, "blocked_by": "OpenCV and production provider are absent"},
        "colorfulness": {"dependencies": {"numpy": True}, "blocked_by": "production provider is absent"},
    }
    return {"schema_version": 1, "local_only": True, "installed_dependencies": installed,
            "conclusion": "real ten-signal smoke blocked: all ten production providers must be available together",
            "signals": signals}


def _cli(root: Path, command: str) -> dict:
    script = Path(__file__).with_name("cache_ten_signal.py")
    result = subprocess.run([sys.executable, str(script), command, "--cache-dir", str(root / "cache")],
                            check=True, text=True, capture_output=True)
    value = json.loads(result.stdout)
    _json(root / f"{command}.json", value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    args = parser.parse_args()
    if args.count != 1000:
        raise SystemExit("this smoke is fixed at exactly 1,000 fixture images")
    root = args.output.resolve()
    if root.exists():
        raise SystemExit(f"refusing to overwrite smoke output: {root}")
    root.mkdir(parents=True)
    _json(root / "provider-inventory.json", _provider_inventory())
    contract = _contract()
    _json(root / "contract.synthetic-test-only.json", contract)
    items = _fixtures(root, args.count)
    config = RunnerConfig(root / "cache", batch_size=32, decode_workers=4,
                          prepared_queue_batches=2, shard_size=128, image_size=64,
                          progress_interval_seconds=0.25)
    first_calls: dict[str, int] = {}
    started = time.perf_counter()
    initial = TenSignalRunner(config, _providers(first_calls)).run(items, contract, run_id="synthetic-1000-initial")
    initial_seconds = time.perf_counter() - started
    _json(root / "initial-result.json", initial)
    resumed_calls: dict[str, int] = {}
    started = time.perf_counter()
    resumed = TenSignalRunner(config, _providers(resumed_calls)).run(items, contract, run_id="synthetic-1000-resume")
    resume_seconds = time.perf_counter() - started
    _json(root / "resume-result.json", resumed)
    status, failures, verify = (_cli(root, name) for name in ("status", "failures", "verify"))
    decode_utilization_raw = initial["decode_seconds"] / (initial_seconds * config.decode_workers)
    metrics = {
        "schema_version": 1, "label": "SYNTHETIC_TEST_PROVIDER_INFRASTRUCTURE_SMOKE_ONLY",
        "not_production_inference": True, "fixture_images": args.count,
        "generated_fixture_format": "deterministic RGB JPEG", "config": {
            "batch_size": config.batch_size, "decode_workers": config.decode_workers,
            "prepared_queue_batches": config.prepared_queue_batches, "shard_size": config.shard_size,
        },
        "initial": {"elapsed_seconds": initial_seconds, "images_per_second": args.count / initial_seconds,
                    "result": initial, "provider_call_images": first_calls},
        "resume": {"elapsed_seconds": resume_seconds, "images_per_second": args.count / resume_seconds,
                   "result": resumed, "provider_call_images": resumed_calls},
        "provider_timing_seconds": initial["provider_seconds"],
        "resident_memory_gb": initial["resident_memory_gb"],
        "failures": failures, "status": status, "verify": verify,
        "decode_wait_utilization_proxy": {
            "kind": "aggregate_decode_seconds_over_wall_time_x_workers",
            "value": min(1.0, decode_utilization_raw), "raw_value": decode_utilization_raw,
            "formula": "initial.decode_seconds / (initial.elapsed_seconds * decode_workers)",
            "interpretation": "bounded concurrency utilization proxy, capped at 1.0",
            "queue_wait_separately_instrumented": False,
            "limitation": "This is not queue-wait time; the runner does not expose queue wait separately.",
        },
    }
    _json(root / "metrics.json", metrics)
    print(json.dumps({"output": str(root), "initial_images_per_second": metrics["initial"]["images_per_second"],
                      "resume_provider_calls": resumed_calls, "verify_errors": verify["errors"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
