#!/usr/bin/env python3
"""Measure real cached-fusion training overhead on CPU and PyTorch MPS.

The script deliberately uses the frozen development split and cached joined-ten
features.  It never opens locked-test rows and it never runs an optimizer long
enough to produce a trainable artifact.  Results are written to an ignored
output directory so backend choices are based on end-to-end measurements rather
than microbenchmarks alone.

Example (on the Apple Silicon desktop host)::

    .venv/bin/python scripts/training/benchmark_fusion_backends.py \
      --output-dir outputs/fusion-backend-benchmark-v1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from image_aesthetics.data.fusion_training_dataset import (  # noqa: E402
    DatasetBalancedBatchSampler,
    FusionTrainingDataset,
)
from image_aesthetics.data.normalization import FeatureNormalizer  # noqa: E402
from image_aesthetics.models.fusion import build_baseline  # noqa: E402
from image_aesthetics.models.fusion.contract import SCALAR_DIM, split_features  # noqa: E402
from image_aesthetics.models.fusion.losses import pairwise_ranking_loss  # noqa: E402


class ProvisionalF5(nn.Module):
    """Architecture-shaped workload, not a production F5 proposal.

    It mirrors the documented branch dimensions, scalar conditioning and linear
    residual path so its throughput is useful before an architecture decision.
    """

    def __init__(self) -> None:
        super().__init__()
        self.siglip = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU())
        self.clip = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU())
        self.scalar = nn.Sequential(nn.LayerNorm(SCALAR_DIM), nn.Linear(SCALAR_DIM, 32), nn.GELU())
        self.gates = nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Linear(32, SCALAR_DIM))
        self.fusion = nn.Sequential(nn.Linear(64 + 64 + 32 + SCALAR_DIM, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU())
        self.aesthetic = nn.Linear(128, 1)
        self.technical = nn.Linear(128, 1)
        self.scalar_skip = nn.Linear(SCALAR_DIM, 1)

    @property
    def param_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        siglip, clip, scalars = split_features(features)
        siglip_hidden = self.siglip(siglip)
        clip_hidden = self.clip(clip)
        # A residual gate preserves the raw scalar evidence instead of allowing
        # an early gate to erase it completely.
        gated_scalars = scalars * (1.0 + 0.25 * torch.tanh(self.gates(torch.cat((siglip_hidden, clip_hidden), dim=-1))))
        scalar_hidden = self.scalar(gated_scalars)
        latent = self.fusion(torch.cat((siglip_hidden, clip_hidden, scalar_hidden, scalars), dim=-1))
        return (self.aesthetic(latent).squeeze(-1) + self.scalar_skip(scalars).squeeze(-1)).sigmoid()


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _rss_bytes() -> int:
    # ru_maxrss is bytes on macOS and KiB on Linux.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _mps_bytes() -> int | None:
    if not torch.backends.mps.is_available():
        return None
    try:
        return int(torch.mps.current_allocated_memory())
    except (AttributeError, RuntimeError):
        return None


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    percentile_index = max(0, math.ceil(0.90 * len(ordered)) - 1)
    return {
        "samples": len(values),
        "mean_ms": mean(values) * 1000,
        "median_ms": median(values) * 1000,
        "p90_ms": ordered[percentile_index] * 1000,
        "min_ms": ordered[0] * 1000,
    }


def _time(device: torch.device, fn: Callable[[], object], *, warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    _sync(device)
    samples = []
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        samples.append(time.perf_counter() - start)
    return _summary(samples)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True))
    os.replace(temporary, path)


def _model_factories() -> dict[str, Callable[[], nn.Module]]:
    return {
        "F2": lambda: build_baseline("F2"),
        "F4": lambda: build_baseline("F4"),
        "F5_provisional": ProvisionalF5,
    }


def _move_batch(batch: dict[str, object], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return batch["features"].to(device, dtype=torch.float32), batch["target"].to(device, dtype=torch.float32)


def _validation_legacy(model: nn.Module, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> int:
    copied = 0
    model.eval()
    with torch.no_grad():
        for features, target in batches:
            prediction = model(features)
            copied += len(prediction.cpu().tolist()) + len(target.cpu().tolist())
    return copied


def _validation_batched(model: nn.Module, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> int:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for features, target in batches:
            predictions.append(model(features))
            targets.append(target)
    return int(torch.stack((torch.cat(predictions), torch.cat(targets)), dim=-1).cpu().shape[0])


def _parity(factory: Callable[[], nn.Module], features: torch.Tensor, device: torch.device) -> dict[str, float] | None:
    if device.type == "cpu":
        return None
    torch.manual_seed(20261016)
    reference = factory().eval()
    candidate = factory().eval()
    candidate.load_state_dict(reference.state_dict())
    with torch.no_grad():
        cpu_output = reference(features).cpu()
        device_output = candidate.to(device)(features.to(device)).cpu()
    difference = (cpu_output - device_output).abs()
    return {"max_abs": float(difference.max()), "mean_abs": float(difference.mean())}


def _run_model_batch(
    model_name: str,
    factory: Callable[[], nn.Module],
    cpu_batches: list[dict[str, object]],
    device: torch.device,
    *,
    warmup: int,
    repeats: int,
    output_dir: Path,
    train_batches_per_epoch: int,
    validation_batches_per_epoch: int,
    status_interval_batches: int,
) -> dict[str, object]:
    # The loader benchmark intentionally remains host-only; model timing starts
    # with a materialized real batch and reports transfer separately.
    model = factory().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    batch = cpu_batches[0]
    features, target = _move_batch(batch, device)

    def transfer() -> tuple[torch.Tensor, torch.Tensor]:
        return _move_batch(batch, device)

    def step_without_ranking() -> torch.Tensor:
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = F.huber_loss(model(features), target)
        loss.backward(); optimizer.step()
        return loss

    def step_with_ranking() -> torch.Tensor:
        model.train(); optimizer.zero_grad(set_to_none=True)
        prediction = model(features)
        loss = F.huber_loss(prediction, target) + 0.1 * pairwise_ranking_loss(prediction, target)
        loss.backward(); optimizer.step()
        return loss

    def ranking_only() -> torch.Tensor:
        scores = torch.linspace(0, 1, len(target), device=device, requires_grad=True)
        loss = pairwise_ranking_loss(scores, target)
        loss.backward()
        return loss

    validation_source = cpu_batches[: min(len(cpu_batches), 4)]
    validation_batches = [_move_batch(item, device) for item in validation_source]
    status_file = output_dir / f"status-{model_name}-{device.type}.json"
    checkpoint_file = output_dir / f"checkpoint-{model_name}-{device.type}.pt"
    status_payload = {"state": "running", "model": model_name, "batch": 1, "loss": 0.123}

    def checkpoint() -> None:
        temporary = checkpoint_file.with_suffix(".pt.tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": 1}, temporary)
        os.replace(temporary, checkpoint_file)

    result: dict[str, object] = {
        "parameter_count": parameter_count,
        "batch_size": int(features.shape[0]),
        "host_to_device": _time(device, transfer, warmup=warmup, repeats=repeats),
        "forward_backward_optimizer": _time(device, step_without_ranking, warmup=warmup, repeats=repeats),
        "ranking_loss_backward": _time(device, ranking_only, warmup=warmup, repeats=repeats),
        "full_step_with_ranking": _time(device, step_with_ranking, warmup=warmup, repeats=repeats),
        "validation_legacy_host_transfer": _time(device, lambda: _validation_legacy(model, validation_batches), warmup=warmup, repeats=repeats),
        "validation_batched_host_transfer": _time(device, lambda: _validation_batched(model, validation_batches), warmup=warmup, repeats=repeats),
        "status_atomic_json": _time(device, lambda: _atomic_json(status_file, status_payload), warmup=0, repeats=repeats),
        "checkpoint_full_state": _time(device, checkpoint, warmup=0, repeats=max(2, min(5, repeats))),
        "numerical_parity": _parity(factory, batch["features"], device),
        "peak_rss_bytes": _rss_bytes(),
        "mps_allocated_bytes": _mps_bytes(),
    }
    full_step_seconds = result["full_step_with_ranking"]["median_ms"] / 1000
    transfer_seconds = result["host_to_device"]["median_ms"] / 1000 * train_batches_per_epoch
    validation_seconds = result["validation_batched_host_transfer"]["median_ms"] / 1000 * max(1, math.ceil(validation_batches_per_epoch / len(validation_batches)))
    status_seconds = result["status_atomic_json"]["median_ms"] / 1000 * math.ceil(train_batches_per_epoch / status_interval_batches)
    checkpoint_seconds = result["checkpoint_full_state"]["median_ms"] / 1000
    result["estimated_compute_validation_status_checkpoint_seconds"] = full_step_seconds * train_batches_per_epoch + validation_seconds + status_seconds + checkpoint_seconds
    result["estimated_epoch_transfer_seconds"] = transfer_seconds
    result["estimated_training_samples_per_second"] = features.shape[0] / full_step_seconds
    return result


def _load_datasets(args: argparse.Namespace):
    joins = {
        name: args.data_root / name / "features" / "joined-ten-v1"
        for name in ("aadb", "para", "flickr_aes", "tad66k")
    }
    train = FusionTrainingDataset(args.split_manifest, joins, partition="train", validation_fold=args.fold)
    validation = FusionTrainingDataset(args.split_manifest, joins, partition="validation", validation_fold=args.fold)
    stats = FeatureNormalizer.from_file(args.normalization_path, expected_identity=train.normalization_identity())
    return train.normalized(stats), validation.normalized(stats)


def _loader_measurement(dataset, batch_size: int, batches: int, workers: int) -> tuple[dict[str, float], list[dict[str, object]], int]:
    sampler = DatasetBalancedBatchSampler(dataset, batch_size, seed=20260920)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                        persistent_workers=workers > 0)
    iterator = iter(loader)
    # The first materialization includes shard/cache opening; keep it as a
    # separate diagnostic while using subsequent steady-state batches for model
    # timing.
    start = time.perf_counter(); first = next(iterator); first_seconds = time.perf_counter() - start
    loaded = [first]
    samples: list[float] = []
    for _ in range(max(1, batches - 1)):
        start = time.perf_counter(); loaded.append(next(iterator)); samples.append(time.perf_counter() - start)
    steady = _summary(samples) if samples else {"samples": 0, "mean_ms": 0.0, "median_ms": 0.0, "p90_ms": 0.0, "min_ms": 0.0}
    return {"workers": workers, "first_batch_ms": first_seconds * 1000, "steady_state": steady}, loaded, len(loader)


def _device_available(name: str) -> tuple[bool, str | None]:
    if name == "cpu":
        return True, None
    if name == "mps":
        if not torch.backends.mps.is_built():
            return False, "PyTorch was built without MPS"
        if not torch.backends.mps.is_available():
            return False, "MPS is unavailable to this process (headless/sandboxed sessions commonly block Metal)"
        return True, None
    return False, f"unsupported device {name!r}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/Users/rushivyas/datasets/image-aesthetics"))
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--normalization-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "fusion-backend-benchmark-v1")
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--batch-sizes", default="256,512,1024")
    parser.add_argument("--devices", default="cpu,mps")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--loader-batches", type=int, default=6)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--status-interval-batches", type=int, default=25)
    args = parser.parse_args()
    args.split_manifest = args.split_manifest or args.data_root / "fusion" / "prepared" / "v6" / "samples.jsonl"
    args.normalization_path = args.normalization_path or args.data_root / "fusion" / "experiments" / "baseline-matrix-v6" / "normalization" / f"fold{args.fold}.json"
    batch_sizes = [int(value) for value in args.batch_sizes.split(",") if value]
    if not batch_sizes or min(batch_sizes) < 1:
        raise ValueError("batch sizes must be positive")
    if args.warmup < 0 or args.repeats < 1 or args.loader_workers < 0:
        raise ValueError("warmup and loader-workers must be >=0 and repeats must be >=1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, validation = _load_datasets(args)
    report: dict[str, object] = {
        "schema_version": "fusion-backend-benchmark-v1",
        "created_at": time.time(),
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "torch": {"version": torch.__version__, "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available()},
        "data": {"split_manifest": str(args.split_manifest), "normalization_path": str(args.normalization_path), "fold": args.fold, "train_examples": len(train), "validation_examples": len(validation)},
        "config": {"batch_sizes": batch_sizes, "devices": args.devices.split(","), "warmup": args.warmup, "repeats": args.repeats, "loader_workers": args.loader_workers, "status_interval_batches": args.status_interval_batches},
        "results": {},
    }
    for batch_size in batch_sizes:
        loader_result, cpu_batches, train_batches_per_epoch = _loader_measurement(train, batch_size, args.loader_batches, args.loader_workers)
        validation_batches_per_epoch = math.ceil(len(validation) / batch_size)
        batch_key = str(batch_size)
        report["results"][batch_key] = {"data_loading": loader_result, "devices": {}}
        for device_name in (value.strip() for value in args.devices.split(",") if value.strip()):
            available, reason = _device_available(device_name)
            if not available:
                report["results"][batch_key]["devices"][device_name] = {"state": "skipped", "reason": reason}
                continue
            device = torch.device(device_name)
            device_result: dict[str, object] = {"state": "completed", "models": {}}
            for model_name, factory in _model_factories().items():
                torch.manual_seed(20261016)
                try:
                    device_result["models"][model_name] = _run_model_batch(
                        model_name, factory, cpu_batches, device, warmup=args.warmup,
                        repeats=args.repeats, output_dir=args.output_dir,
                        train_batches_per_epoch=train_batches_per_epoch,
                        validation_batches_per_epoch=validation_batches_per_epoch,
                        status_interval_batches=args.status_interval_batches,
                    )
                except RuntimeError as error:
                    device_result["models"][model_name] = {"state": "failed", "error": repr(error)}
                    if device.type == "mps":
                        # Continuing after an MPS allocation/kernel fault can
                        # poison later work in this process, so record a clear
                        # fallback rather than presenting partial speed claims.
                        break
            # With worker prefetching, some ``next`` calls consume an already
            # prepared batch while others wait for a worker.  The mean captures
            # sustained epoch throughput; the median would understate it.
            steady_loader_seconds = loader_result["steady_state"]["mean_ms"] / 1000
            for model_result in device_result["models"].values():
                if model_result.get("state") == "failed":
                    continue
                model_result["estimated_epoch_data_loading_seconds"] = steady_loader_seconds * train_batches_per_epoch
                model_result["estimated_epoch_seconds"] = (model_result["estimated_compute_validation_status_checkpoint_seconds"] + model_result["estimated_epoch_data_loading_seconds"] + model_result["estimated_epoch_transfer_seconds"])
            report["results"][batch_key]["devices"][device_name] = device_result
        (args.output_dir / "benchmark-results.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
