#!/usr/bin/env python3
"""Generate or inspect an external AVA/SigLIP2 Base-224 embedding cache.

This command never downloads AVA or model checkpoints.  The manifest and cache
directory must be supplied explicitly (or be under ``IAM_DATA_ROOT``), and the
model revision must be pinned.  Production deployments wire the Moments
Generator MLX extractor through ``CacheGenerator.embedder_factory``; this CLI
keeps that boundary explicit so it cannot accidentally run a different model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from image_aesthetics.common.config import data_root
from image_aesthetics.embeddings.cache import CacheConfig, CacheGenerator
from image_aesthetics.embeddings.preprocess import load_manifest
from image_aesthetics.embeddings.shards import verify_shard
from image_aesthetics.embeddings.ledger import Ledger


def cache_path(args: argparse.Namespace) -> Path:
    if args.cache_dir:
        return Path(args.cache_dir).expanduser()
    return data_root(args.data_root) / "ava" / "embeddings" / "siglip2-base-patch16-224" / args.cache_version


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run/resume extraction against an existing manifest")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--cache-dir", type=Path)
    run.add_argument("--cache-version", default="v1")
    run.add_argument("--data-root", type=Path)
    run.add_argument("--model-revision", required=True, help="immutable Hugging Face commit hash")
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--decode-workers", type=int, default=8)
    run.add_argument("--prepared-queue-batches", type=int, default=3)
    run.add_argument("--shard-size", type=int, default=4096)
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument("--retry-permanent", action="store_true")
    run.add_argument("--content-fingerprint", action="store_true")
    run.add_argument("--run-id")
    for name, help_text in (("status", "show read-only ledger counts"), ("failures", "list read-only failures"), ("verify", "verify committed shard hashes and norms")):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--cache-dir", type=Path, required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    path = cache_path(args)
    if args.command == "run":
        items = load_manifest(args.manifest)
        config = CacheConfig(path, model_revision=args.model_revision, batch_size=args.batch_size,
                             decode_workers=args.decode_workers, prepared_queue_batches=args.prepared_queue_batches,
                             shard_size=args.shard_size, max_attempts=args.max_attempts,
                             retry_permanent=args.retry_permanent, content_fingerprint=args.content_fingerprint)
        result = CacheGenerator(config).run(items, run_id=args.run_id)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["state"] == "completed" else 2
    ledger = Ledger(path / "work.sqlite3")
    ledger.open()
    if args.command == "status":
        print(json.dumps(ledger.counts(), indent=2, sort_keys=True))
        return 0
    if args.command == "failures":
        rows = ledger.read("SELECT image_id,manifest_position,state,error_type,error_message,attempts FROM items WHERE state IN ('retryable_error','permanent_error') ORDER BY manifest_position")
        print(json.dumps([dict(zip(("image_id", "manifest_position", "state", "error_type", "error_message", "attempts"), row)) for row in rows], indent=2))
        return 0
    errors = []
    for shard in sorted(path.glob("shard-*.safetensors")):
        try: verify_shard(shard)
        except Exception as exc: errors.append({"path": str(shard), "error": str(exc)})
    print(json.dumps({"shards": len(list(path.glob('shard-*.safetensors'))), "errors": errors}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
