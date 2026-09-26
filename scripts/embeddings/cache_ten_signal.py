"""Create, run, inspect, and verify immutable ten-signal feature caches."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

from image_aesthetics.embeddings.feature_contract import (
    load_contract, load_manifest, manifest_sha256, write_contract,
)
from image_aesthetics.embeddings.ledger import Ledger
from image_aesthetics.embeddings.preprocess import load_manifest as load_items
from image_aesthetics.embeddings.shards import read_shard, verify_shard
from image_aesthetics.embeddings.ten_signal_runner import (RunnerConfig, TenSignalRunner,
                                                             source_fingerprints_sha256)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    contract = sub.add_parser("contract")
    contract.add_argument("--manifest", type=Path, required=True)
    contract.add_argument("--image-root", type=Path, required=True)
    contract.add_argument("--contract", type=Path, required=True)
    contract.add_argument("--spec", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--image-root", type=Path, required=True)
    run.add_argument("--contract", type=Path, required=True)
    run.add_argument("--cache-dir", type=Path, required=True)
    run.add_argument("--provider-factory", required=True, help="Python module:callable returning signal factories")
    run.add_argument("--batch-size", type=int, default=16)
    for name in ("status", "failures", "verify", "repair-exhausted"):
        command = sub.add_parser(name)
        command.add_argument("--cache-dir", type=Path, required=True)
        if name == "verify":
            command.add_argument("--manifest", type=Path)
            command.add_argument("--image-root", type=Path)
            command.add_argument("--contract", type=Path)
            command.add_argument("--allow-partial-diagnostics", action="store_true")
        if name == "repair-exhausted":
            command.add_argument("--max-attempts", type=int, default=3)
    return parser


def _verify(cache_dir: Path, ledger: Ledger, args) -> tuple[dict, bool]:
    errors: list[dict] = []
    verified = 0
    shard_rows: dict[int, tuple[dict, object]] = {}
    manifest_coordinates: dict[int, str] | None = None
    metadata = ledger.cache_metadata()
    identity = metadata.get("cache_identity", "")
    if not identity:
        errors.append({"type": "missing_cache_identity"})

    if args.contract:
        try:
            contract = load_contract(args.contract)
            if contract["cache_identity"] != identity:
                errors.append({"type": "cache_identity_mismatch"})
        except Exception as exc:
            errors.append({"type": "invalid_contract", "error": str(exc)})
    if args.manifest:
        if not args.image_root:
            errors.append({"type": "missing_image_root"})
        else:
            try:
                load_manifest(args.manifest, args.image_root)
                if manifest_sha256(args.manifest) != metadata.get("manifest_sha256"):
                    errors.append({"type": "manifest_identity_mismatch"})
                items = load_items(args.manifest, image_root=args.image_root)
                manifest_coordinates = {item.manifest_position: item.image_id for item in items}
                items = TenSignalRunner._fingerprint_sources(items)
                fingerprints_hash = source_fingerprints_sha256(items)
                if fingerprints_hash != metadata.get("source_fingerprints_sha256"):
                    errors.append({"type": "source_fingerprint_mismatch"})
            except Exception as exc:
                errors.append({"type": "invalid_manifest", "error": str(exc)})

    committed = ledger.committed_shards()
    by_shard = {int(row["shard_id"]): row for row in committed}
    completed = ledger.completed_records()
    paths = [Path(row["path"]) for row in committed]
    known_paths = {path.resolve() for path in paths}
    paths.extend(path for path in sorted(cache_dir.glob("shard-*"))
                 if path.suffix in (".safetensors", ".npz") and path.resolve() not in known_paths)
    try:
        import numpy as np
    except ImportError as exc:
        return {"ledger": "error", "verified_shards": 0,
                "errors": [{"type": "missing_numpy", "error": str(exc)}]}, False

    for path in paths:
        if not path.exists():
            errors.append({"type": "missing_shard", "shard": path.name})
            continue
        try:
            embeddings, sidecar = read_shard(path)
            verify_shard(path, int(sidecar.get("embedding_dim", 768)))
            verified += 1
            shard_id = int(path.stem.split("-")[-1])
            shard_rows[shard_id] = (sidecar, embeddings)
            expected = by_shard.get(shard_id)
            if expected is None:
                errors.append({"type": "orphan_shard", "shard": path.name})
                continue
            if expected["sha256"] != sidecar.get("sha256"):
                errors.append({"type": "checksum_mismatch", "shard": path.name, "shard_id": shard_id})
            if expected["row_count"] != int(sidecar.get("row_count", -1)):
                errors.append({"type": "row_count_mismatch", "shard": path.name, "shard_id": shard_id})
            if sidecar.get("cache_identity") != identity:
                errors.append({"type": "cache_identity_mismatch", "shard": path.name, "shard_id": shard_id})
            expected_fingerprints = metadata.get("source_fingerprints_sha256")
            if expected_fingerprints and sidecar.get("source_fingerprints_sha256") != expected_fingerprints:
                errors.append({"type": "source_fingerprint_mismatch", "shard": path.name, "shard_id": shard_id})
            for key in ("manifest_sha256", "ordered_ids_sha256", "item_count"):
                expected_value = metadata.get(key)
                if expected_value is not None and str(sidecar.get(key)) != str(expected_value):
                    errors.append({"type": f"{key}_mismatch", "shard": path.name, "shard_id": shard_id})
            names = sidecar.get("signal_names", [])
            dimensions = sidecar.get("signal_dimensions", {})
            specs = sidecar.get("signal_specs", {})
            if not names or set(names) != set(dimensions) or set(names) != set(specs):
                errors.append({"type": "missing_feature_contract", "shard": path.name, "shard_id": shard_id})
                continue
            offset = 0
            for name in names:
                dimension = int(dimensions[name])
                values = embeddings[:, offset:offset + dimension]
                offset += dimension
                spec = specs[name]
                if values.dtype != np.float32 or not np.isfinite(values).all():
                    errors.append({"type": "invalid_feature_values", "shard": path.name, "signal": name})
                    continue
                low, high = spec["range"]
                if (values < low).any() or (values > high).any():
                    errors.append({"type": "feature_range_violation", "shard": path.name, "signal": name})
                if name in ("siglip_embedding", "clip_embedding"):
                    norms = np.linalg.norm(values, axis=1)
                    if (norms < 1e-6).any() or not np.allclose(norms, 1.0, atol=5e-3):
                        errors.append({"type": "feature_norm_violation", "shard": path.name, "signal": name})
            if offset != embeddings.shape[1]:
                errors.append({"type": "feature_dimension_mismatch", "shard": path.name})
        except Exception as exc:
            errors.append({"type": "corrupt_shard", "shard": path.name, "error": str(exc)})

    for shard_id in by_shard:
        if shard_id not in shard_rows:
            errors.append({"type": "missing_shard", "shard_id": shard_id})
    states = ledger.status_counts()
    expected_count = int(metadata.get("item_count", len(completed)))
    if len(completed) != expected_count:
        errors.append({"type": "manifest_coverage_mismatch", "expected": expected_count, "completed": len(completed)})
    if manifest_coordinates is not None:
        completed_coordinates = {int(item["manifest_position"]): str(item["image_id"]) for item in completed}
        if len(completed_coordinates) != len(completed) or completed_coordinates != manifest_coordinates:
            errors.append({"type": "manifest_id_coverage_mismatch",
                           "expected": len(manifest_coordinates), "completed": len(completed_coordinates)})
    for state, count in states.items():
        if state != "completed" and count:
            errors.append({"type": "incomplete_rows", "state": state, "count": count})

    expected_coordinates = {(item["shard_id"], item["shard_row"]):
                            (item["image_id"], item["manifest_position"]) for item in completed}
    seen_ids: set[str] = set()
    for item in completed:
        image_id = item["image_id"]
        if image_id in seen_ids:
            errors.append({"type": "duplicate_image_id", "image_id": image_id})
        seen_ids.add(image_id)
        pair = shard_rows.get(item["shard_id"])
        if pair is None:
            errors.append({"type": "missing_shard", "image_id": image_id, "shard_id": item["shard_id"]})
            continue
        sidecar, embeddings = pair
        row = item["shard_row"]
        if row is None or row < 0 or row >= sidecar["row_count"]:
            errors.append({"type": "row_out_of_range", "image_id": image_id,
                           "shard_id": item["shard_id"], "shard_row": row})
            continue
        if (sidecar["image_ids"][row] != image_id or
                sidecar["positions"][row] != item["manifest_position"]):
            errors.append({"type": "coordinate_mismatch", "image_id": image_id,
                           "shard_id": item["shard_id"], "shard_row": row})
        checksum = hashlib.sha256(embeddings[row].tobytes()).hexdigest()
        if item.get("embedding_checksum") and checksum != item["embedding_checksum"]:
            errors.append({"type": "checksum_mismatch", "image_id": image_id,
                           "shard_id": item["shard_id"], "shard_row": row})

    observed_coordinates: set[tuple[int, int]] = set()
    for shard_id, (sidecar, _) in shard_rows.items():
        for row, (image_id, position) in enumerate(zip(sidecar["image_ids"], sidecar["positions"])):
            coordinate = (shard_id, row)
            if coordinate in observed_coordinates:
                errors.append({"type": "duplicate_shard_coordinate", "shard_id": shard_id, "shard_row": row})
            observed_coordinates.add(coordinate)
            if image_id in seen_ids and expected_coordinates.get(coordinate) != (image_id, position):
                errors.append({"type": "unexpected_shard_row", "shard_id": shard_id, "shard_row": row, "image_id": image_id})
    errors.sort(key=lambda row: (row.get("type", ""), row.get("image_id", ""), str(row.get("shard_id", ""))))
    return {"ledger": "error" if errors else "ok", "verified_shards": verified, "errors": errors}, not errors


def main() -> None:
    args = _parser().parse_args()
    if args.command == "contract":
        spec = json.loads(args.spec.read_text())
        load_manifest(args.manifest, args.image_root)
        items = TenSignalRunner._fingerprint_sources(load_items(args.manifest, image_root=args.image_root))
        fingerprints_hash = source_fingerprints_sha256(items)
        identity = write_contract(args.contract, input_manifest_sha256=manifest_sha256(args.manifest),
                                  contract=spec, source_fingerprints_sha256=fingerprints_hash)
        print(json.dumps({"cache_identity": identity, "manifest_sha256": manifest_sha256(args.manifest)}))
        return
    if args.command == "run":
        payload = load_contract(args.contract)
        if manifest_sha256(args.manifest) != payload["input_manifest_sha256"]:
            raise SystemExit("manifest bytes differ from the immutable cache contract")
        load_manifest(args.manifest, args.image_root)
        items = TenSignalRunner._fingerprint_sources(load_items(args.manifest, image_root=args.image_root))
        if payload.get("source_fingerprints_sha256") != source_fingerprints_sha256(items):
            raise SystemExit("source bytes differ from the immutable cache contract")
        module_name, separator, attribute = args.provider_factory.partition(":")
        if not separator:
            raise SystemExit("--provider-factory must be module:callable")
        provider_factory = getattr(importlib.import_module(module_name), attribute)
        providers = provider_factory()
        if not isinstance(providers, dict):
            raise SystemExit("provider factory must return a signal-name to factory mapping")
        result = TenSignalRunner(RunnerConfig(args.cache_dir, batch_size=args.batch_size), providers).run(
            items, payload)
        print(json.dumps(result, sort_keys=True))
        if result["state"] != "completed":
            raise SystemExit(1)
        return

    db = args.cache_dir / "work.sqlite3"
    if not db.exists():
        raise SystemExit("ledger missing")
    ledger = Ledger(db)
    ledger.open()
    if args.command == "status":
        print(json.dumps(ledger.status_counts(), sort_keys=True))
    elif args.command == "failures":
        print(json.dumps(ledger.failures(), sort_keys=True))
    elif args.command == "repair-exhausted":
        repaired = ledger.repair_exhausted_attempts(args.max_attempts)
        print(json.dumps({"repaired": repaired, "status": ledger.status_counts()}, sort_keys=True))
    else:
        result, valid = _verify(args.cache_dir, ledger, args)
        if not valid and args.allow_partial_diagnostics:
            result["ledger"] = "partial"
        print(json.dumps(result, sort_keys=True))
        if not valid and not args.allow_partial_diagnostics:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
