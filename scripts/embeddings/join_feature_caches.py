#!/usr/bin/env python3
"""Create an identity-checked row index across immutable feature caches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _strict_verify(cache: Path, manifest: Path, image_root: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, str(ROOT / "scripts/embeddings/cache_ten_signal.py"), "verify",
               "--cache-dir", str(cache), "--manifest", str(manifest), "--image-root", str(image_root)]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"verification did not return JSON for {cache}: {result.stderr}") from None
    if result.returncode or value.get("errors"):
        raise RuntimeError(f"cache verification failed for {cache}: {value}")
    return value


def _metadata(cache: Path) -> tuple[dict[str, str], dict[str, tuple]]:
    db = cache / "work.sqlite3"
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as connection:
        meta = dict(connection.execute("SELECT key,value FROM cache_metadata"))
        rows = {row[0]: row for row in connection.execute(
            "SELECT image_id,manifest_position,source_fingerprint,state,shard_id,shard_row FROM items")}
    return meta, rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _rows_sha256(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _identity_mismatches(left: dict[str, str], right: dict[str, str]) -> dict[str, tuple[str | None, str | None]]:
    # Older local-five caches may store the canonical-item hash in
    # `manifest_sha256`, while newer caches also record the source manifest's
    # byte hash. Compare the normalized manifest identity shared by both
    # generations, not the representation-specific byte hash.
    fields = ("canonical_items_sha256", "source_fingerprints_sha256", "item_count", "ordered_ids_sha256")
    return {key: (left.get(key), right.get(key)) for key in fields if left.get(key) != right.get(key)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-cache", type=Path, required=True, help="local-five cache directory")
    parser.add_argument("--remaining-cache", type=Path, required=True, help="remaining-four cache directory")
    parser.add_argument("--nima-cache", type=Path, help="optional pretrained NIMA cache directory")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    local_verify = _strict_verify(args.local_cache, args.manifest, args.image_root)
    remaining_verify = _strict_verify(args.remaining_cache, args.manifest, args.image_root)
    local_meta, local_rows = _metadata(args.local_cache)
    remaining_meta, remaining_rows = _metadata(args.remaining_cache)
    mismatches = _identity_mismatches(local_meta, remaining_meta)
    if mismatches:
        raise SystemExit(f"cache manifests differ; refusing join: {mismatches}")
    if local_rows.keys() != remaining_rows.keys():
        raise SystemExit("cache image ID sets differ; refusing join")
    nima_verify = nima_meta = nima_rows = None
    if args.nima_cache:
        nima_verify = _strict_verify(args.nima_cache, args.manifest, args.image_root)
        nima_meta, nima_rows = _metadata(args.nima_cache)
        nima_mismatches = _identity_mismatches(local_meta, nima_meta)
        if nima_mismatches:
            raise SystemExit(f"NIMA cache manifest differs; refusing join: {nima_mismatches}")
        if local_rows.keys() != nima_rows.keys():
            raise SystemExit("NIMA cache image ID set differs; refusing join")

    manifest_rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    by_id = {row["image_id"]: row for row in manifest_rows}
    joined = []
    for image_id in sorted(local_rows, key=lambda key: local_rows[key][1]):
        left = local_rows[image_id]
        right = remaining_rows[image_id]
        nima = nima_rows[image_id] if nima_rows is not None else None
        if left[1] != right[1] or left[2] != right[2]:
            raise SystemExit(f"cache row coordinates/source hashes differ for {image_id}")
        if left[3] != "completed" or right[3] != "completed":
            raise SystemExit(f"cache row is incomplete for {image_id}: {left[3]}, {right[3]}")
        if left[4] is None or left[5] is None or right[4] is None or right[5] is None:
            raise SystemExit(f"cache row has no shard location for {image_id}")
        if nima is not None:
            if left[1] != nima[1] or left[2] != nima[2] or nima[3] != "completed":
                raise SystemExit(f"NIMA row identity/state differs for {image_id}")
            if nima[4] is None or nima[5] is None:
                raise SystemExit(f"NIMA row has no shard location for {image_id}")
        source = by_id.get(image_id)
        if source is None:
            raise SystemExit(f"image ID missing from canonical manifest: {image_id}")
        source_sha = source.get("source_sha256", left[2])
        if source_sha != left[2]:
            raise SystemExit(f"manifest source SHA differs from cache ledger for {image_id}")
        joined_row = {
            "dataset_id": args.dataset_id,
            "image_id": image_id,
            "manifest_position": left[1],
            "source_sha256": source_sha,
            "manifest_sha256": local_meta["manifest_sha256"],
            "local_five": {"shard_id": left[4], "row": left[5]},
            "remaining_four": {"shard_id": right[4], "row": right[5]},
            "quarantine": source.get("quarantine", False),
            "eligible_for_fusion": source.get("eligible_for_fusion", False),
            "eligible_as_aesthetic_target": source.get("eligible_as_aesthetic_target", False),
            "training_role": source.get("training_role"),
            "exposure_policy": source.get("exposure_policy"),
            "label_task": source.get("label_task"),
            "label_join_id": source.get("label_join_id"),
            "split": source.get("split"),
            "locked_test": source.get("locked_test", False),
        }
        if nima is not None:
            joined_row["nima"] = {"shard_id": nima[4], "row": nima[5]}
        joined.append(joined_row)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    jsonl_path = output / "join.jsonl"
    audit_path = output / "join-audit.json"
    expected_join_sha = _rows_sha256(joined)
    if audit_path.exists() and not jsonl_path.exists():
        raise SystemExit(f"join audit exists without its row index: {output}")
    if jsonl_path.exists() and hashlib.sha256(jsonl_path.read_bytes()).hexdigest() != expected_join_sha:
        raise SystemExit(f"existing immutable join differs from verified cache rows: {output}")
    audit = {
        "schema_version": 1,
        "dataset_id": args.dataset_id,
        "manifest_sha256": local_meta["canonical_items_sha256"],
        "source_fingerprints_sha256": local_meta["source_fingerprints_sha256"],
        "cache_identities": {
            "local_five": local_meta["cache_identity"],
            "remaining_four": remaining_meta["cache_identity"],
            **({"nima": nima_meta["cache_identity"]} if nima_meta else {}),
        },
        "cache_roots": {
            "local_five": str(args.local_cache.resolve()),
            "remaining_four": str(args.remaining_cache.resolve()),
            **({"nima": str(args.nima_cache.resolve())} if args.nima_cache else {}),
        },
        "image_count": len(joined),
        "quarantined_count": sum(bool(row["quarantine"]) for row in joined),
        "fusion_eligible_count": sum(bool(row["eligible_for_fusion"]) for row in joined),
        "active_signal_groups": ["local_five", "remaining_four"] + (["nima"] if nima_meta else []),
        "blocked_signal": None if nima_meta else "nima",
        "storage": "row index references existing immutable shards; no feature tensors copied",
        "strict_verification": {"local_five": local_verify, "remaining_four": remaining_verify,
                                **({"nima": nima_verify} if nima_verify else {})},
        "join_sha256": expected_join_sha,
    }
    if audit_path.exists():
        existing_audit = json.loads(audit_path.read_text())
        if any(existing_audit.get(key) != audit[key] for key in (
            "dataset_id", "manifest_sha256", "source_fingerprints_sha256",
            "cache_identities", "image_count", "join_sha256",
            "cache_roots",
        )):
            raise SystemExit(f"existing immutable join audit differs from verified cache rows: {output}")
        print(json.dumps(existing_audit, indent=2))
        return 0
    if not jsonl_path.exists():
        _write_jsonl(jsonl_path, joined)
    audit_tmp = audit_path.with_suffix(".json.tmp")
    audit_tmp.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    os.replace(audit_tmp, audit_path)
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
