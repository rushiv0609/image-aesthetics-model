"""Normalize prepared dataset snapshots into deterministic image and label manifests."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from image_aesthetics.embeddings.preprocess import validate_image_decodable

DATASETS = ("ava", "tad66k", "spaq", "para", "flickr_aes", "koniq10k", "aadb")


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_path(row: dict) -> str:
    return str(row.get("source_path") or row.get("path") or row.get("image_name") or row.get("identifier") or "")


def _resolve(raw: str, root: Path, by_name: dict[str, list[Path]]) -> Path:
    candidate_path = Path(raw)
    if not raw:
        raise ValueError("missing source path")
    candidate = candidate_path if candidate_path.is_absolute() else root / candidate_path
    if not candidate.is_file() and not candidate_path.is_absolute():
        matches = by_name.get(candidate_path.name, [])
        if len(matches) != 1:
            raise ValueError("missing source path" if not matches else "ambiguous basename source path")
        candidate = matches[0]
    original = candidate
    if not original.is_absolute():
        original = root / original
    try:
        original.relative_to(root)
    except ValueError:
        pass
    current = Path(original.anchor)
    for part in original.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlink source path: {raw}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        raise ValueError(f"source path is missing or escapes image root: {raw}") from None
    if not resolved.is_file():
        raise ValueError(f"source is not a file: {raw}")
    return resolved


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _prepared_records(dataset_id: str, prepared_root: Path) -> tuple[list[dict], list[dict], dict]:
    if dataset_id not in DATASETS:
        raise ValueError(f"unsupported dataset: {dataset_id}")
    if dataset_id == "ava":
        rows = _rows(prepared_root / "dataset.jsonl")
        split_by_id = {}
        for split in ("train", "validation", "locked-test"):
            split_path = prepared_root / "splits" / f"{split}.jsonl"
            for split_row in _rows(split_path):
                split_by_id[str(split_row["image_id"])] = {
                    "split": split, "locked_test": split == "locked-test",
                    "challenge_id": split_row.get("challenge_id"),
                    "group_id": split_row.get("group_id"),
                }
        rows = [{**row, **split_by_id.get(str(row.get("image_id")), {"split": "unassigned", "locked_test": False})}
                for row in rows]
        labels = [{**row, "label_value": row.get("normalized_mean", row.get("mean_rating"))} for row in rows]
        return rows, labels, {"source_layout": "ava-dataset-jsonl"}
    if dataset_id in ("flickr_aes", "aadb"):
        images = _rows(prepared_root / "images.jsonl")
        annotations = _rows(prepared_root / "samples.jsonl")
        if not images:
            images = annotations
        return images, annotations, {"source_layout": f"{dataset_id}-images-and-samples-jsonl"}
    rows = _rows(prepared_root / "samples.jsonl")
    return rows, rows, {"source_layout": "canonical-samples-jsonl"}


def materialize_feature_manifest(dataset_id: str, prepared_root: Path, image_root: Path,
                                 output_root: Path) -> dict:
    """Write one immutable feature row per image and a separate label join table.

    Paths are resolved and stored relative to image_root. Source hashes are
    recomputed; traversal, escapes, symlinks, and hash mismatches are rejected.
    """
    if dataset_id not in DATASETS:
        raise ValueError(f"unsupported dataset: {dataset_id}")
    prepared_root = Path(prepared_root).resolve(strict=True)
    image_root = Path(image_root).resolve(strict=True)
    if not image_root.is_dir():
        raise ValueError("image_root must be a directory")
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"manifest snapshot already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    features_raw, labels_raw, layout = _prepared_records(dataset_id, prepared_root)
    by_feature_id: dict[str, dict] = {}
    rejected: list[dict] = []
    image_index: dict[str, list[Path]] = defaultdict(list)
    indexed = False
    labels_by_image: dict[str, list[dict]] = defaultdict(list)
    label_splits_by_image: dict[str, set[str]] = defaultdict(set)
    for source_row in labels_raw:
        raw_id = str(source_row.get("image_id") or source_row.get("identifier") or source_row.get("image_name") or source_row.get("path") or "")
        labels_by_image[f"{dataset_id}:{raw_id}"].append(source_row)
        split = source_row.get("split") or source_row.get("partition") or source_row.get("data_split")
        if split not in (None, ""):
            label_splits_by_image[f"{dataset_id}:{raw_id}"].add(str(split).strip().lower())

    for row in features_raw:
        raw_id = str(row.get("image_id") or row.get("identifier") or row.get("image_name") or row.get("path") or "")
        raw_path = _source_path(row)
        qualified_id = f"{dataset_id}:{raw_id}"
        try:
            if not indexed and not (Path(raw_path).is_absolute() or (image_root / raw_path).is_file()):
                for candidate in sorted(image_root.rglob("*")):
                    if candidate.is_file():
                        image_index[candidate.name].append(candidate)
                indexed = True
            path = _resolve(raw_path, image_root, image_index)
            digest = _sha256(path)
            expected_digest = row.get("source_fingerprint") or row.get("sha256")
            if expected_digest and digest != str(expected_digest):
                raise ValueError("source_sha256_mismatch")
            stat = path.stat()
            if row.get("source_size") is not None and int(row["source_size"]) != stat.st_size:
                raise ValueError("source_size_mismatch")
            if row.get("source_mtime_ns") is not None and int(row["source_mtime_ns"]) != stat.st_mtime_ns:
                raise ValueError("source_mtime_mismatch")
            try:
                validate_image_decodable(path)
            except ValueError as exc:
                raise ValueError(f"source_decode_invalid: {exc}") from exc
            record = {
                "dataset_id": dataset_id,
                "image_id": qualified_id,
                "manifest_position": -1,
                "source_path": path.relative_to(image_root).as_posix(),
                "source_sha256": digest,
                "source_size": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns,
                "label_join_id": qualified_id,
                "split": row.get("split"),
                "locked_test": bool(row.get("locked_test", row.get("split") in ("test", "locked-test"))),
                "label_task": "technical" if dataset_id in ("spaq", "koniq10k") else "aesthetic",
                "training_role": "technical_auxiliary" if dataset_id == "spaq" else ("technical_diagnostic_only" if dataset_id == "koniq10k" else ("diagnostic_only_ava" if dataset_id == "ava" else "aesthetic_fusion_candidate")),
                "exposure_policy": "diagnostic_only_ava_exposure" if dataset_id == "ava" else ("technical_auxiliary_spaq" if dataset_id == "spaq" else ("technical_diagnostic_only_koniq10k" if dataset_id == "koniq10k" else "eligible_subject_to_overlap_audit")),
                "eligible_for_fusion": dataset_id not in ("ava", "koniq10k"),
                "eligible_as_aesthetic_target": dataset_id not in ("ava", "spaq", "koniq10k"),
                "quarantine": False,
            }
            previous = by_feature_id.get(qualified_id)
            if previous and previous["source_sha256"] != digest:
                raise ValueError("image_id_maps_to_multiple_source_bytes")
            by_feature_id.setdefault(qualified_id, record)
        except Exception as exc:
            rejected.append({"dataset_id": dataset_id, "image_id": qualified_id,
                             "source_path": raw_path, "reason": str(exc)})

    # AADB's author test archive repeats byte-identical train images.
    unique_by_digest: dict[str, dict] = {}
    for record in by_feature_id.values():
        primary = unique_by_digest.setdefault(record["source_sha256"], record)
        primary.setdefault("source_aliases", []).append(record["label_join_id"])
    digest_to_feature = {digest: record["image_id"] for digest, record in unique_by_digest.items()}
    feature_rows = sorted(unique_by_digest.values(), key=lambda row: row["image_id"])
    alias_split_conflicts = []
    conflicted_features: set[str] = set()
    for position, row in enumerate(feature_rows):
        row["manifest_position"] = position
        aliases = row.get("source_aliases", (row["label_join_id"],))
        splits = sorted({split for alias in aliases for split in label_splits_by_image.get(alias, set())})
        split_conflict = len(splits) > 1
        if split_conflict:
            conflicted_features.add(row["image_id"])
            alias_split_conflicts.append({"feature_id": row["image_id"], "sha256": row["source_sha256"],
                                          "source_aliases": list(aliases), "native_splits": splits,
                                          "reason": "exact_byte_aliases_span_native_splits"})
        row["quarantine"] = (not any(alias in labels_by_image for alias in aliases)) or split_conflict
        if split_conflict:
            row["quarantine_reason"] = "exact_byte_aliases_span_native_splits"
    join_targets = {row["label_join_id"]: digest_to_feature[row["source_sha256"]]
                    for row in by_feature_id.values()}
    label_rows = []
    for label_id in sorted(labels_by_image):
        for ordinal, source_row in enumerate(labels_by_image[label_id]):
            row = dict(source_row)
            row.update({"dataset_id": dataset_id, "label_join_id": label_id,
                        "join_feature_id": join_targets.get(label_id), "label_ordinal": ordinal,
                        "label_task": "technical" if dataset_id in ("spaq", "koniq10k") else "aesthetic",
                        "training_role": "technical_auxiliary" if dataset_id == "spaq" else ("technical_diagnostic_only" if dataset_id == "koniq10k" else ("diagnostic_only_ava" if dataset_id == "ava" else "aesthetic_fusion_candidate")),
                        "exposure_policy": "diagnostic_only_ava_exposure" if dataset_id == "ava" else ("technical_auxiliary_spaq" if dataset_id == "spaq" else ("technical_diagnostic_only_koniq10k" if dataset_id == "koniq10k" else "eligible_subject_to_overlap_audit")),
                        "eligible_for_fusion": dataset_id not in ("ava", "koniq10k"),
                        "eligible_as_aesthetic_target": dataset_id not in ("ava", "spaq", "koniq10k"),
                        "locked_test": bool(source_row.get("locked_test", source_row.get("split") in ("test", "locked-test"))),
                        "quarantine": label_id not in join_targets or join_targets[label_id] in conflicted_features,
                        "quarantine_reason": "exact_byte_aliases_span_native_splits" if join_targets.get(label_id) in conflicted_features else None})
            label_rows.append(row)

    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        _write_jsonl(staging / "features.jsonl", feature_rows)
        _write_jsonl(staging / "labels.jsonl", label_rows)
        _write_jsonl(staging / "rejected.jsonl", sorted(rejected, key=lambda row: (row["image_id"], row["reason"])))
        _write_jsonl(staging / "alias-split-conflicts.jsonl", sorted(alias_split_conflicts, key=lambda row: row["feature_id"]))
        audit = {
            "schema_version": "feature-manifest-v2",
            "dataset_id": dataset_id,
            **layout,
            "feature_rows": len(feature_rows),
            "label_rows": len(label_rows),
            "distinct_label_join_ids": len(labels_by_image),
            "rejected_source_rows": len(rejected),
            "exact_content_alias_groups": sum(1 for aliases in unique_by_digest.values() if len(aliases.get("source_aliases", [])) > 1),
            "cross_native_split_alias_quarantine_groups": len(alias_split_conflicts),
            "aesthetic_target_rows": sum(bool(row["eligible_as_aesthetic_target"]) and not row["quarantine"] for row in feature_rows),
            "technical_auxiliary_rows": sum(row["training_role"] == "technical_auxiliary" and not row["quarantine"] for row in feature_rows),
            "technical_diagnostic_rows": sum(row["training_role"] == "technical_diagnostic_only" for row in feature_rows),
            "flickr_duplicate_labels_preserved": dataset_id == "flickr_aes",
            "path_policy": "resolved relative paths; unsafe paths and hash mismatches rejected",
            "decode_policy": "Pillow full load, EXIF transpose, RGB conversion; truncated images rejected",
        }
        (staging / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(staging, output_root)
        return audit
    except Exception:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        raise
