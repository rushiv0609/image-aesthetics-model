"""Build the cross-dataset deduplication and split snapshot."""
from __future__ import annotations
import argparse, json, os, tempfile
from pathlib import Path
from image_aesthetics.data.fusion_preparation import VERSION, collapse_feature_labels, prepare_fusion

def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def keyed(values: list[str], option: str) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{option} expects DATASET=PATH, got {value!r}")
        dataset, raw_path = value.split("=", 1)
        if not dataset or dataset in result:
            raise SystemExit(f"invalid or duplicate dataset in {option}: {dataset!r}")
        result[dataset] = Path(raw_path)
    return result

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label-manifest", action="append", default=[], metavar="DATASET=PATH")
    p.add_argument("--joined-cache", action="append", default=[], metavar="DATASET=PATH")
    p.add_argument("--image-root", action="append", default=[], metavar="DATASET=PATH")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260920)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    label_manifests = keyed(args.label_manifest, "--label-manifest")
    joined_caches = keyed(args.joined_cache, "--joined-cache")
    image_roots = keyed(args.image_root, "--image-root")
    if not label_manifests or label_manifests.keys() != joined_caches.keys() or label_manifests.keys() != image_roots.keys():
        raise SystemExit("label manifests, joined caches, and image roots must name the same datasets")
    records = []
    input_audits = {}
    for dataset in sorted(label_manifests):
        collapsed, audit = collapse_feature_labels(
            load(label_manifests[dataset]), load(joined_caches[dataset] / "join.jsonl"))
        if any(row["dataset_id"] != dataset for row in collapsed):
            raise SystemExit(f"dataset identity mismatch in {dataset}")
        records.extend(collapsed)
        input_audits[dataset] = audit
    rows, candidates = prepare_fusion(records, image_roots, seed=args.seed)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        with (staging / "samples.jsonl").open("w", encoding="utf-8") as out:
            for row in rows: out.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        with (staging / "near-duplicate-review.jsonl").open("w", encoding="utf-8") as out:
            for row in candidates: out.write(json.dumps(row, sort_keys=True) + "\n")
        eligible = {row["record_id"] for row in rows}
        with (staging / "quarantined-near-duplicate-candidates.jsonl").open("w", encoding="utf-8") as out:
            for candidate in candidates:
                for record_id in (candidate["left"], candidate["right"]):
                    if record_id not in eligible:
                        out.write(json.dumps({"record_id": record_id, "reason": "unresolved_near_duplicate_candidate"}, sort_keys=True) + "\n")
        groups = {row["group_id"] for row in rows}
        distributions = {}
        for dataset in sorted({row["dataset_id"] for row in rows}):
            distributions[dataset] = {}
            subset = [row for row in rows if row["dataset_id"] == dataset]
            ordered = sorted(subset, key=lambda row: (float(row["native_score"]), row["record_id"]))
            for index, row in enumerate(ordered):
                bucket = min(9, index * 10 // len(ordered))
                partition = "test" if row["locked_test"] else "fold" + str(row["development_fold"])
                distributions[dataset].setdefault(partition, [0] * 10)[bucket] += 1
        audit = {"schema_version": VERSION, "seed": args.seed, "records": len(rows), "groups": len(groups), "near_duplicate_candidates": len(candidates), "unresolved_review_candidates": len(candidates), "locked_test": sum(bool(row["locked_test"]) for row in rows), "development": sum(not row["locked_test"] for row in rows), "datasets": {dataset: {"records": sum(row["dataset_id"] == dataset for row in rows), "locked_test": sum(row["dataset_id"] == dataset and row["locked_test"] for row in rows)} for dataset in sorted({row["dataset_id"] for row in rows})}, "dataset_partition_quantile_counts": distributions, "input_audits": input_audits}
        (staging / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(staging, args.output)
        print(json.dumps(audit, sort_keys=True))
    except Exception:
        import shutil; shutil.rmtree(staging, ignore_errors=True); raise
if __name__ == "__main__": main()
