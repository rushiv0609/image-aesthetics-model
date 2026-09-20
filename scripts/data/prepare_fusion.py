"""Build the cross-dataset deduplication and split snapshot."""
from __future__ import annotations
import argparse, json, os, tempfile
from pathlib import Path
from image_aesthetics.data.fusion_preparation import VERSION, prepare_fusion

def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tad", type=Path, required=True)
    p.add_argument("--spaq", type=Path, required=True)
    p.add_argument("--tad-images", type=Path, required=True)
    p.add_argument("--spaq-images", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260920)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    records = load(args.tad / "samples.jsonl") + load(args.spaq / "samples.jsonl")
    for row in records:
        row["record_id"] = str(row["dataset_id"]) + ":" + str(row["image_id"])
    rows, candidates = prepare_fusion(records, {"tad66k": args.tad_images, "spaq": args.spaq_images}, seed=args.seed)
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
        audit = {"schema_version": VERSION, "seed": args.seed, "records": len(rows), "groups": len(groups), "near_duplicate_candidates": len(candidates), "unresolved_review_candidates": len(candidates), "locked_test": sum(bool(row["locked_test"]) for row in rows), "development": sum(not row["locked_test"] for row in rows), "datasets": {dataset: {"records": sum(row["dataset_id"] == dataset for row in rows), "locked_test": sum(row["dataset_id"] == dataset and row["locked_test"] for row in rows)} for dataset in sorted({row["dataset_id"] for row in rows})}, "dataset_partition_quantile_counts": distributions}
        (staging / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(staging, args.output)
        print(json.dumps(audit, sort_keys=True))
    except Exception:
        import shutil; shutil.rmtree(staging, ignore_errors=True); raise
if __name__ == "__main__": main()
