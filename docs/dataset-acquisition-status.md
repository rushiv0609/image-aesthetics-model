# Dataset Acquisition Status

This is the reproducibility record for external fusion datasets. All artifacts live
under `/Users/rushivyas/datasets/image-aesthetics/` and are excluded from Git. Use is
restricted to the approved local, non-commercial research POC; redistribution is
prohibited.

## TAD66K

- Official project: `https://github.com/woshidandan/TANet-image-aesthetics-and-quality-assessment`
- Official Google Drive folder: `1b2D9LeeG5XZzhEa8ldnIZjGh0IHadHhU`
- `labels.zip` Drive ID: `13JQRCgqraNnaTSsZxuPAdO4Mjlmp8r-l`
- `TAD66K.zip` Drive ID: `19qICyVsmCNZ6nAKjvv7t8hjLrDZjwwdh`
- Native labels: 66,327 image-level aesthetic MOS values on the 1–10 scale.
- Native split: 52,248 train and 14,079 test records. The source split is retained as
  metadata, not accepted as our final split.
- Theme memberships are retained from `labels/unmerge/{train,test}/*.csv`.

The large archive is downloaded with `gdown --continue`; interruption is safe and a
subsequent invocation resumes the partial file.

## SPAQ

- Official project: `https://github.com/h4nwei/SPAQ`
- The Google Drive URL published by the official repository returned 404 during this
  acquisition.
- Research mirror: `https://huggingface.co/datasets/AaronCIH/IQA_dataset`, restricted
  to the `SPAQ/` subtree. The mirror's README states educational/research use only.
- Snapshot contents: annotation ZIP, README, `SPAQ.z01`–`SPAQ.z06`, and `SPAQ.zip`.
- Native labels: 11,125 smartphone images with MOS and brightness, contrast,
  noisiness, sharpness, and colorfulness attribute scores on the 0–100 scale.
- Scene memberships and available EXIF columns are retained.

The split archive is downloaded with the Hugging Face snapshot client. Its cache is
resumable. Every part must match the repository SHA-256 metadata before extraction.

## Verification and Preparation Gates

1. All expected files exist and have their recorded byte size.
2. SHA-256 hashes match published metadata where available; otherwise the locally
   acquired hash is frozen in the immutable download manifest.
3. Archive integrity succeeds before extraction.
4. Extraction occurs into a new staging directory with path traversal checks.
5. Annotation audit reproduces 66,327 TAD66K and 11,125 SPAQ accepted records.
6. Every annotated filename resolves to exactly one image; missing, extra, corrupt,
   and duplicate items are written to the rejection audit.
7. Cross-dataset deduplication and new grouped train/development/locked-test splits
   are created later. Native publisher splits are never treated as leakage-safe by
   assumption.

Run metadata-only audits with:

```bash
.venv/bin/python scripts/data/audit_annotations.py \
  --dataset tad66k --annotations /path/to/labels.zip

.venv/bin/python scripts/data/audit_annotations.py \
  --dataset spaq --annotations /path/to/Annotations-*.zip
```
