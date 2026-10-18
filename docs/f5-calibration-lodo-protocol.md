# F5 Calibration and Leave-One-Dataset-Out Protocol

## Purpose

This protocol covers the two later-stage checks that can run after an F5
candidate is selected. Neither workflow accesses the locked test partition.

## Shared monotonic calibration

`scripts/evaluation/export_f5_oof_predictions.py` first reconstructs each
development fold with that fold's persisted normalizer and best checkpoints.
It averages the requested seed predictions, producing exactly one prediction
per image, and records checkpoint identities and hashes. It constructs only
the validation partition. `scripts/evaluation/evaluate_shared_calibration.py`
then accepts this development-only
JSONL predictions with these required fields:

`dataset_id`, `image_id`, `group_id`, `development_fold`,
`source_validation_fold`, `prediction`, `target`, and `locked_test`.

Every prediction must be out-of-fold: `source_validation_fold` must equal the
row's frozen `development_fold`. Duplicate images, locked rows, missing fields,
and groups crossing calibration fit/evaluation folds are rejected.

For each fold, one dataset-agnostic isotonic mapping is fitted on the other
folds and evaluated on the held-out fold. The report contains raw and
cross-fitted calibrated MAE, RMSE, and Spearman globally and per dataset. A
final shared mapping is then fitted on all valid OOF development predictions
for later production packaging. Dataset identity is never an input to the
mapping. The input SHA-256 is recorded in both artifacts.

This workflow evaluates calibration; it does not choose architecture,
hyperparameters, or checkpoints.

## Leave-one-dataset-out transfer

`scripts/training/run_f5_lodo.py` defines 16 outer runs:

- held-out datasets: AADB, PARA, Flickr-AES, and TAD66K;
- models: the selected F5 candidate and F4;
- seeds: 20260920 and 20260921.

For each run, fold 0 rows from the three source datasets are the validation
partition used for early stopping. Remaining source rows are training data.
The held-out dataset is excluded before:

- feature normalization is fitted;
- calibration ranges/heads are created;
- batches and ranking pairs are constructed;
- early stopping and checkpoint selection occur;
- the internal F2 initialization for F5 is trained.

The held-out dataset is loaded only after the best source-selected checkpoint
is frozen. All five non-locked development folds of that dataset are then used
for one transfer evaluation. No held-out labels influence normalization,
training, tuning, calibration fitting, or model selection.

Each LODO normalization identity records `excluded_dataset`,
`visible_dataset_ids`, split identity, and joined-cache identities. This makes
ordinary fold statistics or statistics from a different holdout policy
non-reusable.

### Commands

Plan only (the default):

```bash
.venv/bin/python scripts/training/run_f5_lodo.py --candidate F5-G
```

Execute after the candidate is selected:

```bash
caffeinate -dimsu .venv/bin/python scripts/training/run_f5_lodo.py \
  --execute --candidate F5-G --device mps --batch-size 1024 --num-workers 4
```

Calibrate an exported OOF prediction file:

```bash
.venv/bin/python scripts/evaluation/export_f5_oof_predictions.py \
  --experiment-root /path/to/full-confirmation \
  --candidate F5-G --output /path/to/oof-predictions.jsonl

.venv/bin/python scripts/evaluation/evaluate_shared_calibration.py \
  --oof-predictions /path/to/oof-predictions.jsonl \
  --output-dir /path/to/calibration-output
```

## Interpretation

Calibration may improve absolute score meaning and MAE but cannot repair a
ranking regression. LODO results determine whether gains transfer when an
entire source is unseen. The locked test remains untouched until architecture,
objective, calibration, and production selection policy are frozen.
