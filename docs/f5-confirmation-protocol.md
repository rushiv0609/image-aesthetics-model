# F5-G / F5-X Confirmation Protocol

## Purpose

Confirm the two F5 variants that led the four-run screen before spending effort
on ablations or production integration. The confirmation matrix compares F5-G
and F5-X on the same five grouped folds and three seeds used by the completed
F2/F4 baseline matrix.

## Fixed run matrix

- Candidates: `F5-G`, `F5-X`
- Folds: `0, 1, 2, 3, 4`
- Seeds: `20260920, 20260921, 20260922`
- Total: 30 runs
- Initialization: matching fold-and-seed F2 checkpoint
- Device default: PyTorch MPS
- Training: cached joined-ten features; batch size 1024; four data workers;
  up to 50 epochs; patience 7; minimum 8 epochs
- Technical auxiliary objective: off
- Signal dropout: off
- Data access: train and validation partitions only; locked test is prohibited
- Output root: `/Users/rushivyas/datasets/image-aesthetics/fusion/experiments/f5-confirmation-v1`

The separate versioned root preserves the screen artifacts. Run identity binds
the split manifest, fold normalizer, joined-cache identities, model architecture,
matching F2 initialization, and full training protocol. Completed matching runs
are reusable; checkpoints can resume interrupted runs; failed runs require the
explicit `--retry-failed` option. Identity mismatches are never overwritten.

## Comparison and decision

Report macro and per-dataset Spearman, pairwise accuracy, top-1/5/10 lift and
overlap, MAE, and RMSE. Pair each candidate with F2 and F4 for the same fold and
seed. Summaries should show fold and seed variation and must not treat seeds as
independent datasets. Review TAD66K separately because the screen showed a
potential regression there.

Advance a candidate to focused ablations only if its ranking and top-selection
gains persist across folds, variability is acceptable, and no dataset has a
material unexplained regression. MAE is a calibration diagnostic and should not
alone determine selection. Do not access the locked test until architecture,
training, calibration, and candidate selection are frozen.
