# F5 Improvement Program Final Report

## Outcome

All automatable phases completed. `F5-G-v1` is the frozen final architecture.
It was fitted on all 124,952 non-locked development rows for three seeds, then
evaluated exactly once on the 22,295-row locked test. The one-time receipt is
complete. No locked row informed architecture, preprocessing, objective,
calibration, thresholds, epoch counts, or checkpoint selection.

The production output is the three-seed mean followed by the shared isotonic
calibrator. This improves overall MAE on the locked test, but raw scores remain
better for RMSE, pairwise accuracy, and top-1 selection. That trade-off is now
measured rather than inferred.

## Frozen system

- Architecture: `F5-G-v1`, 297,426 parameters, content-gated embedding residual
  over an F2 scalar base.
- Inputs: immutable `joined-ten-v1` cached feature contract; no query-dependent
  features.
- Training: MPS, batch size 1,024, balanced datasets, calibrated Huber plus 0.1
  compatible-pairwise loss.
- Final schedule: 32 F2 epochs then 34 F5-G epochs for seeds 20260920,
  20260921, and 20260922. These are medians of the confirmation best epochs.
- Checkpoint rule: last fixed epoch; no final-validation selection or early
  stopping.
- Calibration: shared development-OOF isotonic mapping.
- Frozen policy SHA-256: `ca5812c4b4a64887f57ab295cfcdc9ceac33f6c9df540b131cf281d962eabebd`.

## Selection evidence

The 15-run F5-G confirmation reached mean Spearman 0.62678, pairwise accuracy
0.73302, top-1 lift 0.24582, top-5 lift 0.19768, top-10 lift 0.16853, and MAE
0.11883. The targeted 12-run screen promoted none of R15, NoGate, or Pair05:
no option produced a consistent positive Spearman delta across both declared
validation folds without regression. The original F5-G was therefore retained.

Shared isotonic calibration reduced development OOF MAE from 0.11846 to
0.10678 and RMSE from 0.14571 to 0.13753, with a small Spearman change from
0.49584 to 0.49496. Its known caveat is worse TAD66K OOF MAE (0.11874 to
0.12866).

LODO transfer favored F5-G over F4 on three of four held-out datasets. Mean
Spearman deltas were +0.01529 PARA, +0.01296 Flickr-AES, +0.04596 TAD66K, and
-0.01846 AADB. The AADB regression remains the main generalization caveat.

## Locked-test result

| Metric | Raw ensemble | Isotonic production score |
|---|---:|---:|
| Spearman | 0.45385 | 0.45380 |
| Pairwise accuracy | 0.65563 | 0.64758 |
| MAE | 0.11831 | **0.11535** |
| RMSE | **0.14504** | 0.14593 |
| Top-1 lift | **0.21265** | 0.19788 |
| Top-5 lift | **0.15822** | 0.15681 |
| Top-10 lift | 0.13254 | **0.13308** |

Calibrated per-dataset Spearman was 0.69502 AADB, 0.69293 Flickr-AES, 0.82860
PARA, and 0.46349 TAD66K. Calibrated MAE was 0.12799, 0.11051, 0.05448, and
0.12936 respectively. Calibration materially helps Flickr-AES and PARA
absolute error, but hurts TAD66K absolute error and most rank/selection metrics.
Consumers that only need ordering should retain access to the raw ensemble
score; the calibrated score is appropriate for the required [0,1] production
quality value.

## Moments Generator benchmark

The automated pipeline produced complete F0, F1-mean, F4, raw F5-G, and
calibrated F5-G predictions for all 1,598 decode-valid private media items in
21 events. Six non-media/mislabeled files were excluded by label-blind decode
validation. All cache identities, checkpoints, normalizers, and prediction
hashes are recorded, and neither locked labels nor human ratings were used.

The blinded-human product benchmark remains pending real raters. Consequently,
pairwise preference, keep/reject severe-false-positive rate, and human top-k
overlap are intentionally null. No human-facing superiority claim over F0, F1,
or F4 is supported until those annotations exist.

## Artifact inventory

- Frozen policy: `fusion/experiments/f5-finalization/frozen-policy-v1.json`
- Final checkpoints: `fusion/experiments/f5-finalization/final-fit-v1/`
- Locked report: `fusion/experiments/f5-finalization/locked-test-report-v1.json`
- One-time receipt: `fusion/experiments/f5-finalization/locked-test-evaluation.receipt.json`
- Moments predictions: `fusion/experiments/moments-benchmark-v2/predictions-v1/`

These paths are relative to `/Users/rushivyas/datasets/image-aesthetics` and
are deliberately outside Git because they include generated model artifacts
and private benchmark provenance.

## Remaining work requiring people

Only the blinded Moments rating exercise is non-automatable: independent
raters must provide event-local pairwise preferences, keep/reject labels, and
top-k choices. After those real annotations are collected, the existing
evaluator can populate the pending product metrics without retraining or
reopening the locked academic test.
