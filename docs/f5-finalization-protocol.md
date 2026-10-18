# F5 Final Fit and Single Locked Evaluation

This protocol is the terminal, one-way stage after candidate selection, the
12-run ablation gates, shared OOF calibration, LODO transfer analysis, and the
Moments benchmark decision are complete. The locked test is not used to choose
architecture, preprocessing, objective, epochs, seeds, calibrator, checkpoint
rule, or thresholds.

## Frozen policy

Before fitting, create a new external JSON policy with schema
`f5-frozen-policy-v1`. It must freeze:

- one selected F5 candidate;
- fixed F2-initializer and candidate epoch counts and unique seeds;
- the shared isotonic calibration artifact and its SHA-256;
- exact joined-feature hashes and the split-manifest SHA-256;
- architecture, preprocessing, training objective, fixed-epoch checkpoint
  policy, and every decision threshold;
- SHA-256 digests for the ablation, calibration, LODO, and Moments evidence used
  to make the selection;
- the training dataset list, batch size, ranking weight, and ranking gap.

`image_aesthetics.training.f5_finalization.freeze_policy` writes the policy and
its adjacent `.sha256` digest with exclusive creation, flushes both to disk, and
makes them read-only. Existing policy paths are never replaced. Both final
commands verify the digest sidecar and reject incomplete policy objects.

## Full-development fit

The default command is planning only. With `--execute`,
`scripts/training/run_f5_final.py` checks the frozen split and feature-cache
hashes, fits feature normalization on every non-locked development row, then
trains an F2 scalar initializer and the selected F5 model for the exact frozen
epoch counts and each frozen seed. The F5 candidate is initialized from its
matching full-development F2 fit. Training uses cached joined-ten features,
dataset-balanced batches, training-only dataset calibration heads, calibrated
Huber loss, and the frozen compatible-pair ranking loss.

There is no validation dataset, early stopping, epoch selection, or checkpoint
selection in this phase. Only each seed's last fixed-epoch checkpoint is
exported. `fit-status.json` and `fit-manifest.json` bind the checkpoints to the
policy, split, caches, normalization, seeds, and epoch counts. The report marks
`locked_test_accessed: false`. Fit output must be outside the repository, in a
dedicated artifact directory that does not replace the dataset root.

Example, after the policy has been frozen:

```sh
python scripts/training/run_f5_final.py \
  --policy /path/outside-repo/f5-policy.json \
  --output-root /path/outside-data/f5-final-fit
```

Add `--execute` only to launch the full-development fit. No final fit is
launched while policy decisions are still changing.

## One-time locked evaluation

The evaluation command also plans by default; planning reads only the frozen
policy. Execution requires a completed fit manifest, verifies its policy and
checkpoint hashes, and atomically creates the one fixed external receipt at
`<data-root>/fusion/experiments/f5-finalization/locked-test-evaluation.receipt.json`
using exclusive file creation before opening the locked partition. The path is
derived from the data root, so callers cannot choose another receipt path to
retry. A second attempt fails immediately. A started, failed, or interrupted
receipt is consumed permanently; recovery means retaining that evidence and
reviewing it, never rerunning against the locked labels.

The evaluator then rechecks split, feature-cache, normalizer, and calibrator
hashes, loads only the locked partition, averages the frozen seed predictions,
applies the frozen shared isotonic map, and writes raw and calibrated aggregate
metrics. Successful completion updates the receipt with the report digest. Any
error after receipt creation records `failed_closed` and re-raises. Keep the
policy, fit, receipt, and report outside the repository; use dedicated external
artifact paths rather than overwriting source manifests or caches.

```sh
python scripts/evaluation/evaluate_f5_locked_once.py \
  --policy /path/outside-repo/f5-policy.json \
  --fit-root /path/outside-data/f5-final-fit \
  --report /path/outside-data/f5-locked-report.json
```

Add `--execute` for the one allowed evaluation attempt. The locked labels are
read only after the policy and final fit are verified and the durable receipt
exists. There is no `--allow-locked-test` bypass or retry mode.

## Provenance and limits

The receipt is an external single-use guard, not a cryptographic remote ledger;
the operator must preserve it with the final report and policy. The model fit
does not use locked labels for gradient updates, calibrator fitting, epoch
selection, or checkpoint selection. This repository does not create human
ratings for the Moments Generator benchmark. Any blinded-human comparison
remains pending until independent people provide real ratings.
