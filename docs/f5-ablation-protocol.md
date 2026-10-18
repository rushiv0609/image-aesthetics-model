# F5-G focused ablation screen

## Purpose and boundary

This 12-run development screen tests three explanations for the completed F5-G
results: an overly strong bounded residual, a useful content gate, and noisy
small-difference ranking pairs. It uses the existing joined-ten cache and
F5-G/F2 artifacts. The scorer remains query-independent and uses the existing
ten signals. This is a train/validation experiment only; no locked-test data
is read or selectable by the runner.

The protocol is executable as a read-only plan with
`scripts/training/run_f5_ablation.py`; execution requires the explicit
`--execute` flag. Artifacts are isolated under
`fusion/experiments/f5-ablation-screen-v1`.

## Fixed design

- Candidates: `F5-G-R15`, `F5-G-NoGate`, `F5-G-Pair05`.
- Folds: 2 and 3, selected for the strongest observed TAD66K regression and
  strongest Flickr-AES gain, respectively.
- Seeds: 20260920 and 20260921.
- Total: 3 candidates × 2 folds × 2 seeds = 12 runs.
- Matching controls: completed F5-G runs from
  `f5-confirmation-v1` for the identical fold and seed (four controls).
- F2 initialization: same fold and seed as each candidate.
- Device and budget: MPS, batch size 1024, four workers, up to 50 epochs,
  patience 7, minimum 8 epochs.
- Objective: calibrated Huber plus 0.1 compatible-group pairwise ranking.
- Auxiliary technical task and signal dropout: off.
- Pair filtering: for `F5-G-Pair05` only, the ranking term ignores pairs whose
  absolute difference in normalized [0,1] training targets is below 0.05.
  The Huber term still uses every training example. Evaluation metrics are
  computed on every validation example and pair.

### Candidate definitions

| Candidate | Single change | Expected diagnostic |
| --- | --- | --- |
| `F5-G-R15` | Reduce `tanh` residual bound from 0.25 to 0.15 | Tests whether limiting F2-logit correction improves TAD66K ranking and calibration. |
| `F5-G-NoGate` | Fix scalar gate to exactly one and remove its trainable module | Tests whether content-conditioned scalar gating adds value. Embedding branches and residual fusion head remain identical. The removed gate accounts for 4,392 parameters, about 1.5% of G; this is a near-capacity-matched ablation, not an exact parameter match. |
| `F5-G-Pair05` | Ignore normalized target gaps below 0.05 in ranking loss | Tests whether ambiguous, near-tied labels add ranking noise, especially for TAD66K. Regression supervision is unchanged. |

No other loss, feature, preprocessing, split, optimizer, or training-budget
changes are allowed in this screen. Each config identity includes the candidate
architecture and protocol, cache and split hashes, normalizer hash, matching
F2 initialization identity, and matching F5-G control identity/checkpoint hash.
The runner refuses to overwrite artifacts with a different identity and only
reuses completed checkpoints with matching metadata.

## Predeclared advancement gates

At most one candidate advances to full confirmation. It must meet every gate:

1. Paired macro Spearman improvement over F4 is at least +0.030 on these four
   runs.
2. Mean paired TAD66K Spearman change versus F4 is greater than −0.0125,
   improving on the observed approximately −0.0246 regression.
3. Paired macro top-5 lift versus matching F5-G is no worse than −0.002.
4. Paired macro MAE versus matching F5-G is no worse than +0.002.
5. Mean Spearman change versus F5-G is positive within each diagnostic fold;
   a gain confined to one fold is insufficient.
6. Metrics are finite, runs complete, and no unexplained early-epoch collapse
   or provenance inconsistency occurs.

The aggregate stores per-run paired deltas, per-dataset results, fold-wise
changes, and an explicit gate evaluation. Seeds are repeated fits within folds,
not independent datasets. These small-screen thresholds are operational
advancement rules, not statistical significance claims. The screen alone does
not support a production choice or locked-test evaluation.

If none passes, retain the current F5-G and treat the result as evidence
against these specific changes. If multiple pass, advance the simplest
candidate with the strongest fold-consistent TAD66K and top-5 result; do not
combine ablations in this screen.

## Next phase after this screen

Confirm only the selected candidate over all five grouped folds and three
seeds (15 runs), paired with the already completed F5-G, F2, and F4 results.
Then fit and evaluate one shared monotonic calibrator using out-of-fold
development predictions. Continue to leave-one-dataset-out evaluation and the
event-grouped Moments Generator human benchmark. Freeze architecture,
training, calibration, and candidate selection before any single locked-test
evaluation.
