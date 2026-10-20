# F5-G v1 Release and Integration Contract

## Release status

`fusion/releases/f5-g-v1/` is the final exported model. The bundle contains the
three fixed-epoch F5-G seed members, full-development normalizer, shared
isotonic calibrator, frozen policy, feature contract, locked metrics, and
provenance. `manifest.json` binds every payload file by SHA-256.

The model is deliberately an ensemble: the three 1.19 MB `safetensors` files
are averaged before calibration. Do not select a seed at runtime. Do not load
the F2 initializer checkpoints; F2 was used only to initialize final training.

## Input contract

The scorer accepts exactly:

- a 768-float L2-normalized SigLIP 2 embedding;
- a 768-float L2-normalized CLIP ViT-L/14 embedding;
- eight finite scalar signals named `a1_siglip`, `laion_clip`, `nima`,
  `musiq_aesthetic`, `musiq_technical`, `sharpness`, `contrast`, and
  `colorfulness`.

There is no cosine similarity or other query-dependent signal. Feature
assembly must use `F5ReleaseScorer.feature_tensor`; hand-written concatenation
is not an accepted integration path.

## Output contract

The default production output is the shared-isotonic calibrated score in
`[0,1]`. The raw ensemble score may be used in shadow ranking experiments.
On the single locked test, calibration improved overall MAE from 0.11831 to
0.11535, while raw output was slightly stronger for pairwise accuracy and
top-1 lift. Preserve both during the initial Moments shadow run.

## Moments Generator rollout

1. Vendor or install this package and copy the complete release directory
   without changing filenames.
2. Verify the manifest at startup and expose the model version `f5-g-v1`.
3. Compute the score during ingestion after all ten signals are available.
4. Store calibrated score, raw score, and model version in the Qdrant payload.
5. Keep the existing weighted scorer as a rollback path during shadow mode.
6. Compare existing, F4, raw F5-G, and calibrated F5-G selections per event.
7. Promote only after real blinded ratings satisfy the gates in
   `docs/moments-benchmark-protocol.md`.

Existing Qdrant records cannot be assumed compatible. A release rollout needs
a full score backfill or re-index because the learned score and model version
must be present for every candidate.

## Verification

```bash
python -m pytest -q tests/inference/test_f5_release.py
```

The test loads all three members, verifies release hashes, checks strict signal
ordering, and executes an end-to-end CPU prediction. MPS parity should be
tested in Moments Generator on the actual deployment environment.

## Human benchmark boundary

Automated predictions already exist for all 1,598 decode-valid private Moments
items. Human pairwise, keep/reject, and top-k fields remain intentionally null.
Only real raters may complete them. Do not synthesize or infer those labels,
and do not reopen the academic locked test when the human benchmark finishes.
