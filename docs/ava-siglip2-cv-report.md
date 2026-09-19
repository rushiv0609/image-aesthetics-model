# AVA SigLIP2 head development CV

Completed 5 grouped folds × 3 seeds for A1 and A2 using only the materialized development partition. The locked test partition was never loaded. Run artifacts and the machine-readable aggregate report are stored outside the repository.

| Head | Spearman mean ± std | Pairwise accuracy | EMD | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| A1 | 0.71512 ± 0.00332 | 0.76149 | 0.005629 | 0.04344 | 0.05606 |
| A2 | 0.71524 ± 0.00361 | 0.76151 | 0.005610 | 0.04349 | 0.05612 |

Paired A2−A1 Spearman difference across identical fold/seed runs was +0.00012 ± 0.00090; A2 won 10 of 15 pairs. This is practically negligible relative to run variation, while A2 has materially more parameters. A1 is therefore the preferred finalist for the next training stage unless a later integration constraint favors A2.

The CV supervisor skips completed identity-matching runs, preserves interrupted run directories, retries transient failures up to twice, atomically updates aggregate status, and records a clear failed state. Fold IDs are 0–4 as specified by the existing assignment manifest.

## Final A1 release (locked)

A1 was selected before opening the locked test because its development-only CV performance was practically tied with A2 while using materially fewer parameters. The finalization policy was frozen from development CV: train on the complete 90% development partition (`development.jsonl`, 229,890 rows) for 9 epochs, the median A1 best epoch across the 15 grouped fold/seed runs. No test labels were read during selection or fitting.

The locked 10% test (`locked-test.jsonl`, 25,612 rows) was evaluated exactly once after fitting:

| Metric | Locked test |
|---|---:|
| Spearman | 0.771114 |
| Pearson | 0.774713 |
| Pairwise accuracy | 0.788921 |
| EMD | 0.005106 |
| MAE | 0.039930 |
| RMSE | 0.051850 |
| Top-1% lift | 1.307618 |
| Top-5% lift | 1.071030 |
| Top-10% lift | 0.939685 |
| Calibration bias | 0.011317 |
| Calibration slope | 0.600268 |

Final status: **A1 is finalized for integration testing.** The immutable release artifacts are stored at `/Users/rushivyas/datasets/image-aesthetics/ava/releases/siglip2-aesthetic/a1-v1/` (`model.safetensors`, `model-config.json`, `preprocessing.json`, `metrics.json`, and `provenance.json`). Model SHA-256: `b921e8568db6229a5a89810dd5c5823e755261acb9d0e91eb83fc35f594aabf1`.
