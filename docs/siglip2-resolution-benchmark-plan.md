# SigLIP 2 Base Resolution Benchmark Plan

## Status

Approved implementation plan. This document defines a benchmark only. It does not authorize changing the Moments Generator production default.

## Decision Being Evaluated

Determine whether Moments Generator should replace its current SigLIP 2 Base 224 encoder with the matching SigLIP 2 Base 384 encoder for all image embedding, text-to-image retrieval, ingestion, and later aesthetic-model inputs.

The comparison is strictly:

| Candidate | Hugging Face model | Resolution | Patch size | Embedding dimension |
|:---|:---|---:|---:|---:|
| Current | `google/siglip2-base-patch16-224` | 224×224 | 16 | 768 |
| Challenger | `google/siglip2-base-patch16-384` | 384×384 | 16 | 768 |

Do not substitute an SO400M checkpoint. In particular, `mlx-community/siglip2-so400m-patch16-224` is not the model currently used by Moments Generator. SO400M changes model capacity and embedding dimension as well as resolution, so it requires a separate experiment.

## Fixed Product Decisions

- If 384 is later accepted, it replaces 224 throughout Moments Generator rather than becoming a second permanent SigLIP encoder.
- Use the existing 1,000-photo ingestion benchmark corpus.
- Keep the current decode, resize, aspect-ratio, padding, normalization, and ingestion behavior unchanged for this comparison.
- Run the production-style benchmark with the custom MLX implementation on the Apple M5 Pro.
- Use Hugging Face PyTorch MPS as the correctness reference.
- Collect evidence first; do not impose a slowdown threshold before measurement.
- Do not change the production model default automatically. The user will review the report and make the promotion decision.

## Non-Goals

- Training the SigLIP aesthetic MLP
- Comparing Base with Large, SO400M, Giant, or NaFlex
- Redesigning image preprocessing
- Changing CLIP, NIMA, LAION, MUSIQ, or Qdrant behavior
- Re-indexing existing Moments workspaces
- Optimizing or quantizing the 384 model
- Proving that resolution alone improves aesthetics without later aesthetic-head training

## Important Interpretation Constraint

Higher resolution exposes more visual patches but does not by itself prove better pooled embeddings for Moments Generator. The report must separate:

1. **Systems behavior:** latency, throughput, memory, stability, and thermal effects.
2. **Backend correctness:** whether MLX reproduces the corresponding PyTorch checkpoint.
3. **Retrieval behavior:** whether existing text-to-image search remains healthy.
4. **Aesthetic potential:** deferred until identical aesthetic heads are trained on 224 and 384 caches.

The benchmark may establish that 384 is operationally acceptable and does not regress retrieval. It cannot claim superior aesthetic accuracy without the later controlled training experiment.

## Repository Boundaries

### Implementation repository

Benchmark code belongs in:

```text
/Users/rushivyas/development/projects/moments-generator
```

### Planning repository

This plan and the eventual summarized decision belong in:

```text
/Users/rushivyas/development/projects/image-aesthetics-model
```

Do not modify unrelated uncommitted work in Moments Generator. Inspect `git status` before editing and keep benchmark changes narrowly scoped.

## Existing Components to Reuse

- `app/core/embedder.py`
  - `MLXEmbedder`
  - `PyTorchMPSEmbedder`
  - current normalization and public embedding interface
- `app/core/mlx_siglip2.py`
  - custom MLX vision implementation and Hugging Face weight conversion
- `scripts/compare_mlx_vs_pytorch.py`
  - current parity and throughput benchmark patterns
- `scripts/compare_mlx_8bit.py`
  - benchmark reporting patterns only; 8-bit is out of scope
- `scripts/telemetry_utils.py`
  - report persistence and memory helpers
- `scripts/benchmark_semantic_alignment.py`
  - semantic evaluation patterns
- `tests/assets/semantic_test/`
  - existing semantic regression set

Do not copy and fork the entire MLX model into a 384-specific implementation. Generalize the existing implementation from model configuration.

## Expected Engineering Issue

The custom MLX implementation currently assumes 224-resolution patch geometry in places. With patch size 16:

- 224 produces `14 × 14 = 196` visual tokens.
- 384 produces `24 × 24 = 576` visual tokens.

Luna must identify and remove hard-coded image size, patch grid, position count, dummy input shape, and output assumptions. Model configuration must drive these values.

The two official checkpoints have distinct learned positional embeddings. Do not resize or interpolate the 224 positional embeddings into 384. Load each checkpoint's own weights.

## Proposed Implementation Layout

Prefer these changes unless repository inspection reveals a cleaner existing abstraction:

```text
moments-generator/
├── app/core/mlx_siglip2.py
│   └── generalize image size, patch size, grid, and position count
├── app/core/embedder.py
│   └── accept either approved Base model ID without changing defaults
├── scripts/benchmark_siglip2_resolution.py
│   └── new isolated 224-vs-384 benchmark entry point
├── tests/test_embedder.py
│   └── configuration, shape, normalization, and parity tests
└── data/benchmarks/
    ├── siglip2_resolution_<timestamp>.json
    └── siglip2_resolution_<timestamp>.md
```

If generated benchmark files are normally ignored, preserve that convention. Do not commit private image paths, filenames, thumbnails, or embeddings.

## CLI Contract

The new script should expose a reproducible non-interactive interface similar to:

```bash
python scripts/benchmark_siglip2_resolution.py \
  --folder /absolute/path/to/benchmark/photos \
  --count 1000 \
  --warmup-batches 5 \
  --repetitions 3 \
  --batch-sizes 1 8 16 32 \
  --output data/benchmarks/siglip2_resolution
```

Requirements:

- Require the corpus path explicitly or preserve an already documented local default without exposing it in reports.
- Sort the selected file list deterministically before truncating to `--count`.
- Record a non-reversible corpus fingerprint rather than private filenames.
- Support running one candidate at a time so a failed 384 run does not discard completed 224 results.
- Resume or reuse completed raw runs where practical.
- Record exceptions and skipped inputs explicitly.

## Benchmark Environment Capture

Each JSON report must include:

- timestamp and duration
- git commit plus dirty-worktree flag
- macOS version
- Apple chip identifier and available memory
- Python version
- PyTorch, Transformers, MLX, NumPy, and Pillow versions
- model IDs and immutable Hugging Face revisions
- configured precision
- batch sizes and repetition count
- corpus count and fingerprint
- preprocessing description
- whether power mode or thermal state was observable

Do not silently benchmark floating `main` revisions. Resolve and record the exact checkpoint commit for each model.

## Benchmark Phases

### Phase 0: preflight

1. Verify the corpus contains at least 1,000 readable supported images.
2. Build one deterministic ordered list used by both candidates.
3. Validate available disk space for both checkpoints and reports.
4. Record the environment and initial thermal state.
5. Confirm the production default remains 224.

Fail before model execution if the two candidates would receive different image lists.

### Phase 1: generalize MLX model loading

1. Read `image_size`, `patch_size`, hidden size, layer count, attention heads, and intermediate size from the loaded configuration.
2. Derive patch-grid dimensions and number of positions.
3. Allocate model components from configuration rather than 224/Base constants where feasible.
4. Load each checkpoint's own vision and text weights.
5. Retain the existing public output contract: a two-dimensional float32 NumPy array of L2-normalized embeddings.
6. Do not change the default model in application settings.

Exit checks:

- 224 still produces `(N, 768)`.
- 384 produces `(N, 768)`.
- finite values only.
- embedding norms are within tolerance of 1.0.
- text and image embeddings from the same candidate remain in the same embedding space.

### Phase 2: backend correctness

For each resolution independently, compare custom MLX FP16 against Hugging Face PyTorch MPS using the same decoded pixels and preprocessing.

Measure across a representative subset, not one random synthetic image:

- cosine similarity per image
- mean, median, standard deviation, minimum, and percentiles
- embedding L2 error
- maximum absolute coordinate difference
- norm drift
- repeat-run determinism
- text-embedding parity for the semantic test queries
- image-text score differences

Do not describe results as “exact” unless they are actually bit-identical. Report measured tolerances.

Exit gate: no unexplained systematic backend drift. A failed parity gate invalidates subsequent MLX performance conclusions until corrected.

### Phase 3: model loading and warm-up

For each candidate and backend, measure:

- checkpoint download/cache status
- cold load time
- warm cached load time where reproducible
- first-inference latency
- configured warm-up time
- process resident memory before load
- memory after load
- peak memory during first inference
- memory after unload/cleanup where supported

Separate network download time from model initialization. The headline load comparison must use locally cached checkpoints.

### Phase 4: pure model throughput

Use preprocessed in-memory batches so decoding is excluded.

For batch sizes `1, 8, 16, 32`, and optionally the largest stable batch:

- warm up first;
- run at least three timed repetitions;
- synchronize the backend before stopping the timer;
- report images/second;
- report milliseconds/image;
- report median and variability;
- report peak memory;
- record out-of-memory or instability rather than reducing the configured batch silently.

Use identical source pixels for both candidates, with only the required resolution-specific preprocessing differing.

### Phase 5: end-to-end 1,000-photo ingestion benchmark

Exercise the current production-style path from file read through decode, preprocessing, MLX embedding, L2 normalization, and cache/result collection.

Measure:

- total wall-clock time
- successfully processed images
- skipped/failed images
- end-to-end images/second
- decode/preprocessing time
- queue wait time if available
- active model time
- effective batch-size distribution
- peak process memory
- memory after completion
- thermal state or throttling evidence when observable
- throughput by repetition

Run candidates under comparable conditions. Alternate execution order across repetitions if practical (`224→384`, then `384→224`) to reduce warm-cache and thermal-order bias. Allow the machine to return to a comparable thermal state between long runs.

### Phase 6: semantic regression

Run the existing semantic test cases independently in each matching embedding space:

```text
224 text encoder → 224 image embeddings
384 text encoder → 384 image embeddings
```

Never compare a 224 text vector directly with a 384 image vector.

Report:

- top-1 accuracy
- recall at configured k values
- mean reciprocal rank
- nDCG where relevance labels support it
- per-query rank changes
- top-k overlap between candidate systems
- newly fixed and newly regressed queries
- raw score distributions separately for each checkpoint

Do not assume the current hard-coded cosine normalization range transfers to 384. Record raw distributions so any later ranking calibration can be decided explicitly.

### Phase 7: stability and determinism

For a fixed subset and fixed order:

- embed the same inputs at least three times;
- compare same-backend outputs;
- detect NaN, Inf, zero vectors, shape errors, and ordering mismatches;
- run at least one sustained workload long enough to reveal thermal throttling or memory growth;
- verify cleanup does not corrupt a subsequent model load.

## Required Metrics

### Systems table

| Metric | Base 224 | Base 384 | 384/224 ratio |
|:---|---:|---:|---:|
| Cached model load time | | | |
| First inference latency | | | |
| Model/process memory delta | | | |
| Batch-1 ms/image | | | |
| Batch-8 images/s | | | |
| Batch-16 images/s | | | |
| Batch-32 images/s | | | |
| End-to-end 1,000-photo time | | | |
| End-to-end images/s | | | |
| Peak process RAM | | | |
| Failures/skips | | | |

### Correctness table

| Metric | 224 MLX vs MPS | 384 MLX vs MPS |
|:---|---:|---:|
| Mean cosine similarity | | |
| Median cosine similarity | | |
| Minimum cosine similarity | | |
| 1st percentile cosine | | |
| Mean L2 error | | |
| Maximum coordinate error | | |
| Image-text score MAE | | |
| Determinism drift | | |

### Retrieval table

| Metric | Base 224 | Base 384 | Difference |
|:---|---:|---:|---:|
| Top-1 accuracy | | | |
| Recall@5 | | | |
| MRR | | | |
| nDCG | | | |
| Query regressions | | | |
| Query improvements | | | |

## Statistical Treatment

- Report individual repetition values as well as aggregates.
- Prefer median for latency and throughput headlines.
- Include min/max or standard deviation.
- Use the same image order for paired comparisons.
- Clearly label descriptive results from the 1,000-photo corpus; do not overstate them as universal conclusions.
- Preserve raw JSON measurements so later analysis does not depend on rounded Markdown tables.

## Test Plan

Add or extend tests for:

1. Configuration-derived patch grid for 224 and 384.
2. Expected positional-embedding count for each checkpoint.
3. Output shape `(N, 768)` for both resolutions.
4. L2-normalized finite outputs.
5. Batch ordering preservation.
6. Text and image embedding availability for both candidates.
7. MLX-versus-PyTorch parity on deterministic fixtures.
8. Benchmark report schema and required metadata.
9. Production configuration still defaults to Base 224.
10. Clear rejection of incompatible checkpoints or unsupported shapes.

Avoid tests that require downloading both full models on every normal unit-test run. Mark heavyweight integration tests explicitly and keep small configuration/report tests in the default suite.

## Report Requirements

Generate both:

```text
siglip2_resolution_<timestamp>.json
siglip2_resolution_<timestamp>.md
```

The Markdown report must begin with a neutral summary:

- what was compared;
- exact environment;
- whether parity passed;
- measured throughput and memory differences;
- semantic improvements and regressions;
- limitations;
- recommendation options without changing production.

The report should end with one of these evidence summaries:

- `384 is operationally viable; quality decision pending aesthetic-head experiment.`
- `384 has blocking parity or stability problems.`
- `384 has unacceptable measured resource cost for the observed retrieval behavior.`

Do not state that 384 has better aesthetic accuracy based only on resolution or retrieval metrics.

## Safety and Privacy

- Never include private filenames or absolute media paths in committed output.
- Do not copy benchmark photos into the repository.
- Do not upload the personal corpus.
- Do not log image pixels or thumbnails.
- Use hashes or anonymized ordinal IDs in raw per-image measurements.
- Keep all benchmark execution local on the M5 Pro.

## Luna Execution Sequence

1. Read Moments Generator documentation and inspect the dirty worktree before editing.
2. Confirm the current production model and preprocessing from code.
3. Add configuration-driven resolution support without changing defaults.
4. Add focused unit tests and run them.
5. Validate 224 behavior has not regressed.
6. Validate 384 MLX against PyTorch MPS on a small subset.
7. Run pure-throughput tests.
8. Run the full 1,000-photo benchmark with repeated/counterbalanced order.
9. Run semantic regression tests.
10. Generate JSON and Markdown reports.
11. Run `git diff --check` and relevant tests.
12. Stop without switching production to 384.

## Definition of Done

The benchmark implementation is complete when:

- both exact Base checkpoints load through the intended paths;
- MLX support is configuration-driven rather than duplicated;
- the current Base 224 default and behavior remain intact;
- correctness, load, throughput, memory, end-to-end, stability, and retrieval metrics are captured;
- the same deterministic 1,000-photo corpus is used for both candidates;
- private media information is absent from reports;
- JSON and Markdown reports are produced;
- relevant tests pass;
- no production promotion occurs;
- the user has enough evidence to decide whether to proceed to full 384 aesthetic-cache extraction and MLP training.

## Follow-Up Decision

After reviewing the report, choose one:

1. Keep Base 224.
2. Promote Base 384 and plan re-indexing plus 384 aesthetic-head training.
3. Run a larger quality benchmark before deciding.
4. Start a separate Base-versus-SO400M study.
