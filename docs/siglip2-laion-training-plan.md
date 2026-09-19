# SigLIP 2 Aesthetic Head: Training and Implementation Plan

## Objective

Train a small aesthetic predictor on frozen, L2-normalized SigLIP 2 image embeddings. The model will be the SigLIP counterpart of the existing CLIP-based LAION aesthetic predictor and will later become one input to the unified Moments Generator scorer.

The deliverable is an ingestion-safe PyTorch model that accepts one 768-dimensional SigLIP 2 embedding and returns one normalized aesthetic score in `[0, 1]`. Training and inference must work on Apple M5 Pro with 24 GB unified memory. CUDA may be used for an optional training run, but it must not become a dependency.

**Resolution decision:** use `google/siglip2-base-patch16-224`. The 384 benchmark showed substantially higher memory usage, so Base 224 is the fixed embedding contract for this training phase.

## System Architecture

```text
AVA images + vote histograms
            │
            ├── validate metadata and image availability
            ├── deduplicate/group related images
            └── assign immutable train/validation/test split
                              │
                              ▼
Frozen SigLIP 2 image encoder
google/siglip2-base-patch16-224
fixed model revision + fixed preprocessing
                              │
                              ▼
L2-normalized 768-d embedding per image
                              │
                              ├── sharded feature arrays
                              ├── labels/vote counts
                              └── manifest with hashes and provenance
                              │
                              ▼
Small MLP aesthetic head
768 → 512 → 128 → output
                              │
                  ┌───────────┴────────────┐
                  ▼                        ▼
        Scalar regression head      10-bin distribution head
        one normalized mean         probability per AVA rating
                  │                        │
                  └───────────┬────────────┘
                              ▼
Rank-based, top-k, distribution, and calibration evaluation
                              │
                              ▼
Best checkpoint + config + preprocessing contract + metrics
                              │
                              ▼
Moments Generator ingestion
SigLIP embedding → aesthetic head → stored Qdrant payload score
```

## Model Candidates

### Baseline 0: linear probe

```text
768 → 10
```

This is a required sanity baseline. It establishes whether the frozen embedding already contains linearly accessible aesthetic information.

### Baseline 1: scalar MLP

```text
Input embedding (768, L2 normalized)
→ LayerNorm(768)
→ Linear(768, 512)
→ GELU
→ Dropout(0.20)
→ Linear(512, 128)
→ GELU
→ Dropout(0.10)
→ Linear(128, 1)
→ Sigmoid
```

This model has approximately 462K parameters and predicts the normalized AVA mean directly.

### Primary candidate: distribution MLP

Use the same trunk, replacing the final scalar layer with `Linear(128, 10)` and softmax. The production score is the expected rating:

```text
raw_mean = Σ p(rating=i) × i, for i in 1..10
normalized_score = (raw_mean - 1) / 9
```

This remains under 0.5M parameters. It preserves information about rating disagreement while still producing the required scalar.

Do not start with the larger LAION `768 → 1024 → 128 → 64 → 16 → output` topology as the only baseline. It is still feasible, but the smaller head is easier to regularize and sufficient for testing whether SigLIP embeddings carry aesthetic information. Add the LAION topology as a controlled architecture comparison after the two baselines work.

## Losses

### Scalar baseline

- Primary training loss: Huber loss on normalized mean rating.
- Optional comparison: MSE.

Huber is less sensitive than MSE to noisy or polarizing labels.

### Distribution candidate

- Primary loss: squared Earth Mover's Distance on cumulative distributions.
- Auxiliary loss: Huber loss between predicted and true normalized means.
- Initial combination: `loss = EMD + 0.25 × mean_huber`.

The coefficient is a starting hyperparameter, not a fixed product decision. Compare against plain cross-entropy and scalar regression.

## Dataset and Split Rules

1. Parse AVA vote counts and compute both the normalized ten-bin distribution and normalized mean.
2. Create the split before embedding extraction and persist it in the manifest.
3. Keep test labels unavailable to model-selection code until the final evaluation.
4. Detect exact duplicates and, where metadata permits, related variants; keep each group in only one split.
5. Use one deterministic split seed and record it.
6. Do not train the SigLIP head on any image later used as validation/test data for the unified fusion model. The fusion project should either reserve a second untouched split now or use cross-fitting to generate out-of-fold SigLIP-head predictions.

Recommended initial split:

- 80% SigLIP-head training
- 10% validation
- 10% locked test

Before extracting all features, run the complete pipeline on a stratified 1,000-image smoke subset.

## Embedding Cache Contract

Each cache version must record:

- AVA image ID and source-relative path
- split (`train`, `validation`, or `test`)
- raw 10-bin vote counts, normalized distribution, mean rating, and vote total
- 768-dimensional L2-normalized embedding
- embedding dtype
- SigLIP model identifier and immutable model revision/commit
- processor configuration and library versions
- source image hash and extraction status/error
- cache schema version and creation timestamp

Recommended storage:

- sharded `safetensors` or NumPy arrays for embeddings;
- Parquet for IDs, labels, splits, hashes, and provenance;
- a small JSON manifest describing schema, model revision, preprocessing, shard hashes, and counts.

Float16 is acceptable for storage if a sample comparison shows negligible rank/score drift relative to float32. Convert batches to float32 for MLP training. For 255K images, one SigLIP cache is about 0.78 GB in float32 or 0.39 GB in float16, excluding metadata.

The cache is immutable and content-addressed. A changed model revision, preprocessing rule, or image set creates a new cache version instead of overwriting the old one.

## Training Configuration

Initial defaults:

| Setting | Value |
|:---|:---|
| Optimizer | AdamW |
| Learning rate | `3e-4` |
| Weight decay | `1e-4` |
| Batch size | 512 initially; increase if profiling supports it |
| Maximum epochs | 50 |
| Early stopping | 7 validation epochs without primary-metric improvement |
| LR schedule | cosine decay with 2 warm-up epochs |
| Gradient clipping | norm 1.0 |
| Training dtype | float32 initially |
| Seeds | at least 3 final runs |

The model is small enough that dataloader and metric overhead may dominate compute. Load only the current batch to the selected device; memory-map or shard the cache rather than coupling training to a monolithic in-memory format.

## Evaluation

### Primary metrics

- Spearman rank correlation between predicted score and AVA mean
- pairwise ordering accuracy on reproducibly sampled pairs
- top-k lift at 1%, 5%, and 10% relative to the dataset mean and baselines

### Secondary metrics

- Pearson correlation
- MAE, RMSE, and Huber loss on the normalized mean
- EMD/cross-entropy for distribution output
- score calibration by predicted-score decile

### Required baselines and ablations

- constant dataset-mean predictor
- linear probe
- scalar MLP
- distribution MLP
- CLIP-based LAION predictor evaluated on exactly the same split
- optional larger LAION-topology SigLIP MLP
- float16-cache versus float32-cache parity

After the locked AVA test, evaluate without retraining on TAD66K or another approved external aesthetic dataset and on a small Moments-style personal-media benchmark.

## Hardware and Portability

### Recommended execution location

Run the authoritative pipeline locally on the M5 Pro. This satisfies the repository constraint and proves that future reproduction does not depend on cloud infrastructure.

There are two very different workloads:

1. **Embedding extraction:** expensive frozen SigLIP vision inference over all images; likely hours and primarily worth accelerating.
2. **MLP training:** sub-500K-parameter training over a cache smaller than 1 GB; likely minutes per run and not a reason to rent a cloud GPU.

Cloud CUDA is optional for faster cache construction or large sweeps. If used, reproduce at least the winning configuration locally before acceptance.

### CUDA-to-MPS weight compatibility

Standard PyTorch `state_dict` tensors are portable across CUDA, MPS, and CPU. Save the unwrapped model state dictionary and load with `map_location="cpu"`, then move the model to MPS. Avoid serializing a CUDA-bound whole-model object or relying on custom CUDA-only operators.

Acceptance requires numerical-parity checks on a fixed embedding batch across CPU and MPS, and across CUDA too if cloud training is used. Small floating-point differences are normal; rankings and scores should remain within documented tolerances.

### Embedding-backend decision

Use the production **MLX FP16** encoder on the M5 Pro to build the authoritative training cache unless a representative parity study proves that a faster CUDA extractor is interchangeable.

Hardware backends can produce slightly different embeddings because of precision, kernel, and reduction-order differences. These differences are normally small when the checkpoint, preprocessing, pooling, and L2 normalization are identical, but the aesthetic head is trained on the coordinates themselves rather than only cosine similarities. The safest train/serve arrangement is therefore to extract training features with the same implementation and precision used during Moments Generator ingestion.

The existing Moments Generator MLX implementation converts the same Hugging Face SigLIP 2 weights, uses the same vision head, operates in FP16, and L2-normalizes the result. Its current comparison script is encouraging but is not a sufficient acceptance test because a single/random sample can miss content-dependent drift.

Before allowing CUDA-generated embeddings as the training cache, compare MLX FP16, PyTorch MPS FP16, and CUDA FP16 on at least 5,000 representative images. Record:

- per-image cosine similarity and L2 difference between embeddings;
- per-dimension mean and standard-deviation drift;
- nearest-neighbor overlap and ranking agreement;
- aesthetic-head score MAE, maximum error, Spearman correlation, and top-k overlap;
- failure slices by brightness, aspect ratio, monochrome content, and media source.

Suggested acceptance thresholds are median embedding cosine similarity at least `0.9999`, 1st-percentile similarity at least `0.999`, aesthetic-score MAE at most `0.002`, Spearman at least `0.999`, and top-10% overlap at least `99%`. These are starting thresholds and should be tightened or relaxed only from measured product impact.

Do not mix embeddings from different backends or precisions within one cache version. In particular, MLX 8-bit quantized embeddings require a distinct cache and a separate parity decision. If production will use 8-bit MLX, either train on that representation or demonstrate that a model trained on MLX FP16 remains equivalent on 8-bit inputs.

## Implementation Phases

### Phase 0: contracts and smoke data

- Pin SigLIP model revision and preprocessing.
- Define cache schema, config schema, and deterministic split manifest.
- Validate AVA metadata and licenses/source availability.
- Run end-to-end extraction and training on 1,000 images.

Exit gate: repeatable cache hashes/counts, no train/test overlap, and a model that overfits a tiny diagnostic subset.

### Phase 1: full cache

- Extract all embeddings once with resumable shards.
- Validate dimensions, norms, NaN/Inf absence, labels, missing images, and shard hashes.
- Produce a cache audit report.

Exit gate: all usable images accounted for and cache validation passes.

### Phase 2: baselines

- Train constant, linear, and scalar MLP baselines.
- Evaluate CLIP-LAION on the identical split.
- Establish deterministic metrics and experiment tracking.

Exit gate: scalar MLP reliably beats constant and preferably linear baselines across seeds.

### Phase 3: distribution model

- Train EMD plus mean-loss model.
- Compare distribution, scalar, and larger LAION topology.
- Tune only a small predefined search space.

Exit gate: select the simplest model that wins on validation ranking/top-k metrics and remains stable across seeds.

### Phase 4: locked and external evaluation

- Evaluate the selected configuration once on the locked AVA test.
- Run external-dataset and Moments-style evaluations.
- Document failure slices, including low light, monochrome, portraits, landscapes, blur, and video-transition frames.

Exit gate: evidence supports using the signal as a dependency of the later unified scorer.

### Phase 5: export and integration contract

- Export model weights, architecture config, normalization contract, and metric report.
- Add a minimal CPU/MPS inference wrapper and parity tests.
- Define the Qdrant payload key and model/cache version fields, but defer Moments Generator integration until the signal is accepted.

## Immediate Next Steps

1. Confirm the exact AVA source and local dataset path.
2. Pin the Hugging Face revision for `google/siglip2-base-patch16-224` and its processor.
3. Decide whether to reserve fusion-model holdout data now or use cross-fitting later; reserving now is simpler and safer.
4. Implement configuration, split creation, cache schema, and the 1,000-image smoke pipeline.
5. Benchmark local embedding throughput before considering cloud extraction.
6. Only after the benchmark, estimate the full local runtime and compare it with a concrete cloud quote.

## Open Questions Before Implementation

1. Where will the AVA images and annotations come from, and what exact annotation format is available?
2. Is research-only dataset licensing acceptable for this development phase, and what distribution/commercial constraints must the eventual trained artifact satisfy?
3. Should we reserve part of AVA exclusively for the later unified scorer now, or use out-of-fold SigLIP predictions when building that scorer?
4. Do we want experiment tracking to remain entirely local (recommended initially), or integrate an external service?
