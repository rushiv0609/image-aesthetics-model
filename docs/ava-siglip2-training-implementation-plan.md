# AVA SigLIP 2 Base 224 Training Implementation Plan

## Status and Goal

This is the implementation plan for the first learned dependency of the Moments Generator unified scorer: a SigLIP 2 aesthetic head trained on AVA.

The embedding contract is now fixed to:

```text
google/siglip2-base-patch16-224
768-dimensional L2-normalized image embedding
production-compatible MLX FP16 extraction on Apple Silicon
```

The trained head must accept one cached 768-dimensional embedding and return:

- a ten-bin aesthetic-rating distribution for ratings 1 through 10; and
- a derived scalar score in `[0, 1]` for ingestion-time storage.

This work is separate from the later multi-signal fusion model. Dataset tooling, embedding extraction, head training, and evaluation should have distinct packages and output directories.

## Dataset Source

### Recommended source

Use the Hugging Face mirror `Iceclear/AVA`, which currently provides a roughly 33.2 GB `AVA_dataset.zip` bundle containing images and AVA data. Pin the dataset repository revision in the local manifest rather than downloading an unversioned moving target.

Source:

```text
https://huggingface.co/datasets/Iceclear/AVA
```

The mirror is operationally convenient, but the original AVA image rights remain with the photographers and DPChallenge. Do not redistribute the images or commit them to this repository. Record the source and terms reviewed for every acquisition.

### Fallback source

If the mirror is unavailable or incomplete, use the original AVA annotations and reconstruct image URLs from DPChallenge IDs. The original annotation documentation defines URLs in this form:

```text
http://www.dpchallenge.com/image.php?IMAGE_ID=<Image_ID>
```

Downloader repositories are convenience tools, not authoritative annotation sources. Validate all resulting image IDs against `AVA.txt`.

### Do not confuse datasets

The Google AVA Actions dataset is a video action-recognition dataset and is unrelated to AVA Aesthetic Visual Analysis. Reject any download containing movie clips, action boxes, or action IDs.

## Storage Outside the Repository

Recommended root:

```text
/Users/rushivyas/datasets/image-aesthetics/ava
```

Make the location configurable with:

```text
IAM_DATA_ROOT=/Users/rushivyas/datasets/image-aesthetics
```

Never place images, archives, embeddings, or trained weights inside the Git repository. Only schemas, small manifests without private paths, configuration examples, and aggregate reports belong in Git.

Recommended external layout:

```text
$IAM_DATA_ROOT/
└── ava/
    ├── downloads/
    │   ├── AVA_dataset.zip
    │   └── download-manifest.json
    ├── raw/
    │   ├── annotations/
    │   │   ├── AVA.txt
    │   │   ├── tags.txt
    │   │   └── challenges.txt
    │   └── images/
    ├── prepared/
    │   └── v1/
    │       ├── dataset.parquet
    │       ├── rejected.parquet
    │       ├── splits.json
    │       └── audit.json
    ├── embeddings/
    │   └── siglip2-base-patch16-224/<cache-version>/
    │       ├── manifest.json
    │       ├── metadata.parquet
    │       ├── shard-00000.safetensors
    │       └── shard-*.safetensors
    ├── experiments/
    │   └── siglip2-aesthetic/<run-id>/
    │       ├── config.json
    │       ├── metrics.json
    │       ├── checkpoints/
    │       └── logs/
    └── releases/
        └── siglip2-aesthetic/<version>/
            ├── model.safetensors
            ├── model-config.json
            ├── preprocessing.json
            ├── metrics.json
            └── provenance.json
```

Expected free-space budget should be at least 80 GB for the archive, extracted images, temporary extraction space, embeddings, and experiment artifacts. Confirm actual archive contents before extraction because temporary peak usage may exceed the final dataset size.

## Repository Code Layout

Keep use cases separate:

```text
image-aesthetics-model/
├── configs/
│   ├── data/ava.yaml
│   ├── embedding/siglip2_base_224.yaml
│   └── training/siglip2_aesthetic.yaml
├── src/image_aesthetics/
│   ├── data/                 # acquire, parse, validate, split
│   ├── embeddings/           # frozen encoder and cache generation
│   ├── models/siglip_head/   # MLP definitions only
│   ├── training/             # losses, loop, checkpointing
│   ├── evaluation/           # metrics, slices, reports
│   └── common/               # config, hashing, device selection
├── scripts/
│   ├── data/prepare_ava.py
│   ├── embeddings/cache_siglip2_ava.py
│   ├── training/train_siglip2_head.py
│   └── evaluation/evaluate_siglip2_head.py
└── tests/
    ├── data/
    ├── embeddings/
    ├── models/
    ├── training/
    └── evaluation/
```

Do not reuse the resolution-benchmark package for training.

## Phase 1: Acquire and Verify AVA

1. Download the archive into `downloads/` with a resumable downloader.
2. Record source URL, repository revision, download timestamp, byte size, and SHA-256.
3. Extract into a temporary directory under the external dataset root.
4. Validate expected annotation files before moving into `raw/`.
5. Inventory images by AVA image ID without depending on filesystem order.
6. Reject corrupt, unreadable, missing, zero-sized, or duplicate files into `rejected.parquet` with reason codes.
7. Never silently drop an annotated image.

The preparation audit must report:

- annotation row count;
- unique image IDs;
- available and missing images;
- exact duplicate groups by content hash;
- rating-vote totals and invalid vote rows;
- mean-score histogram;
- challenge/category counts;
- final usable count.

## Phase 2: Normalize Labels

For every valid image, retain the raw AVA columns and derive:

```text
vote_counts[10]       raw votes for scores 1..10
vote_total            sum(vote_counts)
distribution[10]      vote_counts / vote_total
mean_rating           Σ distribution[i] × (i + 1)
normalized_mean       (mean_rating - 1) / 9
```

Do not turn AVA into only a high/low classification dataset. The score distribution and mean are both required.

## Phase 3: Leakage-Resistant Splits

### Grouping rule

Split by AVA photography challenge ID where available, not by individual image. Images from the same challenge may share subject matter, rules, participants, and visual style. Keeping a challenge in one split reduces content leakage.

Additionally, keep exact duplicate hashes and known related variants in one group.

### Fixed outer split

Create one immutable split manifest:

| Partition | Approximate share | Purpose |
|:---|---:|:---|
| Development train | 80% | Parameter fitting |
| Development validation | 10% | Early stopping and routine comparison |
| Locked test | 10% | One-time final evaluation |

Use grouped, score-stratified allocation. Balance normalized-mean score deciles as closely as group constraints allow. Record the algorithm version and seed.

The locked test labels must not be loaded by the training or hyperparameter-selection entry points.

### Hyperparameter validation

Avoid selecting a model from one lucky validation split:

1. Use the fixed validation set for rapid development and debugging.
2. For serious candidates, run five-fold `GroupKFold` or stratified-group cross-validation inside the 90% development portion.
3. Train each finalist with at least three random seeds.
4. Rank candidates by mean validation Spearman/top-k performance and report standard deviation.
5. Select architecture and hyperparameters before opening the locked test set.
6. Evaluate the selected configuration on the locked test exactly once.

Do not repeatedly tune against test results. A test failure leads to a documented new model version and a new external benchmark, not continued tuning on the same test.

### Future fusion leakage

The later unified scorer must not receive in-sample predictions from this SigLIP head. Choose one of:

- generate out-of-fold SigLIP-head scores for fusion training; or
- reserve a fusion-training partition never used to fit this head.

Out-of-fold predictions preserve more AVA data and are recommended once the basic pipeline is stable.

## Phase 4: Pre-Embedding Generation

### Authoritative extractor

Use the production-compatible MLX FP16 implementation of `google/siglip2-base-patch16-224`, pinned to an immutable Hugging Face revision.

Use the same image contract intended for Moments Generator:

- apply EXIF orientation;
- convert to RGB;
- preserve aspect ratio;
- fit within 224×224;
- pad rather than crop;
- normalize with SigLIP mean/std (`0.5`, `0.5`);
- run the official Base 224 vision head;
- L2-normalize the 768-dimensional output.

Record the exact decoder and interpolation mode. Decoder differences must create a new cache version.

### Cache generation behavior

- Process deterministic shards, for example 4,096 images per shard.
- Write a shard atomically only after validating every included row.
- Resume by verifying completed shard hashes.
- Record per-image failures without shifting row alignment.
- Store explicit image IDs alongside embeddings.
- Never rely on array position alone to join labels.
- Validate shape `(N, 768)`, finite values, and unit norms.
- Compare a sample against PyTorch MPS before accepting the cache.

### High-throughput batched pipeline

Implement cache generation as a bounded producer-consumer pipeline, following the
Moments Generator ingestion architecture but omitting Qdrant and unrelated scorers:

```text
ordered manifest
      ↓
ThreadPoolExecutor: read + decode + orient + resize/pad
      ↓ bounded queue of prepared uint8 images
single batching/GPU worker: normalize + MLX SigLIP inference
      ↓ bounded queue of embeddings and image IDs
single shard writer: validate + checkpoint + atomic rename
```

The GPU worker is the only owner of the MLX model. Do not invoke the same MLX model
concurrently from several Python threads: multiple inference callers compete for the
same Metal device, complicate ordering, and normally reduce rather than increase
throughput. Parallelism belongs before and after inference so the GPU worker always
has a prepared batch available and never waits for disk or image decoding.

#### Decode stage

- Use macOS ImageIO/CoreGraphics as the primary decoder, as in Moments Generator,
  because decoding and resizing execute in native code and release the Python GIL.
- Use Pillow only as a recorded fallback. Decoder choice is part of the cache
  provenance because pixel differences can change embeddings.
- Each worker performs file read, EXIF orientation, RGB conversion, aspect-preserving
  resize, and padding. It emits a contiguous `(224, 224, 3)` `uint8` array plus image
  ID and manifest position.
- Use threads rather than processes initially. Processes duplicate state, add image
  serialization overhead, and increase unified-memory pressure. Add a process-pool
  experiment only if profiling demonstrates Python-bound decoder work.
- Catch failures per image and emit an explicit failure result so ordering and row
  identity remain deterministic.

#### Batching and inference stage

- Accumulate full batches from the prepared-image queue; flush the final partial
  batch at end-of-stream. A short timeout may flush stalled partial batches during
  smoke tests, but the normal full-dataset path should favor full batches.
- Preallocate/reuse the host batch array where practical, and stack exactly once.
- Convert and normalize the whole batch vectorially. Do not normalize individual
  images in Python loops.
- Run the compiled MLX FP16 forward pass and force evaluation before reusing batch
  storage. Perform final L2 normalization vectorially.
- Preserve manifest positions in every result. Decode completion order must not
  determine cache row order.
- Maintain at least one complete batch ready while the current batch is executing.
  Two to four batches of queue capacity are the starting range; avoid hundreds of
  decoded batches because CPU arrays and Metal share the M5 Pro's 24 GB memory.

#### Writer and checkpoint stage

- The writer consumes completed batches while decoding and GPU inference continue.
- Buffer only the current deterministic shard, validate it, write to a temporary
  file, `fsync`, compute its checksum, then atomically rename it.
- Commit the shard manifest only after the data file is durable. On restart, verify
  completed shard hashes and skip them; regenerate only incomplete/corrupt shards.
- Write failures to a separate table keyed by image ID, without inserting placeholder
  vectors into the embedding matrix.

#### Durable SQLite work ledger

Use SQLite as the authoritative per-image work ledger. A JSON manifest remains useful
for immutable run/cache metadata, but it is not sufficient for concurrent status
updates or reliable crash recovery across hundreds of thousands of images.

Keep the database in the external cache directory, for example:

```text
$IAM_DATA_ROOT/ava/embeddings/siglip2-base-patch16-224/<cache-version>/work.sqlite3
```

Minimum schema:

```sql
CREATE TABLE items (
    image_id           TEXT PRIMARY KEY,
    manifest_position  INTEGER NOT NULL UNIQUE,
    source_path        TEXT NOT NULL,
    source_size        INTEGER NOT NULL,
    source_mtime_ns    INTEGER NOT NULL,
    source_fingerprint TEXT,
    state              TEXT NOT NULL,
    attempts           INTEGER NOT NULL DEFAULT 0,
    shard_id           INTEGER,
    shard_row          INTEGER,
    embedding_checksum TEXT,
    decoder            TEXT,
    error_type         TEXT,
    error_message      TEXT,
    first_started_at   TEXT,
    last_started_at    TEXT,
    finished_at        TEXT,
    updated_at         TEXT NOT NULL
);

CREATE TABLE runs (
    run_id              TEXT PRIMARY KEY,
    state               TEXT NOT NULL,
    model_revision      TEXT NOT NULL,
    preprocessing_hash  TEXT NOT NULL,
    cache_schema_version INTEGER NOT NULL,
    started_at          TEXT NOT NULL,
    heartbeat_at        TEXT NOT NULL,
    finished_at         TEXT,
    config_json         TEXT NOT NULL,
    error_message       TEXT
);
```

Allowed item states are:

```text
pending → decoding → decoded → embedding → staged → completed
                    ↘ retryable_error
                    ↘ permanent_error
```

`completed` means the embedding is present in an atomically committed shard whose
checksum has been verified. Finishing GPU inference alone must never mark an item as
completed. `staged` means the vector exists only in the writer's current uncommitted
shard buffer and must be recovered as pending after a crash.

SQLite updates must not run on decoder or GPU inference threads. All stages emit
small status events to a bounded ledger queue. One dedicated ledger-writer thread
owns the write connection and commits status updates in transactions, either every
`250–1,000` events or every `0.5–2` seconds. Producers must normally use non-blocking
or short-timeout queue insertion. The queue must have sufficient bounded capacity;
if it remains full, apply backpressure rather than silently dropping terminal states.

Configure SQLite for this single-writer/multiple-reader workload:

```sql
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=5000;
```

Do not share one SQLite connection across worker threads. Monitoring tools and Luna
open independent read-only connections. Keep transactions short. The ledger writer
must flush terminal events and run-heartbeat state during orderly shutdown.

At startup:

1. acquire an exclusive cache-run lock so two embedding jobs cannot write the same
   cache version;
2. validate the model revision, preprocessing hash, dataset fingerprint, and cache
   schema before resuming;
3. verify every shard recorded as committed, including row count and checksum;
4. retain `completed` only for rows backed by valid committed shards;
5. reset orphaned `decoding`, `decoded`, `embedding`, and `staged` rows to `pending`;
6. schedule `pending` and eligible `retryable_error` rows only;
7. quarantine corrupt shards and reset their rows rather than trusting ledger state.

Retry policy:

- retry transient I/O, decoder, and resource errors with bounded exponential backoff
  and jitter;
- default to three total attempts per image;
- classify missing files, unsupported/corrupt files, and deterministic validation
  failures as `permanent_error` after confirmation;
- allow a command-line override to retry permanent failures explicitly;
- truncate/sanitize stored exception messages and write full tracebacks to the run
  log so one malformed exception cannot bloat the database;
- a resumed default run processes pending and retryable failures, never valid
  completed rows.

Source changes must invalidate prior work. Compare stored path, size, modification
time, and—where configured—content fingerprint. If the source or preprocessing/model
identity changes, mark that item pending or require a new cache version. Never mix
embeddings from different model revisions or preprocessing contracts in one cache.

The ledger and shard commit form a recoverable two-phase protocol:

1. embeddings enter `staged` while the writer assembles a shard;
2. write and validate a temporary shard;
3. `fsync` the shard and containing directory, then atomically rename it;
4. in one SQLite transaction, register the committed shard and mark its rows
   `completed` with shard coordinates/checksum;
5. after a crash between steps 3 and 4, startup reconciliation discovers and adopts
   a valid orphan shard or safely regenerates it.

Add indexes for `state`, `shard_id`, and retry scheduling. Expose read-only commands
such as `status`, `failures`, and `verify` so progress can be inspected without
touching the running pipeline.

#### Autotuning on the target Mac

Do not assume the Moments defaults (`12` decode workers and batch size `32`) are
optimal for AVA. Before the full run, benchmark a representative 5,000–10,000-image
sample across:

- decode workers: `4, 8, 12`;
- MLX batch size: `16, 32, 48, 64`, stopping when memory pressure or swapping begins;
- prepared-queue depth: `2, 3, 4` batches.

Warm the compiled model before timing. Select the largest stable batch and smallest
worker count that keep the prepared queue non-empty without memory pressure. The
autotuner must impose safety limits rather than trying larger batches after an
out-of-memory event.

Record these throughput diagnostics:

- total images/second and wall-clock duration;
- decode throughput and p50/p95 decode latency;
- inference throughput and p50/p95 batch latency;
- writer throughput and time spent blocked on output;
- GPU starvation time (inference worker waiting for prepared images);
- decode backpressure time (workers waiting on a full queue);
- average/maximum queue occupancy;
- process resident memory, system memory pressure, and swap delta;
- failures by decoder and file type.

Acceptance criteria for the tuned run:

- no swap growth and no sustained macOS memory-pressure warning;
- less than 5% GPU starvation after warm-up;
- no missing or duplicate image IDs;
- restart produces byte-identical completed shard hashes;
- the selected configuration beats the single-threaded baseline materially.

#### Initial M5 Pro smoke profiling (2026-09-20)

A deterministic 3,000-image sample was profiled with the pinned
`google/siglip2-base-patch16-224` revision. The sample and raw measurements are
external to Git; `scripts/embeddings/profile_siglip2_ava.py` reproduces the
decode-only and warmed-inference measurements.

| Stage/configuration | Throughput | Key observation |
|:---|---:|:---|
| Decode/preprocess, 4 threads | 1,267 images/s | CPU is faster than inference |
| Decode/preprocess, 8 threads | 2,228 images/s | More than sufficient |
| Decode/preprocess, 12 threads | 2,301 images/s | Small gain, higher per-image latency |
| Warmed MLX inference, batch 32 | 404 images/s | Forward-only, model load excluded |
| Warmed MLX inference, batch 48 | 404 images/s | No material gain over 32 |
| Warmed MLX inference, batch 64 | 403 images/s | No gain; higher memory |
| Full overlap, batch 48 / 4 decode threads | 248 images/s | Same as 8 threads |
| Full overlap, batch 48 / 8 decode threads | 248 images/s | GPU wait ~0.02% |

The full run spent approximately 7.65 seconds of a 12.4-second wall-clock run in
warmed MLX inference, while the prepared queue was backpressured for roughly
7.3 seconds and the inference worker waited for decoded images for only 1.6 ms.
Atomic shard writing took roughly 1.6 seconds and was mostly waiting for inference.
Peak process RSS was 0.9--1.2 GiB with no swap growth. This establishes MLX/Metal
inference, not CPU decode or storage, as the primary bottleneck. Use four decode
workers and batch 48 as the current full-cache starting point; retain the bounded
queue so the CPU can stay ahead without accumulating unified-memory buffers.

For future Moments Generator ingestion, share one decode fan-out across SigLIP,
CLIP, and other image scorers. Overlap CPU-only metrics with one batched GPU model
owner, but schedule multiple Metal-heavy models serially or through a bounded
shared scheduler. Do not assume that creating concurrent MLX model owners improves
throughput: this run already had effectively zero GPU starvation, so additional
owners would add Metal command contention and unified-memory pressure.

A controlled probe compared one owner against two independently loaded owners on
disjoint halves. Two owners averaged 399.7 images/s versus 399.3 images/s for one
owner (1.001x, within run-to-run noise); RSS was about 1.35 GiB and system free
memory fell from 65% to 58%, with no swap delta. An isolated probe calling one
read-only owner from two Python threads reached about 421 images/s, but this is
not promoted: the MLX thread-safety contract is not guaranteed by the production
adapter and the gain is small. Both probes matched sequential outputs within max
absolute error `2.6e-4` and minimum cosine similarity `0.99999982`.

Concurrency improves utilization, not mathematical output. With the same decoder,
preprocessing, model revision, precision, and normalization, batching/threading must
produce embeddings within the cache parity tolerance of serial extraction.

#### Live progress and agent-readable status

Provide two views over the same progress state:

1. an interactive `tqdm` progress bar for a human-operated terminal; and
2. an atomically replaced `progress.json` file for Codex/Luna, scripts, and resumed
   sessions.

The terminal view should show completed/total images, percent, images/second, elapsed
time, ETA, active shard, failures, batch size, decode workers, and queue occupancy.
Use `tqdm.write()` for warnings so log messages do not corrupt the progress bar. When
stdout is not a TTY, disable the animated bar and emit a compact periodic log line.

Write `progress.json` every 5 seconds and at every shard boundary using a temporary
file followed by atomic rename. It should contain at least:

```json
{
  "schema_version": 1,
  "run_id": "...",
  "state": "running",
  "phase": "embedding",
  "total_images": 255000,
  "completed_images": 100000,
  "failed_images": 12,
  "images_per_second": 0.0,
  "elapsed_seconds": 0.0,
  "eta_seconds": 0.0,
  "current_shard": 24,
  "completed_shards": 24,
  "total_shards": 63,
  "batch_size": 32,
  "decode_workers": 8,
  "prepared_queue_size": 64,
  "prepared_queue_capacity": 96,
  "gpu_starvation_percent": 0.0,
  "resident_memory_gb": 0.0,
  "swap_delta_gb": 0.0,
  "started_at": "ISO-8601 timestamp",
  "updated_at": "ISO-8601 timestamp",
  "last_checkpoint": "...",
  "last_error": null
}
```

Final states are `completed`, `failed`, and `cancelled`; write one of them from a
`finally` block so a monitoring agent can distinguish a stopped process from stale
telemetry. Include the worker PID and a heartbeat timestamp. A monitor should treat
the run as stale if the PID is absent or the heartbeat has not advanced for a
configured interval, rather than relying on percentage alone.

The embedding command must be safe to run in a terminal, `tmux`, or another durable
shell session. Luna can poll `progress.json` and the process state to report progress,
but Luna should not be the lifetime owner of a many-hour job. Terminal/tmux ownership
prevents an agent turn, UI restart, or connection interruption from terminating the
embedding process. Progress polling must remain read-only and reasonably infrequent
(for example every 5–15 minutes unless troubleshooting).

`progress.json` is a disposable summary derived from the in-memory counters and
SQLite ledger; SQLite plus committed shard files are the sources of truth. On resume,
regenerate the JSON snapshot from the ledger rather than trusting an old progress
file.

#### Robustness and fault-injection requirements

The pre-cache implementation must handle normal termination, `SIGINT`, `SIGTERM`,
decoder exceptions, individual corrupt images, disk-full errors, writer failures,
and process termination without losing already committed work. Shutdown order is:
stop scheduling, let or cancel decode workers, finish/cancel the active GPU batch,
flush or discard the uncommitted shard safely, drain terminal ledger events, write
the final run state, and release the run lock.

Required tests include:

- resume after termination during decode, inference, temporary-shard write, atomic
  rename, and ledger commit;
- duplicate event delivery and idempotent ledger updates;
- corrupt/truncated image and unsupported format;
- missing or modified source after manifest creation;
- disk-full/write exception with no false `completed` rows;
- corrupted committed shard detected on `verify` and repaired on resume;
- two simultaneous writers rejected by the run lock;
- slow ledger writer and full queue apply safe backpressure;
- deterministic IDs/order despite out-of-order decode completion;
- completed rows skipped on resume while pending/retryable rows are scheduled;
- graceful cancellation produces a resumable cache and a final status snapshot.

Before the full AVA run, pass a 1,000-image smoke test and a fault-injection run that
terminates and resumes the process repeatedly. Report recovery counts and prove that
the final cache matches an uninterrupted reference run within the embedding parity
tolerance.

### Cache dtype

Generate float32 embeddings first. Measure float16-storage drift on at least 10,000 examples:

- embedding cosine similarity;
- downstream linear-probe score error;
- Spearman and top-k agreement.

Use float16 storage only if it passes the documented parity threshold. Convert batches to float32 during head training.

### Cache manifest

The manifest must include:

- cache schema version;
- source dataset fingerprint;
- split-manifest hash;
- model ID and immutable revision;
- MLX, Transformers, and decoder versions;
- preprocessing definition;
- embedding dtype and dimension;
- shard paths, row counts, and SHA-256 hashes;
- total successes and failures;
- extraction timings and device information.

## Phase 5: Model Depth and Architecture

The head is not learning a new visual representation from pixels. SigLIP already did that. It is learning a comparatively smooth mapping from an abstract 768-dimensional representation to human aesthetic ratings. Excessive depth is therefore more likely to memorize AVA-specific patterns than to “understand the embedding better.”

Evaluate these controlled candidates:

### A0: linear probe

```text
LayerNorm(768) → Linear(768, 10)
```

Purpose: minimum-capacity baseline and diagnostic for linearly accessible aesthetic information.

### A1: one hidden layer

```text
LayerNorm(768)
→ Linear(768, 256)
→ GELU
→ Dropout
→ Linear(256, 10)
```

Approximately 200K parameters.

### A2: two hidden layers — recommended starting model

```text
LayerNorm(768)
→ Linear(768, 512)
→ GELU
→ Dropout(0.20)
→ Linear(512, 128)
→ GELU
→ Dropout(0.10)
→ Linear(128, 10)
→ Softmax for inference
```

Approximately 462K parameters. This provides enough nonlinear capacity for content-dependent aesthetic patterns while staying below the project’s 1M-parameter target.

### A3: three hidden layers

```text
LayerNorm(768)
→ 512 → GELU → Dropout
→ 256 → GELU → Dropout
→ 128 → GELU → Dropout
→ 10
```

Use only as a depth ablation. Promote it only if grouped cross-validation consistently beats A2.

### Depth recommendation

Start with **two hidden layers**. Test zero, one, two, and three hidden layers. Do not go beyond three hidden layers during the initial phase. AVA has many images, but only 768 cached input features and one aesthetic target; deeper networks add optimization and overfitting risk without adding visual detail.

### Should the prediction head be a BERT-style transformer?

Not for the initial pooled-embedding model. The cached input is one 768-dimensional
vector per image. Self-attention over a sequence of length one is degenerate: its
attention weight is always one, so it provides no token-to-token interaction and is
effectively an expensive stack of projections, residual connections, and MLPs. It
cannot recover the spatial information already pooled away by SigLIP.

A transformer becomes meaningful only if the cache contract changes to retain the
196 patch tokens produced by the 14×14 patch grid (and possibly the pooled token).
Then a small attention-pooling head could learn which regions contribute to
aesthetics. That is a distinct experiment because it:

- expands each cached sample from `768` values to roughly `196 × 768` values;
- increases float16 cache storage from roughly 0.4 GB to roughly 75 GB for 255K
  images, before metadata and temporary files;
- substantially increases training memory and compute;
- risks relearning spatial relationships already represented by SigLIP;
- no longer matches the planned Moments integration, which stores pooled vectors.

Therefore use the two-hidden-layer MLP for the first milestone. Keep a later,
isolated **patch-token attention-pooling ablation** only if the MLP plateaus and error
analysis shows systematic composition/local-region failures. It should be judged by
grouped cross-validation and top-k ranking improvement large enough to justify its
storage and ingestion cost; do not call it BERT, because there are no language tokens
or masked-language-model objective in this task.

## Phase 6: Training Algorithm

### Forward pass

The network outputs ten logits. Convert them to probabilities only when computing the distribution loss or inference metrics.

```text
probabilities = softmax(logits)
predicted_mean = Σ probabilities[i] × (i + 1)
predicted_normalized_mean = (predicted_mean - 1) / 9
```

### Loss

Initial primary loss:

```text
loss = squared_EMD(predicted_distribution, target_distribution)
     + α × Huber(predicted_normalized_mean, target_normalized_mean)
```

Start with `α = 0.25`. Compare against:

- EMD alone;
- Huber mean regression with a scalar head;
- cross-entropy/KL-style distribution loss;
- EMD plus Huber with `α ∈ {0.1, 0.25, 0.5}`.

### Optimization defaults

| Parameter | Initial value |
|:---|:---|
| Optimizer | AdamW |
| Learning rate | `3e-4` |
| Weight decay | `1e-4` |
| Batch size | 512 |
| Maximum epochs | 50 |
| Warm-up | 2 epochs |
| Schedule | cosine decay |
| Gradient clipping | 1.0 |
| Early stopping | 7 validation checks without primary improvement |
| Training dtype | float32 initially |
| Checkpoint metric | validation Spearman, with top-k as tie-breaker |

Fit feature normalization statistics on the training partition only. Although embeddings are L2-normalized, `LayerNorm` remains inside the model; do not compute global normalization using validation or test data.

### Reproducibility

Each run records:

- configuration and config hash;
- code commit and dirty state;
- split and cache hashes;
- random seeds;
- device and dependency versions;
- epoch-level losses and metrics;
- best checkpoint and selection reason.

## Phase 7: Hyperparameter Search Without Test Overfitting

Use staged search rather than a large unconstrained sweep.

### Stage 1: architecture screen

On the fixed train/validation split, compare:

- depth: `0, 1, 2, 3` hidden layers;
- width for A1: `128, 256, 512`;
- A2 widths: `(256,64)`, `(512,128)`, `(512,256)`;
- dropout: `0.0, 0.1, 0.2, 0.3`.

Use one seed to eliminate clearly inferior or unstable models.

### Stage 2: optimization screen

For the best two architectures, compare:

- learning rate: `1e-4, 3e-4, 1e-3`;
- weight decay: `0, 1e-5, 1e-4, 1e-3`;
- batch size: `256, 512, 1024`;
- loss formulation and `α` values.

### Stage 3: robust confirmation

- Keep the best three configurations.
- Run grouped five-fold cross-validation inside development data.
- Run at least three seeds per finalist.
- Prefer the smaller model when confidence intervals overlap.

### Stage 4: locked test

- Freeze the chosen architecture, hyperparameters, epoch policy, and preprocessing.
- Retrain according to the predefined final-fit policy.
- Evaluate the locked test once.
- Publish all primary and secondary metrics, including negative results.

## Phase 8: Metrics and Model Selection

Primary:

- Spearman rank correlation;
- pairwise ordering accuracy;
- top-k lift at 1%, 5%, and 10%;
- top-k overlap with human-score rankings.

Secondary:

- Pearson correlation;
- MAE and RMSE of normalized mean;
- EMD of rating distribution;
- calibration by predicted-score decile;
- high/low classification accuracy at mean rating 5 only for comparison with prior work.

Required comparisons:

- constant mean predictor;
- linear probe;
- A1, A2, and A3;
- existing CLIP-based LAION predictor on identical test images;
- scalar-head versus distribution-head training;
- float32 versus float16 cache storage.

Required slices:

- score deciles;
- vote-count/confidence ranges;
- AVA challenge groups;
- content tags where available;
- monochrome versus color;
- portrait, landscape, indoor, low-light, and high-detail proxies where available.

## Phase 9: Export and Acceptance

Export only the selected checkpoint:

- `model.safetensors` with CPU tensors;
- architecture configuration;
- model and dataset provenance;
- preprocessing contract;
- training and test metrics;
- input/output schema;
- fixed inference examples and expected outputs.

Acceptance requires:

- reproducible cache and split hashes;
- no group or duplicate leakage;
- stable results across seeds;
- a meaningful improvement over the linear probe;
- competitive or complementary performance versus CLIP-LAION;
- CPU/MPS inference parity;
- under 1M parameters;
- negligible ingestion-time head latency;
- successful external or Moments-specific evaluation before production claims.

## Execution Order

1. Create package skeleton, configs, and tests.
2. Implement AVA parser and audit on annotations only.
3. Download/extract into the external dataset root.
4. Produce and freeze the group-disjoint split manifest.
5. Run a 1,000-image end-to-end smoke cache.
6. Train the linear probe and deliberately overfit a tiny diagnostic subset.
7. Generate the full resumable embedding cache.
8. Train A0, A1, A2, and A3 on the fixed development split.
9. Run staged hyperparameter selection.
10. Confirm finalists with grouped cross-validation and three seeds.
11. Freeze the winner and evaluate the locked test once.
12. Evaluate against CLIP-LAION and an external/Moments-specific set.
13. Export the versioned artifact for later fusion-model work.

## Immediate Next Step

Implement the repository skeleton, configuration loader, AVA annotation parser, audit report, grouped split generator, and tests before downloading or embedding the full dataset. This allows the dataset contract and leakage checks to be validated cheaply.
