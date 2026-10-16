# Final Fusion Model: Detailed Implementation Plan

## Goal

Build, evaluate, and export a sub-1M-parameter scorer for Moments Generator that:

- consumes exactly the ten signals in `docs/signals.md`;
- returns one query-independent aesthetic score in `[0, 1]` at ingestion time;
- trains locally on the M5 Pro from pre-cached features;
- learns from multiple independent datasets without mixing incompatible label scales;
- is selected by cross-dataset ranking and Moments-specific top-k quality;
- never depends on downloaded datasets or model artifacts being committed to Git.

This plan implements the architecture and data decisions in
`docs/final-fusion-model-plan.md`.

## Execution Principles

1. Complete cheap, testable infrastructure before downloading large datasets.
2. Treat dataset access terms as data, not prose: every source needs a versioned
   registry record and an external snapshot of its terms.
3. Never train from raw directory layout. Every dataset must pass through the same
   prepared-manifest contract.
4. Build F5 only after F1-F4 baselines and feature ablations exist; keep its
   candidate comparisons controlled and separate from architecture design.
5. Reuse the AVA cache pipeline's durability primitives; do not copy its code into a
   parallel implementation.
6. Keep acquisition, preparation, feature extraction, model training, and release
   artifacts in separate directories and commands.

## External Storage Contract

```text
/Users/rushivyas/datasets/image-aesthetics/
├── registry/
│   ├── datasets.json
│   ├── licenses/<dataset>/<snapshot>
│   └── audits/
├── <dataset>/
│   ├── downloads/
│   ├── raw/
│   ├── prepared/v1/
│   ├── features/<feature-contract>/
│   ├── experiments/
│   └── releases/
└── fusion/
    ├── prepared/v1/
    ├── experiments/
    └── releases/
```

Repository code and small aggregate reports belong in Git. Images, raw annotations,
archives, SQLite ledgers, feature shards, checkpoints, and releases stay external.

## Phase 0: Freeze Dependency Contracts

### Deliverables

- Record A1 `a1-v1` model hash, architecture, score formula, and SigLIP revision.
- Inventory the exact production checkpoints for CLIP-LAION, NIMA, MUSIQ-AVA, and
  MUSIQ-KonIQ.
- Produce `reports/signal-exposure-matrix.json` containing:
  - signal name;
  - model/checkpoint identifier and SHA-256;
  - known training datasets;
  - preprocessing identity;
  - output range/direction;
  - production readiness and failure behavior.
- Fix Moments Generator scorers before using them for a cache:
  - NIMA must load a real trained checkpoint;
  - LAION must fail closed and load weights strictly;
  - MUSIQ must receive its documented aspect-ratio-preserving input.

### Exit gate

All ten features can be reproduced for a fixed 20-image fixture, and no scorer can
silently emit values from random or incompatible weights.

## Phase 1: Dataset Registry and Licensing Gate

Project use mode is fixed to **non-commercial research**. Commercial use and dataset
redistribution are prohibited. Research-only and academic-only datasets may be used
locally; unclear terms still require source review rather than being treated as
permission. The machine-readable decision is in
`configs/data/project-use-policy.json`.

### Repository implementation

Create:

```text
configs/data/datasets/
src/image_aesthetics/data/registry.py
scripts/data/dataset_registry.py
tests/data/test_dataset_registry.py
```

The registry schema must include:

```text
dataset_id
display_name
task                 # aesthetic, technical, evaluation-only
official_url
download_method      # direct, google-drive, form, manual
declared_license     # exact source text/reference
license_class        # broad, research-only, academic-only, unclear
commercial_status    # allowed, forbidden, unclear
redistribution_status
source_version
expected_files
known_upstream_overlap
enabled
```

The CLI supports `list`, `show`, `validate`, and `acquisition-plan`. It performs no
downloads initially. Validation fails when an enabled dataset lacks a license decision,
official source, expected-file contract, or overlap declaration.

### Initial registry entries

- `tad66k`: aesthetic; unclear terms; Google Drive.
- `para`: aesthetic; academic-only; Google Drive/Baidu.
- `aadb`: aesthetic; unclear image-level terms/access.
- `flickr_aes`: aesthetic; research-only; Google Drive.
- `spaq`: technical; unclear commercial terms; Google Drive/Baidu.
- `live_challenge`: technical evaluation; broad permission; form/Box.
- `piq23`: technical evaluation; request required.
- `koniq10k`: contaminated diagnostic; research-community release.
- `eva`: AVA-overlap diagnostic only; CC0 repository metadata.

### Exit gate

The registry validates and prints deterministic commercial-safe and non-commercial
research acquisition plans. The research plan admits broad, research-only, and
academic-only sources, but excludes disabled and unclear-license sources.

## Phase 2: Acquisition Adapters

Implement only after the licensing decision.

### Commands

```text
scripts/data/acquire_dataset.py --dataset <id> --data-root <root>
scripts/data/verify_download.py --dataset <id> --data-root <root>
```

### Required behavior

- resumable transfers where the source supports them;
- no credentials or cookies written into Git;
- temporary download followed by size/checksum validation and atomic rename;
- immutable `download-manifest.json` with URL, timestamp, size, SHA-256, tool version,
  and user-supplied files clearly distinguished;
- `--verify-only` and dry-run modes;
- manual/form-gated datasets produce exact instructions and then stop successfully;
- never scrape or bypass an access form;
- reject extraction paths escaping the dataset root.

### Acquisition order

1. TAD66K and SPAQ after approval.
2. PARA after approval and source-overlap audit design.
3. FLICKR-AES and AADB after access/terms review.
4. LIVE Challenge and PIQ23 as untouched evaluation sources.

## Phase 3: Dataset Adapters and Canonical Manifests

### Interface

Each adapter converts native annotations into the canonical sample contract without
discarding raw labels:

```python
class DatasetAdapter(Protocol):
    def inventory(self) -> DatasetAudit: ...
    def iter_samples(self) -> Iterable[CanonicalSample]: ...
    def grouping_keys(self, sample: CanonicalSample) -> tuple[str, ...]: ...
```

### Canonical outputs

```text
prepared/v1/samples.parquet
prepared/v1/rejected.parquet
prepared/v1/audit.json
prepared/v1/source-manifest.json
```

Canonical samples retain dataset ID, native score/range, individual votes when
available, attributes, themes/scenes, rater identities, source groups, SHA-256,
perceptual hash, dimensions, and decoding status.

### Exit gate

Counts reproduce the official dataset documentation; every source item is either
accepted or rejected with a reason; prepared manifests are deterministic.

## Phase 4: Cross-Dataset Deduplication and Splits

### Deduplication stages

1. exact SHA-256 union;
2. source URL/ID and EXIF identity union;
3. perceptual-hash candidate generation;
4. SigLIP/CLIP nearest-neighbour candidates;
5. deterministic visual-review queue for uncertain candidates;
6. connected-component group assignment across all datasets and AVA.

The process must explicitly remove known PARA images originating from SPAQ/KonIQ and
block any sample exposed to an upstream label predictor where determinable.

### Splits

- Freeze per-dataset development and locked-test manifests.
- Preserve dataset-native groups such as TAD theme/source, SPAQ repeated scenes and
  devices, Flickr source identities, and rater structure.
- Create five development CV folds per training dataset.
- Reserve LIVE, PIQ23, and the Moments benchmark from architecture selection.

### Exit gate

No connected duplicate/source group crosses a split. A machine-readable report lists
all removed overlaps and uncertainty queues.

## Phase 5: Generalized Ten-Signal Feature Cache

Refactor the existing AVA implementation around a dataset-neutral input manifest.

### Feature contract

```text
siglip_embedding[768]
clip_embedding[768]
laion_clip
a1_siglip
nima
musiq_aesthetic
musiq_technical
sharpness
contrast
colorfulness
```

Each cache version records every model hash, decoder/preprocessing identity, scalar
normalization formula, feature dtype, source manifest hash, and failure.

### Performance design

- one bounded parallel decode stage;
- share decoded/native-resolution representations where contracts allow;
- separate 224 padded inputs from native/aspect-preserving MUSIQ inputs;
- one bounded Metal model owner at a time unless profiling proves safe concurrency;
- async SQLite ledger and atomic deterministic shards;
- live progress JSON and read-only `status`, `failures`, and `verify` commands;
- resume only pending/retryable rows and never recompute verified shards.

### Smoke protocol

Run 1,000 varied images per dataset before a full cache. Measure per-scorer throughput,
GPU starvation, decode rate, RSS, swap, failure rate, and end-to-end images/second.

## Phase 6: Training Dataset and Label Semantics

### Current implementation status

The frozen `fusion-prep-v6` snapshot contains the four independent aesthetic
targets: AADB, PARA, Flickr-AES, and TAD66K. It has 147,247 usable image-level
records after quarantining unresolved near-duplicate candidates. Dataset-native
locked-test rows are preserved as a hard boundary before assigning five
development folds; the audit records the resulting counts. AVA and KonIQ
remain diagnostic-only; SPAQ remains a technical auxiliary and is not allowed to
define the shared aesthetic target.

`FusionTrainingDataset` joins this snapshot to the immutable 1,544-float
`joined-ten-v1` indexes, prevents accidental locked-test access, and exposes a
specific development fold as validation. `DatasetBalancedBatchSampler` emits
dataset-homogeneous batches in equal dataset turns so TAD66K/Flickr-AES cannot
silently dominate AADB/PARA. Readiness validation must successfully collate
finite 32×1,544 batches from every fold before training begins.

### Multi-dataset loader

- batches contain one label-compatible dataset/task at a time;
- dataset-balanced sampler prevents large sources dominating;
- pairwise samples are constructed only within compatible dataset/theme/rater groups;
- dataset ID is available to training-only calibration heads but is not a production
  model input;
- feature normalization is fitted on development-train only.

### Targets

- shared latent `q_aes` for aesthetic ranking;
- dataset-specific monotonic calibrator for native MOS regression;
- shared latent `q_tech` for technical IQA auxiliary supervision;
- vote-distribution EMD only where actual distributions exist;
- attribute heads only for explicitly annotated attributes.

No binary dataset is allowed to define the shared numeric scale.

## Phase 7: Baselines F0-F4

### Current implementation status

F0-F4 are implemented under `src/image_aesthetics/models/fusion/` against the
actual non-contiguous `joined-ten-v1` storage contract. F0 reproduces the
query-independent portion of the current Moments fallback weighting; F1 can
evaluate each named scalar or mean/median ensembles; F2-F4 are trainable and
return a production score in `[0,1]`.

`scripts/training/train_fusion_baseline.py` provides fold/seed/model selection,
train-only streaming normalization, dataset-specific monotonic calibration,
dataset-homogeneous balanced batches, compatible ranking loss, atomic status,
best/last checkpoints, exact identity-checked resume, early stopping, and
per-dataset/macro ranking and top-k metrics. `fusion-prep-v6` canonicalizes all
Flickr-AES labels to one `[0,1]` scale. The separate immutable
`spaq-technical-v2` split and `SPAQTechnicalDataset` are ready for an F5
technical head; 29 exact overlaps with the aesthetic corpus are excluded so an
auxiliary batch cannot expose aesthetic validation/test imagery. F0-F4
intentionally remain single-head baselines.

Implement and compare:

- F0: current Moments weighted sum;
- F1: every single score plus mean/median ensembles;
- F2: regularized linear/GAM using eight scalars;
- F3: scalar-only MLP;
- F4: flat 1,544-feature MLP with matched parameter budget.

Every candidate uses identical grouped splits, three seeds, and the same training
budget. Report dataset-macro and per-dataset Spearman, pairwise accuracy, top-k lift,
MAE/calibration, parameter count, and CPU/MPS latency.

## Phase 8: F5 Branch-Aware Gated Residual MLP

Implementation status (2026-10-16): the historical F3 remains unchanged and
`F3C` provides the corrected no-per-image-LayerNorm control. `F5-R`, `F5-G`,
and `F5-X` now implement the approved shapes in
`docs/f5-architecture-decision-memo.md`. Each uses the same-fold/same-seed F2
scalar logit plus a zero-initialized bounded residual; CLI initialization
validates the F2 run identity, split manifest, normalization artifact, fold,
seed, and subset limit. Stable architecture/provenance fields participate in
new-run/resume identity. Technical scoring remains an opt-in model API only;
the aesthetic screen has no technical task or SPAQ loader. No F5 screen,
confirmation matrix, LODO experiment, or locked-test evaluation has started.

Use only the `smoke_f5.py` synthetic check while validating code paths:

```bash
python scripts/training/smoke_f5.py --device cpu --samples 16 --epochs 1
# On a host where PyTorch MPS is available:
python scripts/training/smoke_f5.py --device mps --samples 16 --epochs 1
```

These commands use deterministic in-memory rows and temporary output folders;
they do not access research data or count as candidate training runs.

After preflight succeeds on an MPS-capable host, start the fixed screen with:

```bash
caffeinate -dimsu .venv/bin/python scripts/training/run_f5_screen.py \
  --execute --device mps --batch-size 1024 --num-workers 4 \
  --status-interval-batches 25 --epochs 50 --patience 7 --min-epochs 8
```

The screen's candidate/fold/seed matrix is fixed in code; its CLI cannot expand
to other folds, seeds, technical supervision, or a test partition.

Initial frozen search space:

```text
SigLIP branch: LN(768) -> 128 -> GELU -> 64
CLIP branch:   LN(768) -> 128 -> GELU -> 64
Scalar branch: standardized 8 -> 32 -> GELU
Gate:          concat embedding branches -> 32 -> 8 -> tanh
Fusion:        64 + 64 + 32 + residual scalars -> 256 -> 128
Heads:         q_aes, q_tech, optional distribution/attributes
```

Use ranking loss plus calibrated Huber for aesthetics. Add technical/attribute losses
with coefficients selected only inside development CV. Initialize, but do not force,
an A1 residual skip.

Required ablations:

- no CLIP embedding;
- no SigLIP embedding;
- no scalar family, then leave-one-scalar-out;
- no conditional gate;
- no A1 skip;
- no technical auxiliary task;
- shared calibration versus dataset-specific monotonic calibration.

Only after F5 is stable, compare one parameter-matched MoE and one signal-token
Transformer. Stop if F4/F5 already tie within variation.

## Phase 9: Moments-Specific Benchmark

Build an event-grouped benchmark from representative personal media. Store only
manifests/annotations in the repository when licensing/privacy permits; media remains
external.

Required labels:

- within-event pairwise preference;
- keep/reject for severe failures;
- top-k selections for realistic curation tasks;
- failure tags for blur, low light, transition frames, awkward expression, duplicate
  burst, screenshot/meme, monochrome, and intentional motion.

The final release must improve top-1%/5% selection and not increase severe false
positives versus F0, F1, and F4.

## Phase 10: Export and Moments Integration

Release directory:

```text
fusion/releases/<version>/
├── model.safetensors
├── model-config.json
├── feature-contract.json
├── calibration.json
├── metrics.json
├── provenance.json
└── fixed-inference-fixtures.json
```

Integration requirements:

- strict checkpoint and feature-contract hashes;
- fail closed on missing/incompatible signals;
- deterministic CPU/MPS parity tests;
- one score stored during ingestion with model version in Qdrant payload;
- complete workspace re-indexing;
- downstream A/B comparison of learned-score-only versus relevance/quality weighting.

## Agent and Usage Strategy

Use inexpensive agents for bounded mechanical tasks and reserve stronger reasoning for
decisions that can invalidate large runs:

- **Luna:** registry schema/CLI, config entries, adapter boilerplate, unit tests,
  manifest verification, status tooling, documentation synchronization.
- **Terra:** dataset adapter framework, safe acquisition framework, generalized cache
  refactor, baseline implementation, experiment supervisors.
- **SOL or stronger:** cross-dataset leakage audit, loss/calibration review, architecture
  selection, failure analysis, release decision, and security/correctness review.

Do not use an LLM as a minute-by-minute monitor. Long processes write atomic local
status files and notify an agent only on failure, completion, or required decisions.

## Immediate Work Queue

1. Luna: implement Phase 1 registry/config/CLI/tests only; no downloads.
2. Review and commit Phase 1.
3. Use the recorded non-commercial research policy; manually review unclear sources.
4. Terra: implement Phase 2 acquisition framework with dry-run/manual-source support.
5. Luna: add approved dataset registry records and verification fixtures.
6. Terra: implement the first two native adapters after archives exist.
7. SOL review before any full multi-model feature-cache run.
