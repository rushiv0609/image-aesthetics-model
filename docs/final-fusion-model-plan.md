# Final Multi-Signal Scorer: Data, Architecture, and Implementation Plan

## Decision Summary

The first production candidate will be a **branch-aware, content-conditioned residual
MLP trained with multi-dataset, multi-task supervision**. It consumes exactly the ten
signals in `docs/signals.md` and produces one query-independent aesthetic score in
`[0, 1]` during Moments Generator ingestion.

This decision deliberately separates three concerns:

1. **Architecture:** a small MLP with separate SigLIP, CLIP, and scalar branches plus
   content-conditioned scalar gates.
2. **Training objective:** a shared latent aesthetic rank trained across independent
   aesthetic datasets, with dataset-specific monotonic calibration heads. Technical
   IQA is an auxiliary task, not a second label silently mixed into aesthetics.
3. **Product policy:** the returned value is aesthetic quality. Query relevance,
   diversity, temporal continuity, and later personalization remain downstream.

The model is selected by evidence, not by architecture fashion. A linear/elastic
baseline, scalar-only MLP, flat concatenation MLP, and the proposed gated model must
all run on identical splits. The gated model is promoted only if it improves
cross-dataset ranking and Moments-style top-k selection by a meaningful margin.

## Why AVA Is Not the Final Fusion Training Set

AVA is appropriate for training the SigLIP2 A1 dependency head, but it is contaminated
for final fusion because several final-model inputs were themselves trained on AVA:

| Input signal | Known label-data exposure |
|---|---|
| SigLIP2 A1 aesthetic score | AVA |
| NIMA aesthetic score | AVA |
| MUSIQ aesthetic score | AVA |
| LAION CLIP aesthetic score | SAC, LAION logos, and AVA |
| MUSIQ technical score | KonIQ-10k |

If the fusion model sees those predictions and AVA labels for the same images, it can
learn which upstream model memorized AVA best instead of learning a rule that
generalizes to personal photographs. Therefore:

- AVA trains and evaluates the A1 dependency model.
- AVA may be used for diagnostic fusion ablations only, clearly marked contaminated.
- AVA must not decide the final fusion architecture or headline performance.
- KonIQ-10k must not supervise the final model's technical head while the released
  KonIQ-trained MUSIQ score is an input; it is retained as a contaminated diagnostic.
- Cross-dataset exact and near-duplicate detection is mandatory before any split.

## Dataset Portfolio

All datasets live outside Git under:

```text
/Users/rushivyas/datasets/image-aesthetics/
├── ava/
├── tad66k/
├── para/
├── aadb/
├── flickr-aes/
├── spaq/
├── live-challenge/
├── piq23/
└── registry/
```

Each dataset directory uses `downloads/`, `raw/`, `prepared/v1/`, `features/`, and
`licenses/`. The registry records source URL, access date, archive SHA-256, declared
terms, image provenance where available, label schema, and every overlap exclusion.

### Tier 1: independent aesthetic training data

| Dataset | Size and labels | Product value | Risks and decision |
|---|---|---|---|
| TAD66K | About 66K images, 1-10 MOS, 47 themes, balanced low/general/high collection | Strongest theme diversity; directly tests content-conditioned weighting | Official release does not state a clear artifact license. Acquire only after terms are accepted; train and split by theme/source groups. |
| PARA | 31,220 images, about 25 raters/image, 1-5 aesthetics plus quality/composition/color/content/light and scene attributes | Consumer-like content and rich auxiliary labels | Academic-purpose only. It includes images sourced from other IQA datasets; remove SPAQ/KonIQ and other overlaps before training. |
| AADB | 10,000 Flickr images, 5 raters/image, overall score plus 11 photographic attributes and rater identity | Useful consumer/professional balance and within-rater ranking pairs | Small and noisy; access and image-level licence provenance must be audited. Do not treat its mean as numerically identical to other MOS scales. |
| FLICKR-AES | 40,000 Creative Commons Flickr images, 5 ratings/image on 1-5 scale, rater identity | Broad consumer-photo domain; supports within-rater pairwise supervision | Official release says research use only. Five votes make absolute MOS noisy; emphasize within-rater ranking. |

### Tier 2: technical-quality auxiliary data

| Dataset | Size and labels | Role | Risks and decision |
|---|---|---|---|
| SPAQ | 11,125 smartphone photos, MOS plus brightness/color/contrast/noise/sharpness and scene labels | Primary technical auxiliary training set; closest public data to Moments capture conditions | Accessed through the official project drive; terms must be recorded. Split repeated scenes/devices as groups, not random images. |
| LIVE In the Wild | 1,162 authentic mobile images with roughly 350K opinions | External technical generalization test; optionally a small training fold after the first benchmark | Small, but its published permission is broad and explicit. Keep an untouched external test initially. |
| PIQ23 | 5,116 portrait images from more than 100 smartphones | Portrait/people technical-quality slice, highly relevant to Moments | Access requires a request; narrow portrait domain. Use as evaluation first, not as dominant training data. |
| KonIQ-10k | 10,073 authentic images and 1.2M ratings | Diagnostic only while the KonIQ-trained MUSIQ signal is an input | Direct target leakage for the current signal stack. Do not use for headline training/evaluation. |

### Evaluation or exclusion only

- **EVA:** useful annotations, but all 4,070 images are selected from AVA. It is not
  independent data and must not be advertised as external validation.
- **CUHK-PQ:** binary high/low labels are too coarse for the primary scorer. It may be
  a stress-test after provenance review.
- **PCCD:** critique captions are useful for explanations, not direct scalar
  supervision.
- **BAID/LAPIS:** artistic works are a domain mismatch for personal photos and videos.
- **AI-generated-image aesthetic datasets:** exclude from the initial product model;
  their artifacts and preference distributions differ from camera media.
- **Synthetic distortion datasets:** exclude initially. They overweight artificial
  corruption types and do not represent normal phone/camera capture failures.

### Primary acquisition sources

- TAD66K: <https://github.com/woshidandan/TANet-image-aesthetics-and-quality-assessment>
- PARA: <https://yuzheyang.github.io/blog/Dataset/PARA.html>
- AADB paper/project link: <https://arxiv.org/abs/1606.01621>
- FLICKR-AES: <https://github.com/alanspike/personalizedImageAesthetics>
- SPAQ: <https://github.com/h4nwei/SPAQ>
- LIVE In the Wild: <https://live.ece.utexas.edu/research/ChallengeDB/>
- KonIQ-10k: <https://database.mmsp-kn.de/koniq-10k-database.html>
- PIQ23: <https://github.com/DXOMARK-Research/PIQ2023>
- EVA: <https://github.com/kang-gnak/eva-dataset>

Use these authoritative project releases rather than third-party mirrors. A mirror is
acceptable only when the official checksum can verify it and the original terms still
apply.

## Acquisition Gate

Do not bulk-download first and ask licensing questions later. Before acquiring each
Tier 1/Tier 2 dataset:

1. Save the official terms or README under `licenses/` with source and access date.
2. Classify use as `broad`, `research-only`, `academic-only`, or `unclear`.
3. Record whether the intended trained weights may be redistributed or used in a
   distributed Moments Generator application.
4. Require an explicit project decision for `research-only`, `academic-only`, and
   `unclear` sources.
5. Never commit images, annotations, access tokens, or downloaded archives.

Initial acquisition order after this gate:

1. TAD66K and SPAQ: highest diversity/product fit.
2. PARA: rich annotations after removing known source overlaps.
3. FLICKR-AES and AADB: valuable pairwise/rater supervision, subject to access.
4. LIVE Challenge and PIQ23: holdout evaluation.

## Canonical Data Contract

Every prepared example has a common envelope, but labels retain their native meaning:

```text
sample_id
dataset_id
source_image_id
source_group_ids[]        # photographer, challenge, scene, user, device where known
source_hash_sha256
perceptual_hash
native_score
native_score_min/max
normalized_mos            # metadata convenience, never assumed cross-dataset equivalent
vote_histogram?           # only where actually available
rater_id_scores?          # only where actually available
theme_or_scene?
attribute_targets?
split
license_class
```

The feature cache adds the exact ten production signals, their checkpoint hashes,
preprocessing versions, and per-feature status. Cache generation reuses the robust
bounded producer/inference/writer architecture already built for AVA.

## Leakage and Split Protocol

### Cross-dataset deduplication

Run before split assignment:

1. exact SHA-256 grouping;
2. EXIF/source-ID matching where available;
3. perceptual-hash candidate generation;
4. SigLIP/CLIP nearest-neighbour review for resized, cropped, filtered, or recompressed
   variants;
5. union all related items into one connected group;
6. exclude any image known to belong to an upstream scorer's training data when that
   scorer is used as an input signal.

### Dataset-native grouping

- TAD66K: group by known source/near-duplicate family; stratify by theme and score.
- PARA: group by source image and known source dataset; preserve user/rater structure.
- AADB/FLICKR-AES: group by Flickr/source identity and use rater-disjoint checks for
  personalized annotations.
- SPAQ: group repeated scenes and, for a harder test, hold out phone models.
- Moments benchmark: group burst, event, and video; frames from one event never cross
  partitions.

### Evaluation layers

1. **Within-dataset grouped CV** checks optimization and label fit.
2. **Leave-one-dataset-out evaluation** measures generalization instead of memorizing
   a dataset's scoring culture.
3. **External technical tests** use LIVE/PIQ23.
4. **Moments benchmark** is the release gate: event-grouped personal photos and video
   frames with pairwise preferences and reel-level top-k judgments.

## Label Harmonization Without Destroying Meaning

Raw 1-5, 1-10, binary, and MOS labels must not be concatenated after a linear rescale.
Different datasets have different raters, anchors, content distributions, and score
compression.

Use one shared latent aesthetic value `q_aes` and a monotonic calibration head per
dataset:

```text
production model: signals -> q_aes in [0, 1]
training only:     q_aes -> monotonic calibrator(dataset_id) -> native label scale
```

The calibrator may start as positive-slope affine calibration in logit space. A small
monotonic spline is allowed only if residual plots show that affine calibration is
insufficient. Because calibration is monotonic, dataset-specific numeric habits can
change scale without reversing the shared ranking.

Losses are computed only within compatible label spaces:

- pairwise logistic/ranking loss within the same dataset, theme, or rater;
- Huber loss after the dataset-specific calibrator;
- EMD only where a real vote distribution exists;
- auxiliary attribute losses only where that attribute was annotated;
- dataset-balanced sampling so AVA-sized or heavily annotated sources cannot dominate.

Technical IQA uses a separate latent `q_tech` and separate dataset calibrators. It
shares the representation with aesthetics, but its labels are never called aesthetic
labels. The production return value remains `q_aes`; technical supervision teaches
the shared trunk to recognize defects, while the aesthetic datasets decide how those
defects affect overall appeal.

## Architecture Candidates

### Required baselines

| ID | Model | Why it is required | Main limitation |
|---|---|---|---|
| F0 | Current hand-tuned weighted sum | Product baseline | Static and uncalibrated |
| F1 | Best single signal and mean/median ensembles | Detects whether fusion adds value at all | No learned interactions |
| F2 | Regularized linear/GAM on eight scalars | Strong interpretable baseline | Weak use of semantic embeddings |
| F3 | Scalar-only MLP | Cheap test of the deployed score stack | Cannot recover information lost by upstream heads |
| F4 | Flat concatenation MLP | Capacity-matched neural baseline | Embedding dimensions dominate; poor structural prior |

### F5, recommended candidate: branch-aware content-conditioned residual MLP

```text
SigLIP embedding (768) -> LayerNorm -> 128 -> GELU -> 64 -----+
                                                               |
CLIP embedding   (768) -> LayerNorm -> 128 -> GELU -> 64 -----+--> fusion trunk
                                                               |      168 -> 256 -> 128
8 standardized scalars -> 32 -> GELU -------------------------+             |
        ^                                                                    +--> q_aes
        | content-conditioned gates from both embedding branches             +--> q_tech (training auxiliary)
        + original scalar values retained through a residual path            +--> vote distribution/attributes when available
```

The SigLIP and CLIP branches prevent their 1,536 coordinates from numerically
drowning eight scalar inputs. The gate learns statements such as "sharpness is less
decisive for this content," while the residual path prevents a saturated gate from
deleting information. An initialized skip from the frozen A1 score gives a strong
starting point but remains learnable and removable in ablations.

Expected size is roughly 300K-500K parameters, comfortably below the 1M limit. It has
negligible head latency on CPU/MPS. Gates are useful diagnostics, not causal
explanations.

### Approaches considered critically

| Approach | What it can do well | Why it is not the first production candidate |
|---|---|---|
| Flat MLP | Universal nonlinear approximation; fast and robust | Ignores signal structure and can over-rely on correlated AVA-trained inputs; keep as baseline |
| Gated mixture of experts | Specialization and content-dependent routing | Experts/gates often collapse without explicit supervision; preassigning aesthetic/technical experts bakes in assumptions; harder to calibrate |
| Signal-token Transformer | Flexible interactions when there are many meaningful tokens | Ten heterogeneous tokens do not require attention; scalar-to-token projection manufactures dimensions; attention is not an explanation; likely no gain over an MLP at this data scale |
| Patch-token Transformer | Can learn spatial composition from unpooled regions | Changes the production/cache contract, adds roughly two orders of magnitude of storage, and duplicates backbone work |
| Factorized bilinear fusion | Explicit content-by-signal interactions | More fragile and parameter-hungry than simple conditional gates; test only if gated MLP misses clear interactions |
| Gradient-boosted trees | Excellent on eight scalar features, interpretable, little tuning | Poor fit for two dense 768-D embeddings and less convenient for differentiable multi-task heads; valuable scalar-only baseline |
| Generalized additive/monotonic model | Transparent contributions and stable behavior | Cannot capture rich content-conditioned exceptions without many hand-built interactions |
| Learning-to-rank model | Directly optimizes ordering/top-k behavior | This is an objective, not an architecture; combine it with F2-F5 rather than treating it as a competing network |
| List/set-level scorer | Can optimize an entire reel or library jointly | Violates the ingestion-time standalone score contract; relevance/diversity belong to the Director layer |
| End-to-end backbone fine-tuning | Maximum representational flexibility | Too costly, risks changing retrieval embeddings, makes caches stale, and conflicts with the local/reproducible scope |
| Ensemble of several heads | Better stability and uncertainty estimates | Multiplies deployment/versioning complexity; use during research, then distill if the gain is real |
| Bayesian/uncertainty head | Can flag uncertain false positives | Human disagreement and dataset shift make uncertainty hard to calibrate; add after a strong deterministic baseline |
| Online bandit/RL | Learns an individual's observed preferences | No exploration signal exists in offline generic training; defer until Moments collects user feedback |

## Training Recipe

### Stage 0: freeze dependencies

- Finalize and version the SigLIP2 A1 head.
- Fix NIMA, LAION-CLIP, MUSIQ-aesthetic, and MUSIQ-technical checkpoints.
- Make all scorers fail closed on missing or incompatible weights.
- Produce an exposure matrix and preprocessing/checkpoint hashes.

### Stage 1: acquire, audit, and split

- Complete the licensing gate.
- Download into external dataset roots with resumable transfers and SHA-256 manifests.
- Normalize annotations without discarding raw votes/rater IDs.
- Deduplicate across every dataset and all AVA images.
- Freeze grouped development/validation/locked-test manifests per dataset.

### Stage 2: pre-cache all ten signals

- Reuse the asynchronous SQLite ledger, atomic shards, bounded decode queues, live JSON
  status, and retry semantics from the AVA cache pipeline.
- Share each decoded image among pixel-based scorers.
- Schedule Metal-heavy models through one bounded owner at a time; profile before any
  concurrent-model experiment.
- Store vectors as float16 only after score/rank parity; train heads in float32.
- Reject a cache if any scorer used random/default weights or a preprocessing identity
  differs from Moments Generator.

### Stage 3: baseline screen

- Train F1-F4 with three seeds on identical folds.
- Report per-dataset and macro-averaged Spearman, pairwise accuracy, top-k lift, and
  calibration; never let the largest dataset alone choose the winner.
- Run leave-one-dataset-out tests.

### Stage 4: proposed model and limited ablations

- Train F5 with ranking + calibrated Huber losses.
- Ablate each embedding, each scalar family, conditional gates, A1 skip, technical
  auxiliary loss, and dataset calibrators.
- Compare a small MoE and signal-token Transformer only after F5 is stable. Match
  parameter count and training budget so the comparison is fair.

### Stage 5: Moments release gate

Build a small but high-quality product benchmark rather than substituting another web
dataset. Sample whole events/bursts and representative video frames, then collect:

- within-event pairwise aesthetic preferences;
- keep/reject labels for unusable frames;
- top-k selections for realistic prompts while keeping prompt cosine outside this
  model;
- failure tags: blur, low light, duplicate/burst, awkward expression, transition
  frame, screenshot/meme, monochrome, and deliberate motion.

Release only if the proposed model improves human-selected top-k quality over F0, the
best single signal, and F4 without increasing severe false positives.

## Selection Rule

Use macro-averaged cross-dataset Spearman as the primary offline metric, with
within-event pairwise accuracy and top-k precision on the Moments benchmark as the
product gate. Calibration and MAE are secondary.

Promote F5 over F4 only when:

- the gain is consistent across seeds and at least three independent aesthetic
  datasets;
- leave-one-dataset-out performance does not regress materially;
- Moments top-k precision improves, especially at 1% and 5%;
- severe technical false positives do not increase;
- every component and the final model fit the M5 Pro memory/latency contract.

If F5 and F4 are statistically tied, ship F4. If F3 matches models using embeddings,
ship F3 and avoid carrying redundant vector inputs into the final head.

## Immediate Next Steps

1. A1 final fit, one-time locked-test evaluation, and versioned export are complete.
2. Decide whether research/academic-only datasets are acceptable for this phase and
   whether trained weights are intended for redistribution/commercial use.
3. Implement the dataset registry and download/audit adapters for approved sources.
4. Acquire TAD66K and SPAQ first; audit labels, licenses, and duplicates before
   downloading the remaining portfolio.
5. Create the cross-dataset exposure/dedup report.
6. Generalize the AVA feature-cache pipeline to dataset adapters without changing its
   concurrency or durability guarantees.
7. Pre-cache all ten signals, then train F1-F4 before implementing F5.
