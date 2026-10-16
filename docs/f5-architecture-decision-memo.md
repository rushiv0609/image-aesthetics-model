# F5 architecture council: decision memo

Status: approved for implementation. F3C and F5-R/G/X model implementations,
provenance-checked F2 initialization, and synthetic smoke checks are in place.
The proposed 12-run screen and full F5 matrix have **not** been started.

## Product and evidence boundary

Moments Generator needs one query-independent aesthetic score in `[0, 1]` per
photo or sampled video frame at ingestion time. The scorer consumes exactly the
existing ten signals: SigLIP 2 and CLIP embeddings (768 dimensions each), plus
A1, LAION-CLIP, NIMA, MUSIQ aesthetic, MUSIQ technical, sharpness, contrast,
and colorfulness. The vision models run upstream; F5 trains on cached features
and must train and infer on an Apple M5 Pro with 24 GB unified memory. Relevance,
diversity, event importance, and personalization belong outside this scorer.

The `fusion-prep-v6` aesthetic corpus contains 147,247 usable records from
AADB, Flickr-AES, PARA, and TAD66K. It has grouped five-fold development
splits and a separate locked test. SPAQ is a distinct technical auxiliary
dataset (`spaq-technical-v2`), with 29 exact aesthetic overlaps excluded. AVA
and KonIQ labels cannot select F5: A1, NIMA, MUSIQ-aesthetic, and LAION have
AVA exposure; MUSIQ-technical has KonIQ exposure. Dataset ID can select a
training-only monotonic calibration head, but cannot be an inference input.

The completed 100-run F0-F4 matrix provides the following 15-run means for
each trained model (five folds, three seeds). These are development-validation
results, not a Moments or locked-test result.

| Model | Macro Spearman ↑ | Pairwise accuracy ↑ | Top 1% lift ↑ | Top 5% lift ↑ | Top 10% lift ↑ | MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F2, linear scalars | 0.5877 | 0.7161 | **0.2420** | **0.1949** | **0.1634** | 0.1243 |
| F3, scalar MLP | 0.4178 | 0.6483 | 0.1624 | 0.1388 | 0.1209 | 0.1299 |
| F4, flat 1,544-feature MLP | **0.5884** | 0.7156 | 0.2284 | 0.1831 | 0.1577 | **0.1107** |

F2 and F4 differ by only +0.00066 in macro Spearman in F4's favor. F2 has
better top-k lift and Flickr-AES Spearman (0.6192 versus F4's 0.5670); F4 has
lower MAE and better Spearman on AADB, PARA, and TAD66K. The strongest single
scalar is A1, but its AVA-trained score is correlated with the SigLIP branch.
F5 must prove that its added interactions improve actual selection quality.

One implementation detail changes the interpretation of F3. Its first layer
is `LayerNorm(8)` across the eight scores **for each image**. That operation
removes the shared level and scale of those scores, which likely carry quality
information. F3's weak result therefore cannot establish that scalar
nonlinearity itself is harmful. A corrected F3 control using only the
fold-fitted feature standardization and no per-image scalar LayerNorm is
required before drawing that conclusion.

## Council discussion, summarized

**Principal applied scientist:** F2 is the anchor. Its simple scalar mapping
almost matches F4 and wins top-k. Start F5 from the fold-specific F2 solution;
make the embeddings learn a bounded residual, not a replacement score. Measure
whether the residual helps difficult content such as monochrome photos,
deliberate motion, and low-light portraits. Do not add new signals or depth
before these hypotheses survive paired tests.

**Senior applied scientist:** An additive embedding residual may recover F4's
MAE and cross-dataset gains but still lose Flickr and top-k. Test one
content-conditioned scalar gate and one low-rank scalar-by-content interaction.
Both use the existing cache. Keep a small two-expert model as a conditional
follow-up only if the first variants show different, repeatable failure modes.

**Principal:** The technical auxiliary task is a training choice, not evidence
that a technical score should dominate the aesthetic output. Compare each
architecture under the same aesthetic objective first, then add SPAQ to the
survivor. Do not make a gated model win by giving it extra labels or compute.

**Senior:** Correct F3 and compare the same folds and seeds. Use grouped paired
differences rather than treating the 15 runs as 15 independent datasets. A
validation Spearman tie must be resolved by top-k and the event-grouped Moments
benchmark, with severe false positives explicitly counted.

## Proposed model family

All shapes below use `split_features()` from the immutable `joined-ten-v1`
contract. Embedding branches may use `LayerNorm(768)`; the scalar path uses
fold-fitted standardization only. `Linear(a,b)` includes a bias. Parameter
counts include an aesthetic head, a one-output technical head for a later
SPAQ experiment, and the nine-parameter F2-style scalar base. They exclude the
training-only dataset calibrators (eight parameters for four aesthetic
datasets) and frozen upstream vision models.

The implementation keeps the technical head disabled for aesthetic screening.
Its explicit `forward_with_aux` API taps the candidate's penultimate residual
representation and returns a separate technical score; normal `forward`
continues to return only the aesthetic score. The trainer CLI does not enable
the auxiliary task or load SPAQ data in the initial screen. Each F5 CLI run
requires fold-fitted normalization and a best F2 checkpoint from the same
fold, seed, split manifest, and normalization artifact. The copied F2 scalar
weights form the shared pre-sigmoid logit; a zero-initialized, tanh-bounded
residual makes the starting F5 aesthetic score exactly equal to that F2 fit.
`F3C` is the separate no-per-image-LayerNorm control; historical F3 remains
unchanged.

Every candidate predicts an aesthetic logit as
`z = b_F2 + w_F2 · scalars + residual` and returns `sigmoid(z)`.
Initialize the scalar base from an F2 fit on the **same development-training
fold**. Count that shared F2 fit in experiment time. Use a small initial
residual scale or near-zero final residual weights so the initial F5 score
remains close to F2. A1 is already one of the eight scalar-base coefficients;
an additional free A1 skip would be redundant. The A1 skip ablation means
removing its coefficient from that base and retraining.

| ID and hypothesis | Exact tensor path | Approx. parameters | Main risk |
| --- | --- | ---: | --- |
| **R: additive residual**. Embeddings correct cases missed by F2 without disturbing its overall ordering. | Each embedding: `LN(768) → 96 → GELU → 32`. Concatenate `32 + 32 + raw scalars 8 = 72`; residual `72 → 64 → GELU → 32 → 1`. Technical head `32 → 1`. | **163,755** | Residual may learn another global embedding score rather than useful exceptions. |
| **G: protected gated residual**. Content changes how the eight scores should be interpreted. | Each embedding: `LN(768) → 128 → GELU → 64`. Context `128 → 32 → 8`, gate `1 + 0.25*tanh(context)` initialized near 1. Gated scalars `8 → 32 → GELU`. Concatenate `64 + 64 + 32 + original scalars 8 = 168`; residual `168 → 256 → GELU → 128 → 1`. Technical head `128 → 1`. | **297,555** | Gate may saturate or learn unstable shortcuts; original scalar skip must remain. |
| **X: low-rank cross residual**. A small explicit content-by-signal interaction may rank the extreme top better than a broad gate. | Same two `768 → 128 → 64` branches as G. Context `128 → 16`; scalar projection `8 → 16`; elementwise product gives 16 interaction features. Concatenate `64 + 64 + 16 + original scalars 8 = 152`; residual `152 → 128 → GELU → 64 → 1`. Technical head `64 → 1`. | **246,635** | Can overfit dataset-specific interactions; rank-16 factorization may miss exceptions. |
| **M: two-expert residual**, conditional candidate. Different residual functions may handle visually distinct failure modes. | Same two `768 → 128 → 64` branches. Each of two experts takes `[64,64,8] = 136` and maps `136 → 128 → 64 → 1`. Context gate `128 → 32 → 2 → softmax`; weighted expert logits form the residual. Technical head `136 → 64 → 1`. | **281,198** | Expert collapse and harder calibration; inspect utilization, never label gates as explanations. |

The G shape implements the existing branch-aware plan. R tests whether the
gate is necessary. X tests a more constrained interaction. M should run only
if the first three reveal a credible specialization opportunity. F4 contains
431,633 parameters, so F5 variants are smaller; report both parameter count
and end-to-end training time. If a gain appears attributable merely to
capacity, rerun one F4 and one F5 at a matched parameter budget.

No variant needs additional production features or deeper embedding stacks at
this stage. A wider/deeper model is warranted only if training and validation
both underfit after the loss and preprocessing checks; a growing train-to-val
gap argues for less capacity or stronger regularization.

## Shared training choices and controlled ablations

1. Use the current grouped folds, identical cache identity, train-fold
   normalization, dataset-balanced homogeneous batches, seeds, batch budget,
   early-stopping policy, and training-only monotonic affine calibration.
   Freeze all experiment choices before any locked-test evaluation.
2. Start with calibrated Huber plus within-compatible-group ranking loss, as
   for F2-F4. Validate label-difference thresholds and group pairing so tied
   or noisy ratings do not dominate. Tune a small ranking-weight grid inside
   development CV only; then fix one value across variants. Compare top-1%,
   top-5%, and top-10% lift and overlap, not Spearman alone.
3. For the survivor, test technical auxiliary off versus on using SPAQ-only
   technical batches and a separate `q_tech` head. The aesthetic head never
   trains on SPAQ MOS. Select auxiliary weight inside development CV. Keep
   LIVE/PIQ23 external and KonIQ diagnostic only.
4. Test modest training-time scalar-signal dropout (individual score masking)
   and embedding-branch dropout against no dropout. Masking occurs **after**
   train-fold normalization; zero represents the training mean. Production
   inference still requires all ten valid inputs. Do not assume dropout helps:
   correlated AVA-trained signals may benefit, but rare important signals
   could become harder to use.
5. For G, ablate the gate while keeping the same branches and fusion width.
   For the winner, ablate SigLIP, CLIP, A1 term, other scalar families, and
   the technical head. A dropped feature is retrained, not merely zeroed at
   evaluation. Inspect gate saturation, score distribution, and residual size.
6. Keep native dataset calibrators training-only. Test shared versus
   dataset-specific calibration after selecting an architecture. A monotonic
   spline is justified only by reproducible residual curvature, not by a
   small validation gain. Export the single shared aesthetic head.

## Minimal fair experiment matrix

| Stage | Runs | Purpose and decision |
| --- | --- | --- |
| Preflight | F3 without per-image scalar LN on 2 folds × 2 seeds; F2/F4 results reused | Check whether F3's failure was primarily a normalization artifact. If corrected F3 approaches F2, include it as a serious scalar-only challenger. |
| Screen | R, G, X on the same 2 representative folds × 2 seeds, aesthetic loss only, fixed trainer/back-end policy | Eliminate variants with repeated regressions in macro Spearman **and** top-k. A single favorable fold is insufficient. |
| Confirmation | Best two F5 variants plus corrected F3 if competitive, all 5 folds × 3 seeds | Compare paired fold/seed deltas against both F2 and F4. Use group-aware bootstrap intervals, per-dataset results, top-k and calibration. |
| Focused tests | Winner: gate, modality, A1 and scalar-family ablations; SPAQ auxiliary off/on; dropout off/on | Attribute any gain to a component and reject gratuitous complexity. Run M only if stable, distinct failure modes remain. |
| Generalization | Leave-one-dataset-out on four aesthetic datasets, then event-grouped Moments benchmark | Require transfer and product selection benefit before release. Locked test is read once after architecture and training policy are frozen. |

The screen folds should be chosen in advance from contrasting dataset
profiles, then reused for all variants. Training throughput can use CPU or
MPS according to the separate benchmark, but a candidate comparison must use
the same validated numerics and equivalent training budgets. MLX would
require parity checks before joining this matrix.

## Promotion and stop rules

- Evaluate the **paired change** in macro Spearman, top-k lift, pairwise
  accuracy, and per-dataset Spearman versus F2 and F4. Seeds within a fold are
  repeated fits, not independent data splits. Publish uncertainty from
  source-group or fold-aware resampling.
- A candidate advances if it has a repeatable macro-ranking gain or a clear
  top-1%/5% gain without material regression on the other metric, and no
  unexplained loss on a dataset. A practical initial bar is about +0.01 macro
  Spearman or +0.01 normalized top-k lift; this is a decision threshold, not
  a statistical fact, and should be fixed before confirmation runs.
- Stop adding architecture if F5 remains within approximately 0.005 macro
  Spearman of F2/F4 and cannot recover F2's top-k advantage. Prefer the
  simpler model when paired uncertainty overlaps and product results tie.
- Require leave-one-dataset-out transfer, stable scores across seeds, and
  no rise in severe false positives. The Moments event-grouped benchmark must
  show better human-selected top-1%/5% quality for release. Score range,
  feature-hash, CPU/MPS parity, memory, and ingestion latency are hard gates.
- The locked aesthetic test and private Moments held-out set are never used
  to choose layer widths, loss weights, dropout, or checkpoints.

## Recommendation

Implement the corrected F3 control first. Then compare **R, G, and X** with
the same fold-specific F2 warm start. G is the primary candidate because it
tests the original content-conditioned hypothesis while protecting the strong
scalar path. R is the essential simplicity check; X is a constrained
interaction alternative. Defer M, a Transformer, and added signal inputs
until those controlled comparisons show a gap they could plausibly close.

This memo uses the completed `baseline-matrix-v6/aggregate-metrics.json`,
`src/image_aesthetics/models/fusion/`, and
`src/image_aesthetics/training/fusion_trainer.py` as the measured baseline and
implementation reference. Earlier AVA-first and Transformer-first discussion
in `architecture-alternatives.md` is superseded by the independent-dataset
and leakage policy in `final-fusion-model-plan.md`.
