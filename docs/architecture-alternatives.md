# Architecture Alternatives

This document captures our brainstorming on different approaches to building a learned multi-signal fusion model. We evaluated five architectures, ranging from a simple baseline to an advanced hybrid system that combines supervised learning with reinforcement learning.

---

## The Goal

We want to replace this hand-tuned code:

```python
comp_rank = sum(weight[s] * signal_values[s] for s in weights)
```

with a learned model that:
- Accepts all 10 signals (2 embeddings of 768-d each + 8 scalar scores)
- Outputs a single aesthetic quality score (0.0 to 1.0)
- Learns the right fusion strategy from data instead of manual tuning
- Can adapt which signals matter based on the image content

---

## Training Data: The AVA Dataset

All architectures below will be trained on the **AVA (Aesthetic Visual Analysis) dataset**, which is the standard benchmark for image aesthetic quality:

- **255,000 images** from DPChallenge.com (a photography enthusiast community)
- Each image has a **distribution of human ratings** from 1 to 10 (not a single score, but a histogram of votes from many people)
- The **mean score** is used as the training label
- Available via Hugging Face (`Iceclear/AVA`) or academic torrents

To train our fusion model, we need to first **pre-compute all 10 signals** for every AVA image. This is a one-time cost:
- Extract SigLIP 2 embedding for each image (~3-4 hours on M5 Pro)
- Extract CLIP embedding for each image (~3-4 hours)
- Compute NIMA, LAION, MUSIQ scores (~2-3 hours)
- Compute OpenCV metrics (sharpness, contrast, colorfulness) (~30 minutes)

Once pre-computed, all signals are cached to disk. The actual model training then operates only on these cached features, which is extremely fast.

---

## Option A: Supervised Concatenation MLP (Baseline)

### How It Works

The simplest approach: concatenate everything into one big vector and pass it through a multi-layer neural network.

```
Input:
  SigLIP 2 embedding (768)  ─┐
  CLIP embedding (768)       ─┤
  NIMA score (1)             ─┤
  LAION-CLIP score (1)       ─┤
  LAION-SigLIP score (1)     ─┤
  MUSIQ aesthetic (1)        ─┤
  MUSIQ technical (1)        ─┼─→ Concatenate (1,544 total)
  Sharpness (1)              ─┤
  Contrast (1)               ─┤
  Colorfulness (1)           ─┘

Concat (1,544) → FC(512) → ReLU → Dropout → FC(128) → ReLU → FC(1) → Score
```

### Strengths

- Dead simple to implement and debug
- Very fast to train (~10 minutes per epoch on cached features)
- Tiny model (~800K parameters, ~3 MB)
- The fully-connected layers CAN learn some non-linear interactions between signals

### Weaknesses

- In practice, FC layers tend to learn the dominant signal and partially ignore the weaker ones. If NIMA alone predicts 70% of the score, the network might learn to just amplify NIMA and add minor corrections from other signals.
- No content-adaptive weighting — every image gets the same learned function applied, regardless of whether it's a landscape, portrait, or action shot.
- The embeddings (768-d each) dominate the input by sheer dimensionality. The 8 scalar signals (8 dimensions total) can get drowned out unless carefully balanced.

### When to Use

As a **baseline** to benchmark against. If a simple MLP already outperforms the hand-tuned weighted sum, that validates the entire approach.

### Feasibility on M5 Pro 24GB

Trivially feasible. Training uses ~100 MB of model memory plus ~3 GB for the cached dataset. No issues whatsoever.

---

## Option B: Gated Mixture-of-Experts (MoE) Fusion

### How It Works

Instead of one big network, we use multiple small specialist networks ("experts"), each focused on a different aspect of quality. A separate "gating network" looks at the image and decides how much to trust each expert's opinion.

```
                        ┌──────────────────────────────────────────────────┐
                        │              Gating Network                      │
                        │  (looks at the image embeddings and decides      │
                        │   how much to trust each expert)                 │
                        │                                                  │
                        │  Input: SigLIP + CLIP embeddings                 │
                        │  Output: [gate_1, gate_2, gate_3]                │
                        │          (weights that sum to 1.0)               │
                        └──────────┬───────────┬───────────┬───────────────┘
                                   │           │           │
                            weight_1=0.6  weight_2=0.1  weight_3=0.3
                                   │           │           │
                        ┌──────────▼───┐ ┌─────▼─────┐ ┌──▼──────────────┐
                        │  Expert 1     │ │ Expert 2   │ │ Expert 3        │
                        │  "Aesthetic"  │ │ "Technical"│ │ "Semantic"      │
                        │              │ │            │ │                 │
                        │  Inputs:     │ │ Inputs:    │ │ Inputs:         │
                        │  NIMA, LAION,│ │ Sharpness, │ │ SigLIP emb,     │
                        │  MUSIQ-aes   │ │ Contrast,  │ │ CLIP emb        │
                        │              │ │ Color,     │ │                 │
                        │  Small MLP   │ │ MUSIQ-tech │ │ Small MLP       │
                        │  → score_1   │ │ → score_2  │ │ → score_3       │
                        └──────┬───────┘ └─────┬──────┘ └──┬──────────────┘
                               │               │           │
                               └───────────────┼───────────┘
                                               │
                              Final Score = gate_1 × score_1
                                          + gate_2 × score_2
                                          + gate_3 × score_3
```

### Strengths

- **Content-adaptive weighting.** The gating network learns rules like: "for outdoor landscapes, trust the aesthetic expert more (gate_1 = 0.6)" or "for indoor shots, trust the technical expert more (gate_2 = 0.5)." This is the "which signal gets how much weightage" idea applied architecturally.
- Each expert can specialize in a different quality dimension, learning different fusion strategies for its specific subset of signals.
- The gating weights are interpretable — we can inspect them per image to understand why the model scored something highly.
- Moderate complexity, still fast to train.

### Weaknesses

- We need to decide upfront how to partition signals across experts (aesthetic vs. technical vs. semantic). If we partition wrong, we limit what each expert can learn.
- The gating network only produces a weighted combination of expert scores. The experts don't talk to each other — Expert 1 can't use information from Expert 2's signals.

### Connection to Your Multi-Armed Bandit Idea

This is conceptually close to what you described. Each expert is like an "arm" that has its own scoring strategy. The gating network is like a contextual bandit that learns which arm to trust for each image. The key difference from a true bandit: here, we ALWAYS use all arms (weighted combination) rather than picking one.

### Feasibility on M5 Pro 24GB

Easily feasible. 3 small experts + 1 gating network = ~300K parameters, ~1.5 MB. Training takes ~15 minutes per epoch.

---

## Option C: Contextual Multi-Armed Bandit (MAB)

### How It Works

A policy network takes the image context (embeddings) as input and outputs a weight vector — the relative importance of each signal for this specific image. The weighted sum of signals produces the score.

```
Image Context (SigLIP + CLIP embeddings)
        │
        ▼
┌───────────────────────────────────┐
│  Policy Network (small MLP)       │
│                                   │
│  Input: 1536-d (both embeddings)  │
│  Output: 10 weights (softmax)     │
│  [w1, w2, w3, w4, w5, w6, w7,    │
│   w8, w9, w10]                    │
└──────────────┬────────────────────┘
               │
    w1=0.15  w2=0.12  w3=0.18  w4=0.05  ...
               │
               ▼
Score = w1×SigLIP_proj + w2×CLIP_proj + w3×LAION + w4×LAION_SigLIP
      + w5×NIMA + w6×MUSIQ_aes + w7×MUSIQ_tech
      + w8×Sharpness + w9×Contrast + w10×Colorfulness
```

### When MAB Makes Sense vs. Doesn't

The multi-armed bandit framework is designed for **online exploration-exploitation** — situations where you repeatedly make decisions, observe rewards, and improve over time.

**Where it fits our problem:**
- When we have **user feedback**: the user accepts or rejects moments, providing a reward signal. The bandit can explore different weighting strategies and learn which works best for this specific user.
- For **model selection**: if we train multiple models (MLP, MoE, Transformer), a bandit can learn which model produces the best scores for which types of images.

**Where it doesn't fit well:**
- For **offline supervised training on AVA**: here we have a fixed dataset with labels. There's no exploration-exploitation tradeoff. Training a bandit on a fixed dataset is equivalent to training a supervised regression model — it reduces to Option A with extra steps.
- The bandit shines when data comes in **sequentially** and we learn online. AVA is a static batch dataset.

### Strengths

- Naturally adapts to user preferences over time (with online feedback)
- Learns per-image signal importance (not one-size-fits-all)
- Lightweight and fast

### Weaknesses

- In the offline-only setting (no user feedback yet), this is functionally equivalent to a supervised MLP that outputs weights. The "bandit" label doesn't add anything.
- True exploration-exploitation requires an interactive feedback loop, which means building a preference collection UI in the Moments Generator.
- With only 10 "arms" (signals), the exploration space is small enough that simpler approaches work just as well.

### Recommendation

Use MAB not as the core architecture, but as a **later-stage adaptation mechanism**. Specifically: train one of the other architectures (MoE or Transformer) first for strong offline performance, then add a bandit-style online adaptation layer that fine-tunes the model from user feedback.

### Feasibility on M5 Pro 24GB

Trivially feasible. The policy network is ~200K parameters.

---

## Option D: Cross-Attention Transformer

### How It Works

Instead of treating signals as a flat concatenated vector, we treat each signal as a separate "token" — like words in a sentence for a language model. We project everything into a shared dimensional space and let a small Transformer figure out how the signals relate to each other.

```
Signal Tokenization:
  SigLIP 2 (768)     → Linear projection → Token_1 (128-d)
  CLIP (768)          → Linear projection → Token_2 (128-d)
  NIMA (1)            → Linear projection → Token_3 (128-d)
  LAION-CLIP (1)      → Linear projection → Token_4 (128-d)
  LAION-SigLIP (1)    → Linear projection → Token_5 (128-d)
  MUSIQ aesthetic (1) → Linear projection → Token_6 (128-d)
  MUSIQ technical (1) → Linear projection → Token_7 (128-d)
  Sharpness (1)       → Linear projection → Token_8 (128-d)
  Contrast (1)        → Linear projection → Token_9 (128-d)
  Colorfulness (1)    → Linear projection → Token_10 (128-d)
  [CLS] (learned)     → Learnable param   → Token_0 (128-d)

            ┌───────────────────────────────────────────┐
            │         11 tokens, each 128-d              │
            └────────────────┬──────────────────────────┘
                             │
                   ┌─────────▼──────────┐
                   │  Transformer Layer 1 │
                   │  Self-Attention      │
                   │  (4 heads, d=128)    │
                   │                     │
                   │  Each token attends  │
                   │  to every other      │
                   │  token — learning    │
                   │  how signals relate  │
                   └─────────┬───────────┘
                             │
                   ┌─────────▼──────────┐
                   │  Transformer Layer 2 │
                   │  Self-Attention      │
                   │  (4 heads, d=128)    │
                   └─────────┬───────────┘
                             │
                   ┌─────────▼──────────┐
                   │  Take [CLS] token   │
                   │  → FC(128→64)       │
                   │  → FC(64→1)         │
                   │  → Score            │
                   └─────────────────────┘
```

### Why This is Powerful

Self-attention means every signal can "look at" every other signal when deciding its contribution. For example:

- The **sharpness token** can attend to the **CLIP embedding token** and learn: "this looks like a deliberately soft portrait with shallow depth of field — low sharpness is actually fine here, don't penalize it."
- The **colorfulness token** can attend to the **MUSIQ aesthetic token** and learn: "MUSIQ thinks this is well-composed AND it has low color — probably a stunning black-and-white shot, score it high."
- The **NIMA token** can attend to the **MUSIQ technical token** and learn: "NIMA says it's beautiful but MUSIQ-tech says it's blurry — trust MUSIQ-tech more for this image."

None of these conditional interactions are possible with a static weighted sum or even a simple MLP (where the interactions are implicit and harder to learn).

### Interpretability

We can extract the attention weights after inference and see exactly which signals the model paid attention to for any given image. This gives us a "why did it score this highly?" explanation, which is valuable for debugging and trust.

### Strengths

- Richest signal interactions of all options — learns conditional relationships
- Interpretable via attention weight visualization
- Still tiny: only 11 tokens × 2 layers × 4 heads = ~500K parameters
- Inference is extremely fast (microseconds — just matrix multiplications on 11 vectors)

### Weaknesses

- More complex to implement than MLP or MoE
- More parameters to train (though still small in absolute terms)
- Might be overpowered for the problem — if the signal interactions are actually simple, a Transformer will learn the same thing as an MLP with more overhead

### Feasibility on M5 Pro 24GB

Easily feasible. ~500K parameters ≈ 2 MB. Training takes ~25 minutes per epoch on cached features. This is a micro-Transformer, not a large language model.

---

## Option E: Hybrid Supervised + Reinforcement Learning

### The Core Idea

This is not a single architecture but a **training strategy** that can be applied to any of the architectures above (MoE or Transformer work best). It combines two phases:

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  PHASE 1: Supervised Pre-training on AVA                             │
│                                                                      │
│  • Train the fusion model (MoE or Transformer) on 255K AVA images   │
│  • Label: mean human aesthetic rating (1-10 scale)                   │
│  • Loss: MSE or Earth Mover's Distance                              │
│  • Result: Model learns "general human aesthetic taste"              │
│                                                                      │
│  This gives us a solid baseline that reflects crowd preferences.     │
│                                                                      │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  PHASE 2: Personalization via Reinforcement Learning                 │
│                                                                      │
│  • Collect pairwise preferences from the user:                       │
│    "Do you prefer image A or image B?"                               │
│  • Train a reward model (or use Direct Preference Optimization)     │
│  • Fine-tune the fusion model to maximize the USER's reward         │
│  • Result: Model adapts from "general taste" → "YOUR taste"         │
│                                                                      │
│  This is similar to how ChatGPT is trained:                          │
│  Pre-train on data → Fine-tune on human preferences (RLHF)          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Phase 1: Supervised Pre-training (Offline)

This is straightforward regression. We have a dataset (AVA) with images and scores. We compute all 10 signals for each image, feed them to the model, and train it to predict the AVA score.

- **Loss function options:**
  - **MSE (Mean Squared Error):** Predict the mean AVA score. Simple and effective.
  - **Earth Mover's Distance (EMD):** Instead of predicting just the mean, predict the entire 10-bin distribution of ratings. The EMD loss penalizes predictions that are "far" from the true distribution. Research shows this improves results by 3-5% in Spearman correlation.

### Phase 2: RLHF-Style Personalization (Online)

After Phase 1, the model knows "general photography aesthetics." But you might have specific preferences — maybe you value dramatic lighting over technical perfection, or you prefer candid moments over posed shots.

Phase 2 collects your preferences and adapts the model:

1. **Preference Collection:** Build a simple UI in the Moments Generator where you see two images side by side and pick which one you prefer. This produces pairwise comparison data: "for images A and B, the user prefers A."

2. **Reward Modeling or DPO:** Use these preferences to either:
   - Train a separate reward model using the Bradley-Terry framework: P(A preferred over B) = sigmoid(score(A) - score(B))
   - Or skip the reward model entirely and use Direct Preference Optimization (DPO), which is simpler and doesn't need a separate reward training phase.

3. **Fine-tuning:** Update the fusion model to align with your preferences. This is a small adjustment on top of the AVA pre-training — a few hundred preferences can meaningfully personalize the model.

### Where Bandits Fit In This Framework

This is where the multi-armed bandit idea actually shines:

- **Phase 1** is supervised learning. No bandits needed — we have labels.
- **Phase 2** can be framed as a contextual bandit: given an image (context), choose how to weight the signals (action), and the user's accept/reject is the reward. The bandit explores different weighting strategies early on and exploits the learned preferences over time.

So the recommended flow is: Supervised → RL/Bandit, not bandit from the start.

### How Much User Data Is Needed?

Research on RLHF suggests that meaningful personalization begins with as few as **200-500 pairwise comparisons**. A session of 10 minutes of "which do you prefer, A or B?" could produce enough data.

### Strengths

- Best of both worlds: strong general baseline + personalization
- Phase 1 works as a standalone improvement over the current hand-tuned weights
- Phase 2 is optional and incremental — we can ship Phase 1 and add Phase 2 later
- Aligns with state-of-the-art practices (this is literally how modern AI systems are aligned)

### Weaknesses

- Phase 2 requires UI work in the Moments Generator for preference collection
- Needs careful design to avoid "reward hacking" (model optimizing for the score number rather than actual quality)
- Total implementation effort is higher than the other options

### Feasibility on M5 Pro 24GB

Phase 1: Same as Options B or D above — trivially feasible.
Phase 2: Fine-tuning is even lighter than initial training (smaller dataset, fewer epochs). No issues.

---

## Hardware Feasibility Summary (Apple M5 Pro, 24 GB)

### What's Already Loaded During Ingestion

When the Moments Generator is running its ingestion pipeline, these models are loaded:

| Model | Memory |
|:---|:---|
| SigLIP 2 (vision encoder) | ~400 MB |
| CLIP ViT-L/14 | ~900 MB |
| NIMA (MobileNetV2) | ~14 MB |
| LAION MLP head | ~4 MB |
| MUSIQ-AVA (pyiqa) | ~150 MB |
| MUSIQ-KonIQ (pyiqa) | ~150 MB |
| **Total inference stack** | **~1.6 GB** |

That leaves ~18 GB free during ingestion (more than enough), and during training, the fusion models are tiny:

### Training Memory Estimates

| Scenario | Peak Memory | Time (30 epochs) | Feasible? |
|:---|:---|:---|:---|
| Pre-cache SigLIP+CLIP embeddings for AVA (250K images) | ~2.5 GB model + ~1.5 GB batch buffer | 3-4 hours (one-time) | Yes |
| Pre-compute all 8 scalar scores for AVA | ~1.6 GB models loaded | 2-3 hours (one-time) | Yes |
| Train Concat MLP (Option A) on cached features | ~100 MB model + ~3 GB cached data | ~15 minutes | Yes, trivial |
| Train Gated MoE (Option B) on cached features | ~200 MB model + ~3 GB cached data | ~20 minutes | Yes |
| Train Transformer (Option D) on cached features | ~250 MB model + ~3 GB cached data | ~25 minutes | Yes |
| Phase 2 RL fine-tuning | ~250 MB model + ~50 MB preference data | ~5-10 minutes | Yes, trivial |

The key insight: **pre-caching the heavy embeddings to disk is what makes everything feasible.** Without pre-caching, every training epoch would need to run SigLIP 2 and CLIP on 250K images, which would take hours per epoch. With pre-caching, the fusion model trains on small cached vectors and finishes in minutes.

### Cached Data Size

| Data | Size on Disk |
|:---|:---|
| SigLIP 2 embeddings for 250K images (768 × float32) | ~750 MB |
| CLIP embeddings for 250K images (768 × float32) | ~750 MB |
| 8 scalar scores for 250K images | ~8 MB |
| AVA labels | ~5 MB |
| **Total cached training data** | **~1.5 GB** |

This fits comfortably in memory for batch training and is fast to load from SSD.

---

## Comparison Matrix

| Criteria | A: MLP | B: MoE | C: Bandit | D: Transformer | E: Hybrid |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Can learn signal interactions** | Basic | Medium | Low | Best | Best |
| **Adapts to image content** | No | Yes (gating) | Yes (policy) | Yes (attention) | Yes |
| **Interpretable** | No | Medium (gate weights) | Low | High (attention maps) | High |
| **Improves from user feedback** | No | No | Yes | No (alone) | Yes |
| **Implementation effort** | 1 day | 1.5 days | 2 days | 1.5 days | 3-4 days total |
| **Parameters** | ~800K | ~300K | ~200K | ~500K | ~500K |
| **Training time (30 epochs)** | 15 min | 20 min | 15 min | 25 min | 30 min + 10 min |
| **Offline performance (AVA)** | Good | Good | Same as MLP | Best | Best |
| **Online adaptation** | None | None | Strong | None (alone) | Strong |

---

## Recommended Path Forward

Based on everything above, the recommended approach is:

1. **Start with Option A (MLP)** as a quick baseline to validate that learned fusion beats hand-tuned weights. This can be done in a day.

2. **Build Option D (Transformer)** as the main architecture. It has the best signal interaction modeling, is interpretable, and is still tiny. Benchmark it against the MLP baseline.

3. **Add Option E (Hybrid RL)** as Phase 2 once the Transformer is trained and deployed. This adds personalization without changing the core architecture — we just fine-tune the existing Transformer weights using preference data.

4. **Use Option C (MAB)** at the meta-level: if we end up with multiple trained models, use a bandit to select which model to trust for each image type, and adapt this selection from user feedback.

This gives us a clear progression from simple → sophisticated, where each step is independently valuable and deployable.

> **Superseded recommendation (2026-09-20):** the Transformer-first recommendation
> above was made before the dependency experiments and cross-dataset leakage review.
> The current decision is the branch-aware content-conditioned residual MLP in
> `docs/final-fusion-model-plan.md`. A flat MLP remains mandatory as a baseline; MoE
> and Transformer variants are controlled ablations, not presumed upgrades. A
> ten-token Transformer is not automatically more expressive in a useful way, and
> attention weights must not be presented as causal explanations.

## Initial Delivery Decisions (2026-09-19)

1. Train one universal, query-independent model first.
2. Compute and store its score during Moments Generator ingestion.
3. Treat the production ranking formula as an experiment: compare learned-score-only ranking against learned quality combined with query cosine and against the current heuristic.
4. Train the SigLIP-based LAION-style head, but use controlled ablations to decide whether it, the CLIP-based head, or both belong in the final model.
5. Exclude personalization, RL, and bandit adaptation from the initial deliverable.
6. Require complete re-indexing when the learned scorer is integrated.

## Recommended Target and Evaluation Protocol

### Training target

Use the full AVA 10-bin vote distribution as an auxiliary training target and derive the required scalar score as its expected value. Compare it with a mean-score regression baseline. Distribution supervision preserves disagreement and confidence information that a mean alone discards, while the derived expected value still provides the single `[0, 1]` ingestion score required by Moments Generator.

Do not assume distribution prediction is automatically better. It has a larger output head, a more complex loss, and can spend capacity matching annotator uncertainty that does not improve ranking. Promote it only if it beats mean regression on held-out ranking and top-selection metrics.

### Model-selection metrics

Primary metrics should reflect curation rather than exact numerical calibration:

- **Spearman rank correlation:** whether the model orders a collection similarly to human ratings.
- **Pairwise accuracy:** how often it gives the higher score to the human-preferred image in a pair.
- **Top-k quality/lift:** whether the small set of images actually selected for a reel has better human ratings than the heuristic or single-signal baselines.

Use MAE/MSE and calibration as secondary diagnostics. They detect badly scaled scores but should not choose the winner by themselves: a model can have a small numerical error while ordering the best images incorrectly.

### Leakage controls

NIMA, MUSIQ-AVA, and LAION predictors have prior exposure to AVA or related aesthetic data. Therefore:

- use official or reproducible image-disjoint splits;
- never train the SigLIP aesthetic head on images used to validate the fusion model;
- report ablations for embeddings alone, each pretrained score alone, scalar fusion, and full fusion;
- include at least one non-AVA external test set and a small Moments-style personal-media benchmark before claiming production improvement.
