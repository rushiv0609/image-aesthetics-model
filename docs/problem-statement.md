# Problem Statement & Context

## What is the Moments Generator?

The **Moments Generator** is a fully local, offline AI application that runs on Apple Silicon Macs. It solves a specific problem: you have thousands of photos and videos from trips, events, and daily life sitting on your hard drive, and you want to automatically create a highlight reel from a natural-language prompt like _"best sunset moments from Iceland"_ or _"fun group photos with friends."_

Here is how it works, end to end:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MOMENTS GENERATOR PIPELINE                       │
│                                                                         │
│  1. SCAN & DECODE                                                       │
│     Your photo/video library (HEIC, JPEG, MOV, MP4)                     │
│     → Hardware-accelerated decoding via Apple ImageIO & AVFoundation     │
│     → Extract representative frames from videos using scene detection   │
│                                                                         │
│  2. EMBED & SCORE  (← THIS IS WHERE OUR MODEL FITS)                    │
│     Each frame gets:                                                    │
│     • Two semantic embeddings (SigLIP 2 + CLIP, 768-dim each)           │
│     • Multiple aesthetic scores (NIMA, LAION, MUSIQ)                    │
│     • Technical quality metrics (sharpness, contrast, colorfulness)     │
│     • A combined "quality score" via hand-tuned weighted sum ← PROBLEM  │
│                                                                         │
│  3. STORE                                                               │
│     Everything goes into a Qdrant vector database:                      │
│     • Named vectors: {siglip2: [768-d], clip: [768-d]}                  │
│     • Payload: {nima: 0.72, laion: 0.68, sharpness: 0.91, ...}         │
│                                                                         │
│  4. QUERY & RANK                                                        │
│     User types a prompt → Embed prompt → Cosine similarity search       │
│     → Retrieved candidates get ranked by composite score                │
│     → AI Director Agent drafts a storyboard                             │
│                                                                         │
│  5. COMPILE                                                             │
│     Storyboard → FFmpeg with Apple VideoToolbox → Final MP4 highlight   │
└─────────────────────────────────────────────────────────────────────────┘
```

The application runs 100% on-device. No cloud APIs, no uploads, no internet required. It processes roughly 20 photos/second on an Apple M5 Pro.

## What Problem Are We Solving?

The core problem is in **step 2** above: how do we combine multiple quality signals into a single "how good is this image?" score?

### The Current Approach (Heuristic Weighted Sum)

Today, the Moments Generator computes a composite rank for every candidate image using this logic:

```python
# From moments-generator/app/core/director/nodes.py
signal_values = {
    "nima": 0.72,              # NIMA aesthetic score
    "laion_aesthetic": 0.68,   # LAION score on CLIP embedding
    "musiq_aesthetic": 0.75,   # MUSIQ-AVA aesthetic score
    "musiq_technical": 0.61,   # MUSIQ-KonIQ technical score
    "sharpness": 0.91,         # OpenCV Laplacian variance
    "contrast": 0.77,          # OpenCV histogram spread
    "colorfulness": 0.54,      # Hasler-Süsstrunk metric
    ...
}

# Simple weighted sum with hand-picked weights
comp_rank = sum(weight[s] * signal_values[s] for s in weights)
```

There are three "creative profiles" with different weight configurations:

| Profile | NIMA | LAION | MUSIQ-Aes | MUSIQ-Tech | Sharpness | Contrast | Colorfulness |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Cinematic Epic | 0.10 | 0.10 | 0.10 | 0.05 | 0.05 | 0.035 | 0.015 |
| Personal Moments | 0.05 | 0.05 | 0.05 | 0.125 | 0.125 | 0.0875 | 0.0375 |
| Balanced Journey | 0.15 | 0.15 | 0.15 | 0.05 | 0.05 | 0.035 | 0.015 |

These weights were hand-tuned through trial and error.

### Why This is a Problem

1. **Linear assumption.** A weighted sum assumes all signals combine additively. But image quality isn't additive. For example:
   - A sharp, low-color image might be a beautiful black-and-white documentary shot — the model should score it high. But the weighted sum penalizes low colorfulness equally regardless of context.
   - A slightly soft image with amazing lighting and composition (sunset golden hour) should score high, but the sharpness penalty drags it down.
   - The *meaning* of each signal changes depending on the other signals. A weighted sum can't capture these interactions.

2. **Static weights.** The same weights are applied to every image regardless of content. A landscape, a portrait, and an action shot all get scored with the same formula. There's no way for the system to say "for this type of image, trust MUSIQ more than NIMA."

3. **Manual tuning ceiling.** We've iterated on the weight profiles multiple times. Each iteration requires manual inspection of results, adjustment of numbers, and re-testing. This process is slow, subjective, and doesn't scale. We have no empirical evidence that these are optimal weights.

4. **No learning from data.** There are large datasets of human aesthetic judgments (like the AVA dataset with 255,000 rated images). We aren't leveraging any of that data. A learned model could discover patterns in human preferences that manual tuning can never find.

## What We Want to Build

A trained model that:

1. **Takes all available signals as input** — both the high-dimensional embeddings (SigLIP 2, CLIP) and the scalar quality scores (NIMA, LAION, MUSIQ, sharpness, contrast, colorfulness)
2. **Learns the optimal fusion function** from labeled aesthetic data (the AVA dataset, which contains 255,000 images rated by humans on a 1-10 aesthetic scale)
3. **Adapts signal importance based on the image** — for example, automatically learning to weight technical quality higher for portraits and composition higher for landscapes
4. **Is small enough to run on Apple M5 Pro 24GB** at both training and inference time
5. **Can potentially improve over time** from user feedback (the reinforcement learning angle)

## Explicit Scope Boundaries

**In scope:**
- A model that produces a standalone aesthetic quality score for an image, independent of any search query
- Training on the AVA dataset and potentially the user's own rated library
- Architectures that run on Apple Silicon (24GB)
- Integration into the Moments Generator ingestion pipeline

**Out of scope:**
- Query-dependent scoring (cosine similarity is handled separately in the retrieval layer)
- A general-purpose IQA library
- Cloud-based training or inference
- Real-time video quality assessment (we score individual frames)
