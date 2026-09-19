# Input Signals

This document describes every signal available to the aesthetic scoring model. These signals are already being computed during the ingestion layer of the Moments Generator pipeline and stored in the Qdrant vector database alongside each image.

## Important Design Decision: No Query-Dependent Signals

We intentionally **exclude cosine similarity** (the relevance score between a user's text query and the image embedding) from this model. Here's why:

- Cosine similarity is computed at **query time**, not at ingestion time. It depends on what the user types.
- The same image has a different cosine similarity for "sunset over mountains" vs. "group photo at dinner."
- This model produces a **standalone image quality score** — an inherent property of the image itself, independent of any query.
- Query relevance is handled separately in the retrieval layer and combined with the aesthetic score at ranking time.

Think of it this way: this model answers "how aesthetically good is this image?" while cosine similarity answers "how relevant is this image to what the user asked for?" These are two different questions.

---

## Signal 1: SigLIP 2 Embedding (768-dimensional vector)

**What it is:** A dense vector representation of the image's visual content, extracted by Google's SigLIP 2 vision-language model (`google/siglip2-base-patch16-224`).

**How it's computed:** The image is resized to 224×224 pixels, passed through the SigLIP 2 vision encoder (a Vision Transformer), and the output embedding is L2-normalized to unit length. This produces a 768-dimensional float32 vector.

**What it captures:** High-level semantic understanding of the image — what objects are present, the scene type (beach, mountain, city), visual style, composition patterns, and abstract visual concepts. SigLIP 2 was trained with a sigmoid-based pairwise loss, which makes it particularly good at fine-grained similarity discrimination.

**Why it matters for aesthetics:** The embedding implicitly encodes composition, scene type, and visual complexity. Two images of the same mountain from different angles will have similar embeddings, but a well-composed shot will cluster differently in embedding space than a poorly framed one. A learned model can discover these patterns.

**Characteristics:**
- Dimension: 768 floats
- Range: Each dimension roughly [-1, 1], L2-normalized so the vector norm = 1
- Computed by: `moments-generator/app/core/embedder.py` (SigLIP 2, Apple MLX primary / PyTorch MPS fallback)
- Memory per image: 3,072 bytes (768 × 4 bytes)

---

## Signal 2: CLIP Embedding (768-dimensional vector)

**What it is:** A dense vector representation from OpenAI's CLIP model (`ViT-L-14`), providing a second perspective on the image's visual semantics.

**How it's computed:** Same pipeline as SigLIP 2 — image resized to 224×224, passed through the CLIP ViT-L/14 vision encoder, output L2-normalized.

**What it captures:** Similar semantic information to SigLIP 2, but with different biases due to different training (CLIP uses contrastive softmax loss vs. SigLIP's sigmoid loss). CLIP tends to be better at hard-negative discrimination and has stronger text-image alignment for diverse prompts.

**Why we have both SigLIP 2 and CLIP:** They provide complementary views. Having two embedding spaces gives the model more information to work with. The LAION aesthetic predictor (Signal 3) was specifically trained on CLIP embeddings, so we need the CLIP embedding to compute that score.

**Characteristics:**
- Dimension: 768 floats
- Range: Each dimension roughly [-1, 1], L2-normalized
- Computed by: `moments-generator/app/core/clip_embedder.py` (OpenAI CLIP via open_clip, PyTorch MPS)
- Memory per image: 3,072 bytes

---

## Signal 3: LAION Aesthetic Score (CLIP-based)

**What it is:** An aesthetic quality prediction from a small MLP (multi-layer perceptron) trained by the LAION research group. It takes the CLIP embedding as input and predicts how aesthetically pleasing the image is.

**How it's computed:** The 768-dimensional CLIP embedding is passed through a small neural network:

```
CLIP embedding (768) → Linear(768, 1024) → Dropout → Linear(1024, 128)
→ Dropout → Linear(128, 64) → Dropout → Linear(64, 16) → Linear(16, 1)
```

The raw output is on a 1-10 scale (calibrated to AVA dataset scores), which we normalize to 0.0–1.0 using `(score - 1) / 9`.

**What it captures:** High-level aesthetic appeal — visual beauty, pleasing color palettes, good composition. It was trained on a combination of the SAC (Simulacra Aesthetic Captions), LAION logos, and AVA datasets. It essentially learned: "given what this image looks like (CLIP embedding), how would a crowd of people rate its beauty?"

**Why it matters:** This is a proven approach — LAION used this predictor to filter billions of web images for aesthetic quality. It's the baseline we want to beat with a more sophisticated fusion model.

**Characteristics:**
- Range: 0.0 to 1.0 (normalized from 1-10 scale)
- Depends on: Signal 2 (CLIP embedding) — no extra vision pass needed
- Model size: ~4 MB (tiny MLP)
- Computed by: `LAIONAestheticScorer` in `moments-generator/app/core/aesthetic.py`

---

## Signal 4: LAION Aesthetic Score (SigLIP 2-based) — TO BE TRAINED

**What it is:** The same MLP architecture as Signal 3, but trained on **SigLIP 2 embeddings** instead of CLIP embeddings. This does not yet exist — training it is one of the first tasks of this project.

**Why we want it:** To directly compare whether SigLIP 2 or CLIP produces a better aesthetic predictor when used as the base embedding. SigLIP 2's sigmoid training may capture finer-grained aesthetic differences. Having both LAION scores also gives the fusion model two different "opinions" on aesthetics, one from each embedding space.

**Characteristics:**
- Range: 0.0 to 1.0 (same as CLIP-based version)
- Depends on: Signal 1 (SigLIP 2 embedding)
- Status: **Not yet trained.** Will be trained on the AVA dataset as part of this project.

---

## Signal 5: NIMA Aesthetic Score

**What it is:** Neural Image Assessment (NIMA) is a model originally developed by Google that predicts the distribution of human aesthetic ratings for a photograph. Our implementation uses a MobileNetV2 backbone.

**How it's computed:** The 224×224 RGB image is preprocessed with ImageNet normalization and passed through a MobileNetV2 network. Instead of predicting a single score, the model outputs a probability distribution across 10 bins (ratings 1 through 10). The final score is the weighted mean of this distribution:

```
score = sum(i * probability[i] for i in range(1, 11))
```

This is normalized to 0.0–1.0 using `(score - 1) / 9`.

**What it captures:** Photographic aesthetic preferences learned from the AVA (Aesthetic Visual Analysis) dataset — composition, rule of thirds, leading lines, lighting quality, color harmony, and overall visual appeal as judged by photography enthusiasts.

**Why it matters:** NIMA operates directly on pixel data, not on pre-computed embeddings. This means it can capture low-level visual patterns (texture, grain, bokeh quality) that embedding-based scores might miss. It provides a different perspective than the embedding-based LAION score.

**Characteristics:**
- Range: 0.0 to 1.0
- Input: Raw 224×224 RGB pixels (not embeddings)
- Model backbone: MobileNetV2 (~14 MB)
- Computed by: `NimaScorer` in `moments-generator/app/core/aesthetic.py`

---

## Signal 6: MUSIQ Aesthetic Score (AVA-trained)

**What it is:** Google's MUSIQ (Multi-Scale Image Quality Transformer) trained on the AVA dataset for aesthetic quality prediction. Unlike NIMA's CNN approach, MUSIQ uses a Vision Transformer that processes images at their native resolution with multi-scale patches.

**How it's computed:** The image is divided into patches at multiple scales (no forced resize to 224×224, preserving aspect ratio information). These multi-scale patches are fed into a Vision Transformer. The model outputs a scalar score on the 1-10 AVA scale, normalized to 0.0–1.0.

**What it captures:** Multi-scale composition and framing quality. Because it doesn't force-resize images, it can assess composition at the image's natural aspect ratio. A panoramic landscape and a square portrait are evaluated differently, which is more faithful to how humans perceive composition.

**Why it's different from NIMA:** NIMA uses a CNN that processes a fixed 224×224 crop. MUSIQ uses a Transformer on multi-scale patches at native resolution. MUSIQ generally has higher correlation with human judgments on modern IQA benchmarks.

**Characteristics:**
- Range: 0.0 to 1.0 (normalized from 1-10 scale)
- Input: Raw image pixels at native resolution (internally patched)
- Computed via: `pyiqa.create_metric('musiq-ava')` wrapped in `MUSIQScorer`
- Model size: ~150 MB (Vision Transformer)

---

## Signal 7: MUSIQ Technical Score (KonIQ-10k trained)

**What it is:** The same MUSIQ Vision Transformer architecture, but trained on the KonIQ-10k dataset for **technical quality** rather than aesthetic quality.

**How it's computed:** Same multi-scale patching as Signal 6, but the model was trained to predict technical distortion ratings. Output is on a 0-100 scale, normalized to 0.0–1.0.

**What it captures:** Technical image flaws that degrade perceived quality:
- Compression artifacts (JPEG blocking, banding)
- Sensor noise (grain in low-light shots)
- Optical blur (lens softness, diffraction)
- Motion blur (camera shake, subject movement)
- Color aberrations and fringing

**Why it matters:** An image can be aesthetically composed but technically degraded (e.g., a beautiful sunset taken with a shaky hand). This signal specifically catches those technical problems. It's complementary to the aesthetic signals — NIMA/LAION might rate a technically poor image highly if the composition is good, but MUSIQ-technical will flag the quality issues.

**Characteristics:**
- Range: 0.0 to 1.0 (normalized from 0-100 scale)
- Input: Raw image pixels at native resolution
- Computed via: `pyiqa.create_metric('musiq')` (KonIQ-10k variant)
- Model size: ~150 MB

---

## Signal 8: Sharpness

**What it is:** A classical computer vision metric measuring how "in focus" and crisp an image is, using the variance of the Laplacian filter.

**How it's computed:**

```python
gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
sharpness = clip(laplacian.var() / 1200.0, 0.0, 1.0)
```

The Laplacian filter responds to edges and rapid intensity changes. A sharp image has many high-frequency edges (high variance), while a blurry image has few (low variance).

**What it captures:** Focus quality and motion blur. A perfectly focused landscape will score near 1.0. A motion-blurred or out-of-focus shot will score near 0.0.

**Why it matters as a separate signal:** While MUSIQ-technical also detects blur, the Laplacian sharpness metric is:
- Extremely fast (pure OpenCV, no neural network)
- Deterministic and interpretable
- Focused specifically on spatial frequency content, not overall quality

The fusion model can learn to combine this fast, targeted metric with the broader neural quality assessments.

**Characteristics:**
- Range: 0.0 to 1.0
- Computation: OpenCV, CPU-only, <1ms per image
- Normalization: `laplacian.var() / 1200.0` (empirically calibrated for 224px images)

---

## Signal 9: Contrast

**What it is:** A measure of the dynamic range of luminance in the image, computed using histogram percentiles.

**How it's computed:**

```python
gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
p5, p95 = np.percentile(gray, [5, 95])
contrast = clip((p95 - p5) / 220.0, 0.0, 1.0)
```

This measures the spread between the 5th and 95th percentile of pixel brightness. Using percentiles instead of min/max makes it robust to outlier pixels.

**What it captures:** Tonal range and dynamic feel. High-contrast images (dramatic sunsets, mountain silhouettes against bright sky, chiaroscuro lighting) score high. Flat, hazy, or washed-out images score low.

**Why it matters:** Contrast strongly correlates with perceived visual impact. Golden hour photography, dramatic weather, and well-lit scenes naturally have higher contrast. Foggy, overcast, or indoor images without directional lighting have lower contrast.

**Characteristics:**
- Range: 0.0 to 1.0
- Computation: OpenCV + NumPy, CPU-only, <1ms per image
- Normalization: `spread / 220.0` (220 is near-maximum possible spread for 8-bit images)

---

## Signal 10: Colorfulness

**What it is:** A perceptual metric for how vivid and chromatic an image appears, based on the Hasler & Süsstrunk opponent color model.

**How it's computed:**

```python
R, G, B = pixels[:,:,0], pixels[:,:,1], pixels[:,:,2]
rg = abs(R - G)           # Red-green opponent channel
yb = abs(0.5*(R+G) - B)   # Yellow-blue opponent channel

std_root  = sqrt(rg.std()^2 + yb.std()^2)
mean_root = sqrt(rg.mean()^2 + yb.mean()^2)

colorfulness = clip((std_root + 0.3 * mean_root) / 100.0, 0.0, 1.0)
```

**What it captures:** How vivid and saturated the colors in the image are. A vibrant sunset, autumn foliage, or tropical beach scores high. A black-and-white photo, foggy scene, or dimly lit interior scores low.

**Why it matters — and why it needs to be used carefully:** Colorfulness is a double-edged signal:
- **Positive:** Vivid landscapes, sunsets, and colorful scenes genuinely tend to be more visually appealing
- **Negative:** Over-relying on it biases toward flower photos and neon signs rather than genuinely epic moments. A stunning black-and-white photograph would be unfairly penalized.

This is exactly why we need a learned model — it can learn the *conditional* value of colorfulness (high colorfulness is great for sunsets but irrelevant for documentary-style shots), whereas a static weight treats it the same everywhere.

**Characteristics:**
- Range: 0.0 to 1.0
- Computation: NumPy only, CPU, <1ms per image
- Normalization: `metric / 100.0` (empirically calibrated)

---

## Signal Summary Table

| # | Signal | Type | Dimension | Source | Captures |
|:--|:---|:---|:---|:---|:---|
| 1 | SigLIP 2 embedding | Vector | 768 | SigLIP 2 ViT | Semantic content, scene understanding |
| 2 | CLIP embedding | Vector | 768 | CLIP ViT-L/14 | Semantic content (contrastive) |
| 3 | LAION score (CLIP) | Scalar | 1 | MLP on CLIP emb | High-level aesthetic appeal |
| 4 | LAION score (SigLIP) | Scalar | 1 | MLP on SigLIP emb | Aesthetic appeal (to be trained) |
| 5 | NIMA aesthetic | Scalar | 1 | MobileNetV2 on pixels | Composition, lighting, harmony |
| 6 | MUSIQ aesthetic | Scalar | 1 | ViT on multi-scale patches | Multi-scale composition |
| 7 | MUSIQ technical | Scalar | 1 | ViT on multi-scale patches | Distortion, noise, blur |
| 8 | Sharpness | Scalar | 1 | OpenCV Laplacian | Focus quality, motion blur |
| 9 | Contrast | Scalar | 1 | OpenCV histograms | Dynamic range, tonal spread |
| 10 | Colorfulness | Scalar | 1 | Hasler-Süsstrunk | Chroma richness, vividness |

**Total input to the fusion model:** 2 vectors of 768 dimensions each + 8 scalar values = 1,544 features.
