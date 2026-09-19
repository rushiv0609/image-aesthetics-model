# Agent Instructions

## Project Identity

This is the **Image Aesthetics Model** project. It is a standalone training and research repository that produces a scoring model consumed by the **Moments Generator** application (`/Users/rushivyas/development/projects/moments-generator`).

The Moments Generator is a local, offline AI video curation tool for Apple Silicon. It ingests a user's photo/video library, computes semantic embeddings and quality scores, stores everything in a Qdrant vector database, and then uses an AI "Director Agent" to curate highlight reels from natural-language prompts. The aesthetic scoring model built in THIS repo replaces the hand-tuned weighted-sum scoring currently used in that application.

## Hard Rules

1. **Scope is Moments Generator only.** Every architectural decision, training script, and evaluation metric must directly serve integration back into the Moments Generator pipeline. This is not a general-purpose image quality library.

2. **Hardware constraint: Apple M5 Pro, 24 GB unified memory.** All models must be trainable AND runnable on this hardware. No CUDA, no cloud GPU dependencies. Use PyTorch MPS or Apple MLX. If a model is too large to train locally, it is out of scope.

3. **Input signals are fixed.** The model consumes exactly the signals listed in `docs/signals.md`. Do not invent new signals without explicit user approval. In particular, **do not include cosine similarity or any query-dependent signal** — this model produces a standalone image quality score independent of any search query.

4. **Pre-cache heavy computation.** The SigLIP 2 and CLIP vision backbones are expensive (~1.6 GB combined). For training, always pre-cache their embeddings to disk rather than running them in the training loop. The fusion model itself should only operate on cached embeddings and pre-computed scalar scores.

5. **Documentation first.** We are currently in a brainstorming and planning phase. Keep docs updated as decisions are made. Read the full documentation before making changes:
   - `docs/problem-statement.md` — What and why
   - `docs/signals.md` — The 10 input signals
   - `docs/architecture-alternatives.md` — The architectural options under discussion

## Key Context from Moments Generator

The current scoring logic lives in `moments-generator/app/core/director/nodes.py`. The specific line being replaced is:

```python
comp_rank = sum(norm_weights[s] * signal_values[s] for s in norm_weights if s in signal_values)
```

This is a simple weighted sum where `norm_weights` are hand-tuned per "creative profile" (cinematic, personal, journey). The weights are static regardless of image content. The model built in this repo will replace this line with a learned prediction.

The scoring signals are computed during ingestion in `moments-generator/app/core/aesthetic.py` and `moments-generator/app/core/pipeline.py`, then stored as metadata in the Qdrant vector database. At query time, they are read from the database and passed to the scoring function. The model built here must accept these pre-computed signals as input and return a single float score.

## Integration Point

The final trained model will be loaded in the Moments Generator as a `UnifiedScorer` class (similar to the existing `NimaScorer`, `LAIONAestheticScorer`, and `MUSIQScorer` singletons in `aesthetic.py`). It will:
1. Accept a dictionary of pre-computed signal values (the 10 signals from `docs/signals.md`)
2. Return a single float score in the range [0.0, 1.0]
3. Run on MPS or CPU with negligible latency (the model will be tiny — under 1M parameters)
