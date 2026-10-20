# F5-G v1 release

This directory is the complete, public, inference-only release of the final
Moments Generator aesthetic fusion model. It contains no training images,
private benchmark media, labels, optimizer state, or locked-test rows.

Load it from this repository with:

```python
from image_aesthetics.inference import F5ReleaseScorer

scorer = F5ReleaseScorer("fusion/releases/f5-g-v1", device="cpu")
score = scorer.score(siglip_embedding, clip_embedding, {
    "a1_siglip": a1_score,
    "laion_clip": laion_score,
    "nima": nima_score,
    "musiq_aesthetic": musiq_aesthetic,
    "musiq_technical": musiq_technical,
    "sharpness": sharpness,
    "contrast": contrast,
    "colorfulness": colorfulness,
})
```

The default is the calibrated production score in `[0,1]`. Pass
`calibrated=False` to obtain the raw three-seed ensemble score for ranking
experiments. The loader verifies every release file against `manifest.json`
and fails closed on a changed file or incompatible contract.

`safetensors` and PyTorch are runtime requirements. All tensors in this bundle
are CPU-portable; inference can run on CPU or MPS. The scorer expects cached
embeddings and never runs SigLIP 2 or CLIP itself.

The locked-test metrics are recorded in `metrics.json`. The separate blinded
Moments human benchmark is still pending real raters, so this release does not
claim proven user-preference superiority over the existing heuristic.
