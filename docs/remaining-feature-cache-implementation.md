# Remaining Feature Cache: CLIP and MUSIQ

## Status

This cache implements the four remaining signals with locally available,
hash-pinned weights:

| Signal | Artifact | SHA-256 | Status |
|---|---|---|---|
| CLIP ViT-L/14 embedding | `timm/vit_large_patch14_clip_224.openai` cached OpenCLIP state dict | `9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e` | Implemented; strict state-dict load |
| LAION-CLIP score | `sac+logos+ava1-l14-linearMSE.pth` | `21dd590f3ccdc646f0d53120778b296013b096a035a2718c9cb0d511bff0f1e0` | Implemented; strict state-dict load |
| MUSIQ aesthetic | pyiqa `musiq-ava` | `e8d3f0671965dfe4301a1f7a4aedc8f49be8fed33b82dd5d116a9eb3ce2bd16b` | Implemented |
| MUSIQ technical | pyiqa `musiq` / KonIQ-10k | `e95806b9eae5f3814c410f574ba8e552362bd5bc63d758ed5b97860f5d6185aa` | Implemented |
| NIMA | No genuine trained checkpoint found locally | — | Blocked; ImageNet MobileNetV2 weights alone are not a NIMA model |

No weights are downloaded or substituted. CLIP and LAION share one CLIP vision
forward pass. The local CLIP checkpoint is loaded into OpenCLIP `ViT-L-14` with
strict key matching, then the unused text tower is released. Both MUSIQ models
load once per cache process and share decoded RGB inputs.

## Preprocessing and feature contract

- CLIP receives the same 224×224 RGB, aspect-preserving thumbnail with centered
  black padding used by the current local-five cache. It uses OpenAI CLIP mean
  and standard-deviation normalization and returns a 768-dimensional L2-unit
  vector.
- LAION consumes that normalized CLIP vector. Its raw 1–10 estimate is mapped
  to `[0,1]` using `clip((score - 1) / 9, 0, 1)`.
- MUSIQ receives a separate aspect-preserving RGB input capped at a 512-pixel
  longest side. MUSIQ itself evaluates the documented 224- and 384-long-side
  scales and an original-input stream capped at 256 patches. This avoids feeding
  it the square CLIP image. The cap bounds memory and attention cost; this is a
  versioned preprocessing choice and is not full 24MP sensor-resolution input.
- MUSIQ-AVA output is normalized from 1–10 to `[0,1]`; MUSIQ-KonIQ output is
  normalized from 0–100 to `[0,1]`.
- The cache stores float32 CLIP embeddings and scalar scores in immutable,
  checksummed shards. Rows preserve image ID, manifest position, source SHA-256,
  split, quarantine, and exposure metadata through the original manifest.
- The existing local-five shards are never modified. The joined-nine index stores
  references to shard/row locations after strict cache verification and exact
  checks of manifest hash, image ID, position, and source hash. It does not copy
  feature tensors.

## Validation already completed

An AADB deterministic random 1,000-image end-to-end smoke is at:

```text
/Users/rushivyas/datasets/image-aesthetics/aadb/features/remaining-four-smoke-random1000-v1/
```

It completed all 1,000 rows with zero failures; strict verification returned no
errors across four shards; the resume pass decoded and inferred zero rows;
observed RSS was about 1.28 GiB and swap delta was zero. Throughput was 8.59
images/second on CPU. CLIP embedding norms were between `0.99999976` and
`1.00000012`. Score distributions were non-degenerate: LAION-CLIP median 0.443,
MUSIQ-AVA median 0.423, and MUSIQ-KonIQ median 0.702. Every output was finite
and within `[0,1]`. This environment's PyTorch
reports MPS unavailable, so the smoke ran on CPU. The four providers and model
initialization used the Moments Generator virtual environment, which contains
OpenCLIP 3.3.0 and pyiqa 0.1.16.

## Running

From the repository root, the sequential all-dataset run is:

```bash
caffeinate -dimsu /Users/rushivyas/development/projects/moments-generator/.venv/bin/python \
  scripts/embeddings/run_remaining_four_all.py
```

The orchestrator processes AADB, KonIQ-10k, SPAQ, PARA, Flickr-AES, TAD66K, then
AVA. It writes per-dataset caches to
`<dataset>/features/remaining-four-v1/`, strict-verifies each cache, checks
zero-work resume, and writes the identity-checked index to
`<dataset>/features/joined-nine-v1/`. Orchestration state is written to
`fusion/features/remaining-four-v1/orchestrator-status.json`; per-image
progress is in each cache's `cache/progress.json` and SQLite ledger.
The orchestrator restarts the model process after at most 8,000 newly completed
rows (`--chunk-images`), preserving committed shards and resuming pending rows.
This bounds process-lifetime RSS; reduce the chunk size if a machine approaches
the 4 GiB stop gate. A partial chunk does not run full-cache verification; the
final chunk does strict verification and a zero-work resume.

Example status check:

```bash
cat /Users/rushivyas/datasets/image-aesthetics/fusion/features/remaining-four-v1/orchestrator-status.json
cat /Users/rushivyas/datasets/image-aesthetics/aadb/features/remaining-four-v1/cache/progress.json
```

The cache group is deliberately named `remaining-four-v1`, not `remaining-five`:
NIMA remains unavailable until a real trained checkpoint with known provenance
is supplied or approved for acquisition. A final ten-signal join/training cache
must wait for NIMA or explicitly version a nine-signal training contract.
