# AVA SigLIP 2 cache operations

The cache generator is standalone and does not download AVA, model weights, or
personal images.  Images, manifests, SQLite ledgers, shards, and weights belong
under an external `IAM_DATA_ROOT` (recommended:
`/Users/rushivyas/datasets/image-aesthetics`).

## Run and resume

Prepare a deterministic JSONL/CSV/JSON manifest containing `image_id` and
`source_path`, then run:

```bash
IAM_DATA_ROOT=/Users/rushivyas/datasets/image-aesthetics \
python scripts/embeddings/cache_siglip2_ava.py run \
  --manifest /external/ava/prepared/v1/dataset.jsonl \
  --cache-version v1 \
  --model-revision <immutable-hugging-face-commit>
```

The command takes an exclusive lock on the cache version.  A second writer is
rejected.  A resumed run verifies committed shard checksums, resets orphaned
decode/inference/staged rows, and schedules only pending or retryable rows.
Changed source metadata requires a new cache version (or an explicitly rebuilt
manifest); embeddings from different model revisions or preprocessing contracts
are never mixed.

## Feature manifest validation

`materialize_feature_manifest` emits `feature-manifest-v2`. Before a source is
included, it is decoded through the same Pillow open, full-load, EXIF transpose,
and RGB conversion path used by the ten-signal runtime. Pillow's tolerant
truncated-image mode is rejected. Decode failures are excluded from
`features.jsonl` and recorded with a `source_decode_invalid` reason in
`rejected.jsonl`; the decode policy is recorded in `audit.json`. Resizing and
model preprocessing are unchanged.

A corrected manifest has a new manifest/cache identity. The resumable runner
only retains committed rows when cache identity, manifest hash, source
fingerprints, and shard row coordinates all match. It does not support
transferring completed rows across a filtered or reordered manifest, even when
most source images are unchanged. Do not relabel old shard metadata to make it
appear to match a new identity; use the existing cache only with its original
immutable manifest, or start a new cache for the corrected manifest.

The default backend is the repository's standalone MLX FP16 SigLIP2 Base-224
vision implementation. Install the optional `mlx` extra on the target Apple
Silicon environment; tests can inject a deterministic backend without loading
the checkpoint. The loader passes the requested immutable revision to
`transformers.AutoModel.from_pretrained` and never downloads AVA itself.

The external cache contains `work.sqlite3` (authoritative), committed
`shard-*.safetensors` files plus sidecars, `manifest.json`, and an atomically
replaced `progress.json`.  Safetensors is the production storage format.  A
NumPy archive fallback is available only for dependency-light smoke/fault tests
and is identified in each sidecar.

## Read-only monitoring and verification

```bash
python scripts/embeddings/cache_siglip2_ava.py status --cache-dir /external/cache
python scripts/embeddings/cache_siglip2_ava.py failures --cache-dir /external/cache
python scripts/embeddings/cache_siglip2_ava.py verify --cache-dir /external/cache
```

SQLite uses WAL, a single dedicated writer thread, short batched transactions,
and bounded event backpressure.  `progress.json` is a monitoring view only;
SQLite and committed shard hashes are the sources of truth.  Final progress
states are `completed`, `failed`, or `cancelled`.

## Fault and cancellation behavior

`SIGINT`/`SIGTERM` stops scheduling, lets the active writer finish only its
committed shard, drains terminal ledger events, writes a final progress state,
and releases the cache lock.  Temporary shard files are discarded.  A crash
after shard rename but before the ledger transaction is reconciled as an orphan
when the next run starts; a corrupted shard is quarantined and regenerated.
