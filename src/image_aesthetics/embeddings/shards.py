"""Atomic, checksummed embedding shard storage.

Shards use safetensors when available.  A NumPy archive fallback is retained for
lightweight smoke/fault tests; it is still written to the versioned
``.safetensors`` path and identified in the sidecar metadata, so a production
run can enforce the safetensors dependency in its environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised in dependency checks
        raise RuntimeError("cache generation requires numpy") from exc
    return np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def write_atomic(path: Path, embeddings: Any, positions: list[int], image_ids: list[str],
                 *, cache_schema_version: int = 1, metadata_extra: dict[str, Any] | None = None) -> tuple[str, Path]:
    """Write one shard and metadata sidecar, returning checksum and path.

    The data file is fsynced before rename; the sidecar is renamed only after
    the data file is durable.  The caller must commit the SQLite row update
    only after this function returns.
    """
    np = _numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = path.with_suffix(path.suffix + ".json")
    if path.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite existing shard: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.close(fd)
        try:
            from safetensors.numpy import save_file  # type: ignore
        except ImportError:
            with temporary.open("wb") as handle:
                np.savez(handle, embeddings=np.asarray(embeddings, dtype=np.float32))
                handle.flush()
                os.fsync(handle.fileno())
            storage_format = "numpy_npz_compat"
        else:
            save_file({"embeddings": np.asarray(embeddings, dtype=np.float32)}, str(temporary))
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            storage_format = "safetensors"
        checksum = sha256_file(temporary)
        sidecar = {
            "schema_version": cache_schema_version,
            "storage_format": storage_format,
            "row_count": len(positions),
            "embedding_dim": int(np.asarray(embeddings).shape[1]),
            "positions": positions,
            "image_ids": image_ids,
            "sha256": checksum,
        }
        if metadata_extra:
            sidecar.update(metadata_extra)
        fd, side_temporary_name = tempfile.mkstemp(prefix=f".{metadata_path.name}.", dir=path.parent)
        side_temporary = Path(side_temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(sidecar, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            # Publish metadata first.  If the process dies after the data
            # rename but before SQLite commit, startup can still validate and
            # adopt the orphan shard.  A sidecar without its data is ignored.
            os.replace(side_temporary, metadata_path)
            _fsync_directory(path.parent)
        finally:
            if side_temporary.exists():
                side_temporary.unlink()
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        return checksum, metadata_path
    finally:
        if temporary.exists():
            temporary.unlink()


def read_shard(path: Path) -> tuple[Any, dict[str, Any]]:
    np = _numpy()
    metadata_path = path.with_suffix(path.suffix + ".json")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if sha256_file(path) != metadata.get("sha256"):
        raise ValueError(f"checksum mismatch: {path}")
    if metadata.get("storage_format") == "safetensors":
        from safetensors.numpy import load_file  # type: ignore
        embeddings = load_file(str(path))["embeddings"]
    else:
        with np.load(path) as loaded:
            embeddings = loaded["embeddings"]
    if embeddings.ndim != 2 or embeddings.shape[0] != metadata["row_count"] or embeddings.shape[1] != metadata["embedding_dim"]:
        raise ValueError(f"invalid shard shape: {path}")
    if len(metadata.get("positions", [])) != embeddings.shape[0] or len(metadata.get("image_ids", [])) != embeddings.shape[0]:
        raise ValueError(f"metadata row count mismatch: {path}")
    if not np.isfinite(embeddings).all():
        raise ValueError(f"non-finite embedding in shard: {path}")
    return embeddings, metadata


def verify_shard(path: Path, expected_dim: int = 768) -> dict[str, Any]:
    embeddings, metadata = read_shard(path)
    np = _numpy()
    norms = np.linalg.norm(embeddings, axis=1)
    if embeddings.shape[1] != expected_dim:
        raise ValueError(f"expected dimension {expected_dim}, got {embeddings.shape[1]}")
    if metadata.get("unit_norm", True) and len(norms) and not np.allclose(norms, 1.0, atol=5e-3):
        raise ValueError(f"unit-norm validation failed: {path}")
    if len(metadata["positions"]) != len(set(metadata["positions"])):
        raise ValueError(f"duplicate manifest positions: {path}")
    if len(metadata["image_ids"]) != len(set(metadata["image_ids"])):
        raise ValueError(f"duplicate image IDs: {path}")
    return metadata
