from __future__ import annotations

from pathlib import Path

import pytest

from image_aesthetics.embeddings.cache import CacheConfig, CacheGenerator
from image_aesthetics.embeddings.preprocess import ManifestItem


def test_pipeline_preserves_order_and_resumes_completed_rows(tmp_path: Path):
    np = pytest.importorskip("numpy")
    files = []
    items = []
    for position in range(5):
        path = tmp_path / f"{position}.jpg"
        path.write_bytes(b"fixture")
        files.append(path)
        stat = path.stat()
        items.append(ManifestItem(str(position), str(path), position, stat.st_size, stat.st_mtime_ns))

    def prepare(path: Path, size: int):
        value = int(path.stem)
        return np.full((size, size, 3), value, dtype=np.uint8), "fake"

    def backend(images):
        output = np.zeros((len(images), 768), dtype=np.float32)
        for row, image in enumerate(images):
            output[row, 0] = 1.0
        return output

    class Embedder:
        def embed(self, images):
            from image_aesthetics.embeddings.backend import SigLIP2Embedder
            return SigLIP2Embedder("test", "test", backend=backend).embed(images)

    config = CacheConfig(tmp_path / "cache", model_revision="immutable-revision", batch_size=2,
                         decode_workers=2, shard_size=3, progress_interval_seconds=0.01)
    first = CacheGenerator(config, embedder_factory=Embedder, prepare_fn=prepare).run(items, run_id="first")
    assert first["state"] == "completed"
    assert first["completed_images"] == 5
    second = CacheGenerator(config, embedder_factory=Embedder, prepare_fn=prepare).run(items, run_id="second")
    assert second["state"] == "completed"
    assert second["completed_images"] == 5
    assert second["failed_images"] == 0


def test_corrupt_image_is_explicit_failure(tmp_path: Path):
    np = pytest.importorskip("numpy")
    path = tmp_path / "bad.jpg"; path.write_bytes(b"bad")
    stat = path.stat()
    item = ManifestItem("bad", str(path), 0, stat.st_size, stat.st_mtime_ns)

    def prepare(_path, _size):
        raise ValueError("truncated image")

    config = CacheConfig(tmp_path / "cache", model_revision="immutable-revision", progress_interval_seconds=0.01)
    result = CacheGenerator(config, prepare_fn=prepare, embedder_factory=lambda: None).run([item], run_id="fault")
    assert result["state"] == "completed"
    assert result["failed_images"] == 1


def test_resume_retry_never_overwrites_committed_vectors(tmp_path: Path):
    np = pytest.importorskip("numpy")
    from image_aesthetics.embeddings.shards import read_shard

    items = []
    for position in range(3):
        path = tmp_path / f"{position}.jpg"
        path.write_bytes(b"fixture")
        stat = path.stat()
        items.append(ManifestItem(str(position), str(path), position, stat.st_size, stat.st_mtime_ns))

    failed = {"1": True}

    def prepare(path: Path, size: int):
        if failed.get(path.stem, False):
            raise ValueError("injected decoder failure")
        return np.full((size, size, 3), int(path.stem) + 1, dtype=np.uint8), "fake"

    def backend(images):
        output = np.ones((len(images), 768), dtype=np.float32)
        return output / np.linalg.norm(output, axis=1, keepdims=True)

    class Embedder:
        def embed(self, images):
            from image_aesthetics.embeddings.backend import SigLIP2Embedder
            return SigLIP2Embedder("test", "test", backend=backend).embed(images)

    cache_dir = tmp_path / "cache"
    first_config = CacheConfig(cache_dir, model_revision="immutable-revision", batch_size=3,
                               decode_workers=2, shard_size=3, progress_interval_seconds=0.01)
    first = CacheGenerator(first_config, embedder_factory=Embedder, prepare_fn=prepare).run(items, run_id="first")
    assert first["completed_images"] == 2
    original = cache_dir / "shard-00000.safetensors"
    original_bytes = original.read_bytes()

    failed.clear()
    retry_config = CacheConfig(cache_dir, model_revision="immutable-revision", batch_size=3,
                               decode_workers=2, shard_size=3, retry_permanent=True,
                               progress_interval_seconds=0.01)
    second = CacheGenerator(retry_config, embedder_factory=Embedder, prepare_fn=prepare).run(items, run_id="second")
    assert second["completed_images"] == 3
    assert original.read_bytes() == original_bytes
    retry_shard = cache_dir / "shard-00001.safetensors"
    assert retry_shard.exists()
    assert read_shard(original)[1]["image_ids"] == ["0", "2"]
    assert read_shard(retry_shard)[1]["image_ids"] == ["1"]

    import sqlite3
    rows = sqlite3.connect(cache_dir / "work.sqlite3").execute(
        "SELECT image_id,shard_id,shard_row,state FROM items ORDER BY manifest_position"
    ).fetchall()
    assert rows == [("0", 0, 0, "completed"), ("1", 1, 0, "completed"), ("2", 0, 1, "completed")]
