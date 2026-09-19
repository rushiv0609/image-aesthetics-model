from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from image_aesthetics.embeddings.ledger import Ledger, LedgerLock


def test_ledger_duplicate_events_are_idempotent(tmp_path: Path):
    ledger = Ledger(tmp_path / "work.sqlite3", batch_size=1)
    ledger.open()
    ledger.seed_items([{
        "image_id": "a", "manifest_position": 0, "source_path": "/tmp/a",
        "source_size": 1, "source_mtime_ns": 1, "source_fingerprint": None,
    }])
    ledger.start_run("r", model_revision="rev", preprocessing_hash="hash", dataset_fingerprint="d",
                     cache_schema_version=1, config={})
    ledger.start_writer()
    ledger.emit("state", image_id="a", state="decoding")
    ledger.emit("state", image_id="a", state="decoding")
    ledger.emit("state", image_id="a", state="permanent_error", error_type="x", error_message="bad")
    ledger.stop("completed")
    row = ledger.read("SELECT state,attempts FROM items WHERE image_id='a'")[0]
    assert row == ("permanent_error", 2)


def test_cache_lock_rejects_second_writer(tmp_path: Path):
    first = LedgerLock(tmp_path / "cache.lock")
    second = LedgerLock(tmp_path / "cache.lock")
    first.acquire()
    try:
        with pytest.raises(RuntimeError):
            second.acquire()
    finally:
        first.release(); second.release()


def test_shard_is_atomic_and_corruption_is_detected(tmp_path: Path):
    np = pytest.importorskip("numpy")
    from image_aesthetics.embeddings.shards import read_shard, verify_shard, write_atomic
    path = tmp_path / "shard-00000.safetensors"
    values = np.ones((2, 768), dtype=np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    checksum, _ = write_atomic(path, values, [0, 1], ["a", "b"])
    assert read_shard(path)[1]["sha256"] == checksum
    assert verify_shard(path)["row_count"] == 2
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        verify_shard(path)
