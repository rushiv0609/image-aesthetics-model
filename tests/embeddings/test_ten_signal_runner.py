from pathlib import Path
import errno
import os
import sqlite3
import threading
import time

import numpy as np
import pytest

from image_aesthetics.embeddings.feature_contract import DIMENSIONS, FEATURES
from image_aesthetics.embeddings.preprocess import ManifestItem
from image_aesthetics.embeddings.ten_signal_runner import (
    ProviderUnavailable, RunnerConfig, SignalProvider, TenSignalRunner, source_fingerprints_sha256,
)
from image_aesthetics.embeddings.metal_lock import metal_owner
from image_aesthetics.embeddings.feature_contract import cache_identity
from image_aesthetics.embeddings.shards import read_shard, write_atomic


def contract():
    return {"features": {name: {"dimension": DIMENSIONS.get(name, 1), "dtype": "float32",
        "range": [-1, 1] if DIMENSIONS.get(name) else [0, 1], "direction": "higher_is_better",
        "model_revision": "fixture", "checkpoint_sha256": "0" * 64,
        "preprocessing": {"image_size": 8}, "normalization": "fixture"} for name in FEATURES}}


def items(tmp_path, count=5):
    result = []
    for i in range(count):
        path = tmp_path / f"{i}.jpg"; path.write_bytes(bytes([i % 256]))
        stat = path.stat(); result.append(ManifestItem(str(i), str(path), i, stat.st_size, stat.st_mtime_ns))
    return result


def providers(*, fail=None, calls=None):
    factories = {}
    for name in FEATURES:
        dim = DIMENSIONS.get(name, 1)
        def factory(name=name, dim=dim):
            class Fake:
                dimension = dim
                def __init__(self): self.name = name
                def extract(self, images):
                    if calls is not None: calls.append((name, len(images)))
                    if fail == name: raise RuntimeError("fixture provider failure")
                    if name in ("siglip_embedding", "clip_embedding"):
                        value = np.zeros((len(images), dim), dtype=np.float32); value[:, 0] = 1.0
                        return value
                    return np.full((len(images), dim), 0.25, dtype=np.float32)
            return Fake()
        factories[name] = factory
    return factories


def prepare(path, size):
    return np.full((size, size, 3), path.stat().st_size, dtype=np.uint8), "fake"


def test_runner_is_ordered_resumable_and_accounts_providers(tmp_path: Path):
    calls = []
    cfg = RunnerConfig(tmp_path / "cache", batch_size=2, decode_workers=2, shard_size=3,
                       progress_interval_seconds=0.01)
    runner = TenSignalRunner(cfg, providers(calls=calls), prepare_fn=prepare)
    manifest = items(tmp_path)
    first = runner.run(manifest, contract(), run_id="first")
    assert first["state"] == "completed" and first["completed_images"] == 5
    assert first["provider_images"]["nima"] == 5
    assert first["decode_seconds"] > 0
    assert "resident_memory_gb" in first
    shard, metadata = read_shard(cfg.cache_dir / "shard-00000.safetensors")
    assert shard.shape[1] == sum(DIMENSIONS.get(name, 1) for name in FEATURES)
    assert metadata["signal_names"] == list(FEATURES)
    second = TenSignalRunner(cfg, providers(calls=calls), prepare_fn=prepare).run(manifest, contract(), run_id="second")
    assert second["completed_images"] == 5
    assert len(calls) == 30  # three ordered batches; providers are not called on resume


def test_provider_failure_is_explicit_and_not_fabricated(tmp_path: Path):
    cfg = RunnerConfig(tmp_path / "cache", batch_size=2, progress_interval_seconds=0.01)
    runner = TenSignalRunner(cfg, providers(fail="nima"), prepare_fn=prepare)
    result = runner.run(items(tmp_path, 2), contract(), run_id="failure")
    assert result["state"] == "incomplete"
    assert result["failed_images"] == 2
    assert result["provider_failures"]["nima"] == 2
    assert not list((tmp_path / "cache").glob("shard-*.safetensors"))


def test_unavailable_provider_fails_closed(tmp_path: Path):
    factories = providers()
    factories["clip_embedding"] = lambda: (_ for _ in ()).throw(ProviderUnavailable("checkpoint missing"))
    cfg = RunnerConfig(tmp_path / "cache", progress_interval_seconds=0.01)
    with pytest.raises(ProviderUnavailable):
        TenSignalRunner(cfg, factories, prepare_fn=prepare).run(items(tmp_path, 1), contract(), run_id="unavailable")


def test_provider_lifecycle_is_serial_and_decoder_cleanup_is_bounded(tmp_path: Path):
    active = {"value": 0, "max": 0}
    constructions = {}
    factories = {}
    for name in FEATURES:
        dim = DIMENSIONS.get(name, 1)
        def factory(name=name, dim=dim):
            class Fake:
                dimension = dim
                def __init__(self):
                    self.name = name
                    constructions[name] = constructions.get(name, 0) + 1
                def extract(self, images):
                    active["value"] += 1; active["max"] = max(active["max"], active["value"])
                    try:
                        if name in ("siglip_embedding", "clip_embedding"):
                            value=np.zeros((len(images),dim),dtype=np.float32); value[:,0]=1.; return value
                        return np.zeros((len(images), dim), dtype=np.float32)
                    finally: active["value"] -= 1
            return Fake()
        factories[name] = factory
    cfg = RunnerConfig(tmp_path / "cache", batch_size=1, decode_workers=4,
                       prepared_queue_batches=1, shard_size=2, progress_interval_seconds=0.01)
    result = TenSignalRunner(cfg, factories, prepare_fn=prepare).run(items(tmp_path, 12), contract(), run_id="serial")
    assert result["state"] == "completed"
    assert active["max"] == 1
    assert set(constructions.values()) == {1}


def test_decode_permanent_and_retryable_states_resume_correctly(tmp_path: Path):
    manifest = items(tmp_path, 2)
    attempts = {"1": 0}
    def flaky_prepare(path, size):
        if path.stem == "0":
            raise FileNotFoundError("gone")
        attempts["1"] += 1
        if attempts["1"] == 1:
            raise RuntimeError("transient decoder fault")
        return prepare(path, size)
    cfg = RunnerConfig(tmp_path / "cache", batch_size=2, progress_interval_seconds=0.01)
    first = TenSignalRunner(cfg, providers(), prepare_fn=flaky_prepare).run(manifest, contract(), run_id="faults")
    rows = sqlite3.connect(cfg.cache_dir / "work.sqlite3").execute(
        "SELECT image_id,state FROM items ORDER BY image_id").fetchall()
    assert rows == [("0", "permanent_error"), ("1", "retryable_error")]
    assert first["failed_images"] == 2
    second = TenSignalRunner(cfg, providers(), prepare_fn=flaky_prepare).run(manifest, contract(), run_id="resume")
    assert second["completed_images"] == 1
    rows = sqlite3.connect(cfg.cache_dir / "work.sqlite3").execute(
        "SELECT image_id,state FROM items ORDER BY image_id").fetchall()
    assert rows == [("0", "permanent_error"), ("1", "completed")]


def test_decode_errno_classification_persists_across_resume(tmp_path: Path):
    manifest = items(tmp_path, 2)
    transient_ready = {"value": False}

    def classify_prepare(path, size):
        if path.stem == "0" and not transient_ready["value"]:
            raise OSError(errno.EAGAIN, "temporarily unavailable")
        if path.stem == "1":
            raise OSError(errno.EACCES, "permission denied")
        return prepare(path, size)

    cfg = RunnerConfig(tmp_path / "cache", batch_size=2, progress_interval_seconds=0.01)
    TenSignalRunner(cfg, providers(), prepare_fn=classify_prepare).run(manifest, contract(), run_id="errno-first")
    rows = sqlite3.connect(cfg.cache_dir / "work.sqlite3").execute(
        "SELECT image_id,state,error_type FROM items ORDER BY image_id").fetchall()
    assert rows == [("0", "retryable_error", "BlockingIOError"), ("1", "permanent_error", "PermissionError")]
    transient_ready["value"] = True
    resumed = TenSignalRunner(cfg, providers(), prepare_fn=classify_prepare).run(manifest, contract(), run_id="errno-resume")
    assert resumed["completed_images"] == 1 and resumed["failed_images"] == 1


def test_valid_orphan_shard_is_adopted_without_recomputing(tmp_path: Path):
    manifest = items(tmp_path, 1)
    cfg = RunnerConfig(tmp_path / "cache", progress_interval_seconds=0.01)
    cfg.cache_dir.mkdir()
    dimension = sum(DIMENSIONS.get(name, 1) for name in FEATURES)
    vector = np.zeros((1, dimension), dtype=np.float32)
    vector[:, 0] = 1.0; vector[:, 768] = 1.0
    identity, _ = TenSignalRunner._cache_identity(TenSignalRunner._fingerprint_sources(manifest), contract(), None)
    write_atomic(cfg.cache_dir / "shard-00007.safetensors", vector, [0], ["0"],
                     metadata_extra={"unit_norm": False, "signal_names": list(FEATURES),
                                     "signal_dimensions": {name: DIMENSIONS.get(name, 1) for name in FEATURES},
                                     "signal_specs": contract()["features"], "cache_identity": identity,
                                     "source_fingerprints_sha256": source_fingerprints_sha256(TenSignalRunner._fingerprint_sources(manifest)),
                                     "manifest_sha256": TenSignalRunner._cache_identity(TenSignalRunner._fingerprint_sources(manifest), contract(), None)[1],
                                     "ordered_ids_sha256": __import__("hashlib").sha256(b'["0"]').hexdigest(),
                                     "item_count": "1"})
    calls = []
    result = TenSignalRunner(cfg, providers(calls=calls), prepare_fn=prepare).run(manifest, contract(), run_id="adopt")
    assert result["completed_images"] == 1 and calls == []
    row = sqlite3.connect(cfg.cache_dir / "work.sqlite3").execute(
        "SELECT state,shard_id,shard_row FROM items WHERE image_id='0'").fetchone()
    assert row == ("completed", 7, 0)


def test_writer_failure_cancels_decoder_beyond_queue_capacity(tmp_path: Path, monkeypatch):
    manifest = items(tmp_path, 20)
    cfg = RunnerConfig(tmp_path / "cache", batch_size=2, decode_workers=4,
                       prepared_queue_batches=1, shard_size=1, progress_interval_seconds=0.01)
    def fail_write(*args, **kwargs):
        raise RuntimeError("injected shard write failure")
    monkeypatch.setattr("image_aesthetics.embeddings.ten_signal_runner.write_atomic", fail_write)
    with pytest.raises(RuntimeError, match="injected shard write failure"):
        TenSignalRunner(cfg, providers(), prepare_fn=prepare).run(manifest, contract(), run_id="writer-fail")
    deadline = time.monotonic() + 2
    while any(t.is_alive() and t.name.startswith("ten-signal-decode") for t in threading.enumerate()) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not [t.name for t in threading.enumerate()
                if t.is_alive() and t.name.startswith("ten-signal-decode")]


def test_ledger_writer_crash_unblocks_saturated_decoder(tmp_path: Path, monkeypatch):
    manifest = items(tmp_path, 20)
    cfg = RunnerConfig(tmp_path / "cache", batch_size=1, decode_workers=4,
                       prepared_queue_batches=1, progress_interval_seconds=0.01)

    def crash_writer(*args, **kwargs):
        raise RuntimeError("injected ledger writer crash")

    monkeypatch.setattr("image_aesthetics.embeddings.ledger.Ledger._apply_batch", crash_writer)
    with pytest.raises(RuntimeError, match="ledger writer failed"):
        TenSignalRunner(cfg, providers(), prepare_fn=prepare).run(manifest, contract(), run_id="ledger-crash")
    deadline = time.monotonic() + 2
    while any(t.is_alive() and t.name.startswith("ten-signal-decode") for t in threading.enumerate()) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not [t.name for t in threading.enumerate()
                if t.is_alive() and t.name.startswith("ten-signal-decode")]


def test_composite_provider_shares_one_batch_and_packs_declared_subset(tmp_path: Path):
    names = ("siglip_embedding", "a1_siglip", "sharpness", "contrast", "colorfulness")
    calls = []
    class Composite:
        names = ("siglip_embedding", "a1_siglip", "sharpness", "contrast", "colorfulness")
        def extract(self, images):
            calls.append(len(images))
            self.last_timings = {name: 0.001 for name in self.names}
            values = {name: np.full((len(images), DIMENSIONS.get(name, 1)), 0.25, dtype=np.float32)
                      for name in self.names}
            values["siglip_embedding"][:] = 0.0
            values["siglip_embedding"][:, 0] = 1.0
            return values
    cfg = RunnerConfig(tmp_path / "cache", batch_size=2, shard_size=3, progress_interval_seconds=0.01)
    result = TenSignalRunner(cfg, batch_extractor=Composite(), prepare_fn=prepare).run(items(tmp_path, 5), contract(), run_id="composite")
    assert result["completed_images"] == 5 and calls == [2, 2, 1]
    assert set(result["provider_seconds"]) == set(names)
    assert all(value >= 0.003 for value in result["provider_seconds"].values())
    shard, metadata = read_shard(cfg.cache_dir / "shard-00000.safetensors")
    assert metadata["signal_names"] == list(names)
    assert shard.shape[1] == sum(DIMENSIONS.get(name, 1) for name in names)


@pytest.mark.parametrize("bad_timing,missing", [(True, False), (False, True)])
def test_composite_provider_rejects_bad_timing_or_missing_signal(tmp_path: Path, bad_timing, missing):
    class Composite:
        names = ("sharpness", "contrast")
        def extract(self, images):
            self.last_timings = {"sharpness": -0.1} if bad_timing else {"sharpness": 0.1, "contrast": 0.1}
            values = {"sharpness": np.zeros((len(images), 1), dtype=np.float32),
                      "contrast": np.zeros((len(images), 1), dtype=np.float32)}
            if missing: values.pop("contrast")
            return values
    cfg = RunnerConfig(tmp_path / "cache", progress_interval_seconds=0.01)
    with pytest.raises(ValueError, match="negative timing|omitted declared signal"):
        TenSignalRunner(cfg, batch_extractor=Composite(), prepare_fn=prepare).run(items(tmp_path, 1), contract(), run_id="bad-composite")


def test_cache_identity_rejects_contract_and_exact_manifest_changes(tmp_path):
    manifest = items(tmp_path, 1)
    cfg = RunnerConfig(tmp_path / "cache", batch_size=1, progress_interval_seconds=0.01)
    TenSignalRunner(cfg, providers(), prepare_fn=prepare).run(manifest, contract(), run_id="identity")
    changed = contract()
    changed["features"]["nima"]["checkpoint_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="cache identity differs"):
        TenSignalRunner(cfg, providers(), prepare_fn=prepare).run(manifest, changed, run_id="changed-checkpoint")
    original_hash = "2" * 64
    payload = {"schema_version": "ten-signal-cache-v1", "input_manifest_sha256": original_hash,
               "contract": contract(), "cache_identity": cache_identity(original_hash, contract())}
    fresh = tmp_path / "other-cache"
    TenSignalRunner(RunnerConfig(fresh), providers(), prepare_fn=prepare).run(manifest, payload, run_id="exact-manifest")
    other_hash = "3" * 64
    changed_payload = {**payload, "input_manifest_sha256": other_hash,
                       "cache_identity": cache_identity(other_hash, contract())}
    with pytest.raises(ValueError, match="cache identity differs"):
        TenSignalRunner(RunnerConfig(fresh), providers(), prepare_fn=prepare).run(manifest, changed_payload, run_id="manifest-reordered")


def test_source_bytes_are_fingerprinted_and_checked_on_resume(tmp_path):
    import hashlib
    path = tmp_path / "source.jpg"; path.write_bytes(b"original")
    stat = path.stat()
    item = ManifestItem("source", str(path), 0, stat.st_size, stat.st_mtime_ns,
                        hashlib.sha256(b"original").hexdigest())
    cfg = RunnerConfig(tmp_path / "cache", batch_size=1)
    first = TenSignalRunner(cfg, providers(), prepare_fn=prepare).run([item], contract(), run_id="source-first")
    assert first["state"] == "completed"
    path.write_bytes(b"mutated!")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    # Use a fresh cache identity tied to the old manifest values; decoder must
    # notice same-size, restored-mtime content changes.
    cfg2 = RunnerConfig(tmp_path / "cache2", batch_size=1)
    with pytest.raises(ValueError, match="source fingerprint changed"):
        TenSignalRunner(cfg2, providers(), prepare_fn=prepare).run([item], contract(), run_id="source-mutated")


def test_slow_first_decode_keeps_thousand_image_reorder_bounded(tmp_path):
    manifest = items(tmp_path, 1000)
    def slow_first(path, size):
        if path.stem == "0": time.sleep(0.05)
        return np.zeros((size, size, 3), dtype=np.uint8), "fixture"
    cfg = RunnerConfig(tmp_path / "cache", batch_size=32, decode_workers=8,
                       prepared_queue_batches=1, image_size=4, shard_size=128,
                       progress_interval_seconds=0.05)
    result = TenSignalRunner(cfg, providers(), prepare_fn=slow_first).run(manifest, contract(), run_id="slow-first")
    assert result["completed_images"] == 1000
    assert result["max_reorder_depth"] <= 32 * 2 + 8
    assert result["max_decode_ahead"] <= 32 * 2 + 8


def test_machine_wide_metal_owner_serializes_threads():
    active = {"value": 0, "maximum": 0}
    guard = threading.Lock()
    def occupy():
        with metal_owner():
            with guard:
                active["value"] += 1
                active["maximum"] = max(active["maximum"], active["value"])
            time.sleep(0.02)
            with guard: active["value"] -= 1
    workers = [threading.Thread(target=occupy) for _ in range(2)]
    for worker in workers: worker.start()
    for worker in workers: worker.join()
    assert active["maximum"] == 1


def test_machine_wide_metal_lease_serializes_process_lifetimes(tmp_path):
    import os, subprocess, sys
    root=Path(__file__).parents[2]
    environment=os.environ.copy()
    environment["PYTHONPATH"]=str(root/"src")
    environment["IMAGE_AESTHETICS_METAL_LOCK"]=str(tmp_path/"host-metal.lock")
    hold="import time; from image_aesthetics.embeddings.metal_lock import metal_owner; \nwith metal_owner(): print('held',flush=True); time.sleep(.5)"
    enter="from image_aesthetics.embeddings.metal_lock import metal_owner; \nwith metal_owner(): print('entered',flush=True)"
    first=subprocess.Popen([sys.executable,"-c",hold],cwd=root,env=environment,stdout=subprocess.PIPE,text=True)
    try:
        assert first.stdout.readline().strip()=="held"
        second=subprocess.Popen([sys.executable,"-c",enter],cwd=root,env=environment,stdout=subprocess.PIPE,text=True)
        time.sleep(.1)
        assert second.poll() is None, "second process entered while first retained its Metal lease"
        assert first.wait(timeout=5)==0
        assert second.communicate(timeout=5)[0].strip()=="entered"
    finally:
        if first.poll() is None: first.kill(); first.wait()
