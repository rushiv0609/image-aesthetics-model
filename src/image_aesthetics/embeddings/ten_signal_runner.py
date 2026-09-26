"""Dataset-neutral bounded runner for the fixed ten-signal cache.

The runner deliberately owns no model implementations.  Providers are injected
by the application (and by tests), are constructed once on the inference
thread, and are called serially.  This is the boundary that prevents multiple
Metal owners from being created accidentally.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import os
import queue
import errno
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import replace
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, runtime_checkable

from .feature_contract import FEATURES, load_contract, validate_features
from .ledger import Ledger, LedgerLock, utc_now
from .metal_lock import metal_owner
from .preprocess import ManifestItem, prepare_image
from .shards import read_shard, verify_shard, write_atomic


def _np():
    import numpy as np
    return np


class ProviderUnavailable(RuntimeError):
    """Raised when a production provider/checkpoint is not available locally."""


@runtime_checkable
class SignalProvider(Protocol):
    name: str
    dimension: int

    def extract(self, images: Any) -> Any: ...


@dataclass(frozen=True)
class RunnerConfig:
    cache_dir: Path
    batch_size: int = 16
    decode_workers: int = 4
    prepared_queue_batches: int = 2
    shard_size: int = 4096
    max_attempts: int = 3
    retry_permanent: bool = False
    image_size: int = 224
    progress_interval_seconds: float = 5.0
    cache_schema_version: int = 1


@dataclass
class _Decoded:
    item: ManifestItem
    image: Any = None
    decoder: str | None = None
    error: BaseException | None = None
    elapsed: float = 0.0


@dataclass
class _Progress:
    run_id: str
    total_images: int
    state: str = "running"
    phase: str = "decode"
    completed_images: int = 0
    failed_images: int = 0
    decoded_images: int = 0
    inference_batches: int = 0
    provider_seconds: dict[str, float] = field(default_factory=dict)
    provider_images: dict[str, int] = field(default_factory=dict)
    provider_failures: dict[str, int] = field(default_factory=dict)
    provider_memory_gb: dict[str, float] = field(default_factory=dict)
    decode_seconds: float = 0.0
    resident_memory_gb: float = 0.0
    swap_used_gb: float = 0.0
    swap_baseline_gb: float = 0.0
    swap_delta_gb: float = 0.0
    swap_scope: str = "host_wide"
    reorder_depth: int = 0
    max_reorder_depth: int = 0
    decode_backpressure_seconds: float = 0.0
    max_decode_ahead: int = 0
    last_error: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": 1, "run_id": self.run_id, "state": self.state,
            "phase": self.phase, "total_images": self.total_images,
            "completed_images": self.completed_images, "failed_images": self.failed_images,
            "decoded_images": self.decoded_images, "inference_batches": self.inference_batches,
            "provider_seconds": dict(sorted(self.provider_seconds.items())),
            "provider_images": dict(sorted(self.provider_images.items())),
            "provider_failures": dict(sorted(self.provider_failures.items())),
            "provider_memory_gb": dict(sorted(self.provider_memory_gb.items())),
            "decode_seconds": self.decode_seconds,
            "resident_memory_gb": self.resident_memory_gb,
            "swap_used_gb": self.swap_used_gb,
            "swap_baseline_gb": self.swap_baseline_gb,
            "swap_delta_gb": self.swap_delta_gb,
            "swap_scope": self.swap_scope,
            "reorder_depth": self.reorder_depth,
            "max_reorder_depth": self.max_reorder_depth,
            "decode_backpressure_seconds": self.decode_backpressure_seconds,
            "max_decode_ahead": self.max_decode_ahead,
            "updated_at": utc_now(), "last_error": self.last_error,
        }


def _write_progress(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    except OSError:
        pass


def _queue_put(target: queue.Queue, value: Any, cancel: threading.Event) -> bool:
    while True:
        try:
            target.put(value, timeout=0.1)
            return True
        except queue.Full:
            if cancel.is_set():
                return False


def _decode_error_is_permanent(error: BaseException) -> bool:
    """Classify stable bad-input failures without treating resource I/O as bad data."""
    if isinstance(error, (FileNotFoundError, IsADirectoryError, PermissionError, ValueError)):
        return True
    if isinstance(error, OSError):
        # Interrupted reads, temporary resource exhaustion, timeouts, and busy
        # files are retried; other local filesystem errors describe an input
        # that cannot be decoded by this cache version.
        return error.errno not in {errno.EAGAIN, errno.EBUSY, errno.EINTR, errno.EMFILE,
                                   errno.ENFILE, errno.ETIMEDOUT}
    return False


def _provider_error_is_permanent(error: BaseException) -> bool:
    if isinstance(error, (ValueError, TypeError, AssertionError)):
        return True
    if isinstance(error, OSError):
        return _decode_error_is_permanent(error)
    # Runtime/device/resource faults are transient and may succeed on resume.
    return False


def _validate_provider_output(name: str, provider: SignalProvider, value: Any, count: int, dimension: int):
    np = _np()
    output = np.asarray(value)
    if output.dtype != np.dtype("float32"):
        raise ValueError(f"provider {name} returned {output.dtype}, expected float32")
    if output.shape != (count, dimension):
        raise ValueError(f"provider {name} returned {output.shape}, expected {(count, dimension)}")
    if not np.isfinite(output).all():
        raise ValueError(f"provider {name} returned non-finite values")
    return output


def _memory_gb() -> float:
    try:
        if Path("/proc/self/statm").exists():
            pages = int(Path("/proc/self/statm").read_text().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE") / (1024 ** 3)
        import subprocess
        result = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, timeout=1)
        return int(result.stdout.strip()) / (1024 ** 2)
    except (ImportError, AttributeError, OSError, ValueError):
        return 0.0


def _swap_gb() -> float:
    try:
        import subprocess
        result = subprocess.run(["/usr/sbin/sysctl", "vm.swapusage"], capture_output=True, text=True, timeout=1)
        import re
        match = re.search(r"used = ([0-9.]+)([MG])", result.stdout)
        if not match: return 0.0
        value = float(match.group(1))
        return value / 1024 if match.group(2) == "M" else value
    except Exception:
        return 0.0


def canonical_items_sha256(items: Iterable[ManifestItem]) -> str:
    rows = [{"image_id": item.image_id, "manifest_position": item.manifest_position,
             "source_path": item.source_path, "source_size": item.source_size,
             "source_mtime_ns": item.source_mtime_ns,
             "source_fingerprint": item.source_fingerprint}
            for item in sorted(items, key=lambda item: item.manifest_position)]
    # Match the pre-existing cache-identity serialization byte-for-byte
    # (json.dumps' default ensure_ascii=True) for backward verification.
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_fingerprints_sha256(items: Iterable[ManifestItem]) -> str:
    ordered = sorted(items, key=lambda item: item.manifest_position)
    return hashlib.sha256(json.dumps(
        [[item.image_id, item.manifest_position, item.source_fingerprint] for item in ordered],
        separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _validate_signal_values(name: str, values: Any, spec: dict[str, Any]) -> None:
    np = _np()
    if np.asarray(values).dtype != np.float32:
        raise ValueError(f"{name} provider output must be float32")
    low, high = map(float, spec["range"])
    if np.any(values < low) or np.any(values > high):
        raise ValueError(f"{name} output outside declared range [{low}, {high}]")
    if name in ("siglip_embedding", "clip_embedding"):
        norms = np.linalg.norm(values, axis=1)
        if not np.all(np.isfinite(norms)) or np.any(norms < 1e-6) or not np.allclose(norms, 1.0, atol=5e-3):
            raise ValueError(f"{name} embeddings must have nonzero unit norm")


def _extract_one(provider: SignalProvider, name: str, images: Any, dimension: int) -> Any:
    """Run the stage-owned provider on the sole inference thread."""
    with metal_owner():
        if getattr(provider, "name", name) != name:
            raise ValueError(f"provider name mismatch: {name}")
        return _validate_provider_output(name, provider, provider.extract(images), len(images), dimension)


def _check_provider(factory: Callable[[], SignalProvider], name: str, dimension: int) -> None:
    with metal_owner():
        provider = factory()
        try:
            if getattr(provider, "name", name) != name:
                raise ValueError(f"provider name mismatch: {name}")
            if int(getattr(provider, "dimension", dimension)) != dimension:
                raise ValueError(f"provider {name} dimension mismatch")
        finally:
            del provider


class TenSignalRunner:
    """Run fixed providers with bounded decode, ordered batches and resumable shards."""

    def __init__(self, config: RunnerConfig,
                 provider_factories: Mapping[str, Callable[[], SignalProvider]] | None = None, *,
                 batch_extractor: Any | None = None,
                 prepare_fn: Callable[[Path, int], tuple[Any, str]] = prepare_image):
        self.config = config
        self.provider_factories = dict(provider_factories or {})
        self.batch_extractor = batch_extractor
        self.feature_names = tuple(getattr(batch_extractor, "names", FEATURES)) if batch_extractor else FEATURES
        if not self.feature_names or any(name not in FEATURES for name in self.feature_names):
            raise ValueError("batch extractor must declare a non-empty subset of fixed signals")
        self.prepare_fn = prepare_fn
        self.cancel = threading.Event()
        self._progress: _Progress | None = None
        self._providers: dict[str, SignalProvider] = {}
        self._source_fingerprints_sha256 = ""

    def run(self, manifest: Iterable[ManifestItem], contract: dict[str, Any] | Path,
            *, run_id: str | None = None) -> dict[str, Any]:
        # A runner instance may be reused after a failed/cancelled run.
        self.cancel.clear()
        contract_payload = None
        if isinstance(contract, Path):
            contract_payload = load_contract(contract)
            contract = contract_payload["contract"]
        elif "contract" in contract and "cache_identity" in contract:
            contract_payload = contract
            contract = contract_payload["contract"]
        validate_features(contract["features"])
        if self.batch_extractor is None and set(self.provider_factories) != set(self.feature_names):
            missing = sorted(set(FEATURES) - set(self.provider_factories))
            extra = sorted(set(self.provider_factories) - set(FEATURES))
            raise ValueError(f"provider factories must exactly cover ten signals; missing={missing}, extra={extra}")
        items = sorted(list(manifest), key=lambda x: x.manifest_position)
        if len({x.manifest_position for x in items}) != len(items):
            raise ValueError("manifest contains duplicate positions")
        if len({x.image_id for x in items}) != len(items):
            raise ValueError("manifest contains duplicate image IDs")
        items = self._fingerprint_sources(items)
        self._source_fingerprints_sha256 = source_fingerprints_sha256(items)
        run_id = run_id or uuid.uuid4().hex
        cfg = self.config
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        lock = LedgerLock(cfg.cache_dir / "cache.lock")
        lock.acquire()
        ledger = Ledger(cfg.cache_dir / "work.sqlite3")
        progress = _Progress(run_id, len(items))
        self._progress = progress
        progress_thread: threading.Thread | None = None
        error: str | None = None
        try:
            ledger.open()
            identity, manifest_hash = self._cache_identity(items, contract, contract_payload)
            if contract_payload and contract_payload.get("source_fingerprints_sha256") and contract_payload["source_fingerprints_sha256"] != self._source_fingerprints_sha256:
                raise ValueError("source fingerprints differ from immutable cache contract")
            ordered_ids_hash = hashlib.sha256(json.dumps([x.image_id for x in items], separators=(",", ":")).encode()).hexdigest()
            ledger.ensure_cache_identity(identity, manifest_hash, hashlib.sha256(
                json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                len(items), ordered_ids_hash, self._source_fingerprints_sha256,
                manifest_identity_kind="bytes" if contract_payload else "canonical_items",
                manifest_bytes_sha256=(str(contract_payload["input_manifest_sha256"])
                                       if contract_payload else ""),
                canonical_items_sha256=canonical_items_sha256(items))
            ledger.seed_items([item.as_ledger_row() for item in items])
            dimensions = {name: int(contract["features"][name]["dimension"]) for name in self.feature_names}
            self._reconcile(ledger, items, dimensions, self.feature_names,
                            {name: contract["features"][name] for name in self.feature_names})
            counts = ledger.counts()
            progress.completed_images = counts.get("completed", 0)
            progress.failed_images = counts.get("permanent_error", 0) + counts.get("retryable_error", 0)
            ledger.start_run(run_id, model_revision="ten-signal-contract",
                             preprocessing_hash=hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
                             dataset_fingerprint=self._fingerprint(items), cache_schema_version=cfg.cache_schema_version,
                             config={"runner": "ten_signal", "batch_size": cfg.batch_size,
                                     "cache_identity": identity, "manifest_sha256": manifest_hash,
                                     "source_fingerprints_sha256": self._source_fingerprints_sha256})
            ledger.start_writer()
            progress.swap_baseline_gb = _swap_gb()
            progress.swap_used_gb = progress.swap_baseline_gb
            progress.swap_delta_gb = 0.0
            progress_thread = threading.Thread(target=self._progress_loop, args=(ledger,), daemon=True)
            progress_thread.start()
            pending = ledger.pending_items(cfg.max_attempts, cfg.retry_permanent)
            by_id = {item.image_id: item for item in items}
            work = [by_id[row["image_id"]] for row in pending]
            self._pipeline(work, ledger, contract, self._next_shard_id(ledger))
            if self.cancel.is_set():
                progress.state = "cancelled"
            else:
                ledger.emit("heartbeat", wait=True, run_id=run_id)
                counts = ledger.counts()
                unfinished = any(counts.get(state, 0) for state in
                                 ("pending", "decoding", "decoded", "embedding", "staged", "retryable_error", "permanent_error"))
                progress.state = "completed" if counts.get("completed", 0) == len(items) and not unfinished else "incomplete"
        except BaseException as exc:
            progress.state = "failed"
            progress.last_error = str(exc)[:500]
            error = progress.last_error
            raise
        finally:
            self.cancel.set()
            if progress_thread:
                progress_thread.join(timeout=10)
            if progress.state == "running":
                progress.state = "incomplete"
            progress.resident_memory_gb = _memory_gb()
            progress.swap_used_gb = _swap_gb()
            progress.swap_delta_gb = progress.swap_used_gb - progress.swap_baseline_gb
            try:
                if ledger._thread:
                    ledger.stop(progress.state, error)
                # ``stop(..., wait=True)`` is a FIFO durability barrier, so
                # these counts describe the persisted cache, including a
                # resumed run.
                counts = ledger.counts()
                progress.completed_images = counts.get("completed", 0)
                progress.failed_images = counts.get("permanent_error", 0) + counts.get("retryable_error", 0)
                _write_progress(cfg.cache_dir / "progress.json", progress.snapshot())
            finally:
                lock.release()
        return progress.snapshot()

    @classmethod
    def _fingerprint_sources(cls, items: list[ManifestItem]) -> list[ManifestItem]:
        """Bind cache identity to bytes currently at every available source path.

        This runs before ledger reconciliation, so a completed row cannot hide
        a source mutation merely because it would not otherwise be decoded.
        """
        normalized = []
        for item in items:
            path = Path(item.source_path)
            if not path.is_file():
                normalized.append(item)
                continue
            stat = path.stat()
            if item.source_size and item.source_size != stat.st_size:
                raise ValueError(f"source size changed for {item.image_id}")
            if item.source_mtime_ns and item.source_mtime_ns != stat.st_mtime_ns:
                raise ValueError(f"source mtime changed for {item.image_id}")
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            current = digest.hexdigest()
            if item.source_fingerprint and item.source_fingerprint != current:
                raise ValueError(f"source fingerprint changed for {item.image_id}")
            normalized.append(replace(item, source_size=stat.st_size,
                                      source_mtime_ns=stat.st_mtime_ns,
                                      source_fingerprint=current))
        return normalized

    @classmethod
    def _cache_identity(cls, items: list[ManifestItem], contract: dict[str, Any], payload: dict[str, Any] | None):
        if payload is not None:
            manifest_hash = str(payload["input_manifest_sha256"])
            identity = str(payload["cache_identity"])
            return identity, manifest_hash
        manifest_hash = canonical_items_sha256(items)
        identity = hashlib.sha256(json.dumps({"schema": "ten-signal-cache-v1", "manifest_sha256": manifest_hash,
            "contract": contract}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return identity, manifest_hash

    @staticmethod
    def _fingerprint(items: list[ManifestItem]) -> str:
        digest = hashlib.sha256()
        for item in items:
            digest.update(f"{item.image_id}\0{item.manifest_position}\0{item.source_size}\0{item.source_mtime_ns}\n".encode())
        return digest.hexdigest()

    def _progress_loop(self, ledger: Ledger) -> None:
        while not self.cancel.wait(self.config.progress_interval_seconds):
            if self._progress:
                self._progress.resident_memory_gb = _memory_gb()
                self._progress.swap_used_gb = _swap_gb()
                self._progress.swap_delta_gb = (
                    self._progress.swap_used_gb - self._progress.swap_baseline_gb)
                _write_progress(self.config.cache_dir / "progress.json", self._progress.snapshot())
                ledger.emit("heartbeat", run_id=self._progress.run_id)

    def _next_shard_id(self, ledger: Ledger) -> int:
        """Never overwrite a final shard left behind by an interrupted run."""
        ids = [ledger.next_shard_id()]
        for path in self.config.cache_dir.glob("shard-*.safetensors"):
            match = re.fullmatch(r"shard-(\d+)\.safetensors", path.name)
            if match:
                ids.append(int(match.group(1)) + 1)
        return max(ids)

    def _reconcile(self, ledger: Ledger, items: list[ManifestItem], dimensions: dict[str, int],
                   feature_names: tuple[str, ...], feature_specs: dict[str, Any]) -> None:
        """Recover interrupted work before the asynchronous writer starts.

        A final shard without its SQLite commit is adopted only when all of its
        coordinates belong to this exact manifest.  Everything else is
        quarantined, which preserves the evidence while ensuring a retry cannot
        collide with a stale filename.
        """
        by_position = {item.manifest_position: item for item in items}
        conn = sqlite3.connect(ledger.path, timeout=5)
        Ledger._configure(conn)
        try:
            conn.execute("BEGIN")
            conn.execute("UPDATE items SET state='pending',attempts=0,shard_id=NULL,shard_row=NULL,embedding_checksum=NULL,error_type=NULL,error_message=NULL,finished_at=NULL,updated_at=? WHERE state IN ('decoding','decoded','embedding','staged')", (utc_now(),))
            conn.commit()
        finally:
            conn.close()

        expected_dimension = sum(dimensions.values())
        cache_identity = ledger.cache_identity()
        cache_metadata = ledger.cache_metadata()
        committed = {row["shard_id"]: row for row in ledger.committed_shards()}
        # A database row is not proof of a shard: its file and metadata must
        # still validate before its completed rows may be retained.
        for shard_id, record in list(committed.items()):
            path = Path(record["path"])
            try:
                metadata = verify_shard(path, expected_dimension)
                if (metadata.get("sha256") != record["sha256"] or
                        int(metadata.get("row_count", -1)) != int(record["row_count"]) or
                        metadata.get("signal_names") != list(feature_names) or
                        metadata.get("signal_dimensions") != dimensions or
                        metadata.get("signal_specs") != feature_specs or
                        metadata.get("cache_identity") != cache_identity or
                        metadata.get("source_fingerprints_sha256") != self._source_fingerprints_sha256 or
                        metadata.get("manifest_sha256") != cache_metadata.get("manifest_sha256") or
                        metadata.get("ordered_ids_sha256") != cache_metadata.get("ordered_ids_sha256") or
                        str(metadata.get("item_count")) != cache_metadata.get("item_count")):
                    raise ValueError("committed shard ledger/contract mismatch")
            except Exception:
                if path.exists():
                    self._quarantine_shard(path, "corrupt")
                self._reset_shard(ledger, shard_id)
                committed.pop(shard_id, None)
        for path in sorted(self.config.cache_dir.glob("shard-*.safetensors")):
            match = re.fullmatch(r"shard-(\d+)\.safetensors", path.name)
            if not match:
                continue
            shard_id = int(match.group(1))
            try:
                metadata = verify_shard(path, expected_dimension)
                if (metadata.get("signal_names") != list(feature_names) or
                        metadata.get("signal_dimensions") != dimensions or
                        metadata.get("signal_specs") != feature_specs or
                        metadata.get("cache_identity") != cache_identity or
                        metadata.get("source_fingerprints_sha256") != self._source_fingerprints_sha256 or
                        metadata.get("manifest_sha256") != cache_metadata.get("manifest_sha256") or
                        metadata.get("ordered_ids_sha256") != cache_metadata.get("ordered_ids_sha256") or
                        str(metadata.get("item_count")) != cache_metadata.get("item_count")):
                    raise ValueError("signal-name contract mismatch")
            except Exception:
                self._quarantine_shard(path, "corrupt")
                self._reset_shard(ledger, shard_id)
                committed.pop(shard_id, None)
                continue
            # The dimension vector is checked by the packed matrix dimension.
            # A known, valid shard remains authoritative.
            if shard_id in committed:
                continue
            positions = metadata["positions"]
            image_ids = metadata["image_ids"]
            valid = bool(positions) and all(
                position in by_position and by_position[position].image_id == image_id
                for position, image_id in zip(positions, image_ids)
            )
            if not valid:
                self._quarantine_shard(path, "orphan")
                continue
            embeddings, _ = read_shard(path)
            offset = 0
            for name in feature_names:
                dimension = dimensions[name]
                _validate_signal_values(name, embeddings[:, offset:offset + dimension], feature_specs[name])
                offset += dimension
            rows = [{"image_id": image_id, "position": position,
                     "checksum": hashlib.sha256(embeddings[row].tobytes()).hexdigest()}
                    for row, (position, image_id) in enumerate(zip(positions, image_ids))]
            conn = sqlite3.connect(ledger.path, timeout=5)
            Ledger._configure(conn)
            try:
                conn.execute("BEGIN")
                conflicts = conn.execute(
                    "SELECT COUNT(*) FROM items WHERE manifest_position IN (%s) AND state='completed'" % ",".join("?" * len(positions)),
                    tuple(positions),
                ).fetchone()[0]
                if conflicts:
                    raise ValueError("orphan shard overlaps a committed item")
                now = utc_now()
                conn.execute("INSERT INTO shards(shard_id,path,row_count,sha256,min_position,max_position,committed_at) VALUES(?,?,?,?,?,?,?)",
                             (shard_id, str(path), len(rows), metadata["sha256"], min(positions), max(positions), now))
                for row, record in enumerate(rows):
                    changed = conn.execute(
                        "UPDATE items SET state='completed',shard_id=?,shard_row=?,embedding_checksum=?,updated_at=?,finished_at=? WHERE image_id=? AND manifest_position=?",
                        (shard_id, row, record["checksum"], now, now, record["image_id"], record["position"]),
                    )
                    if changed.rowcount != 1:
                        raise ValueError("orphan shard item does not match ledger")
                conn.commit()
            except Exception:
                conn.rollback()
                self._quarantine_shard(path, "orphan")
            finally:
                conn.close()

    @staticmethod
    def _quarantine_shard(path: Path, reason: str) -> None:
        suffix = f".{reason}-{uuid.uuid4().hex}"
        metadata = path.with_suffix(path.suffix + ".json")
        if path.exists():
            os.replace(path, path.with_name(path.name + suffix))
        if metadata.exists():
            os.replace(metadata, metadata.with_name(metadata.name + suffix))

    @staticmethod
    def _reset_shard(ledger: Ledger, shard_id: int) -> None:
        conn = sqlite3.connect(ledger.path, timeout=5)
        Ledger._configure(conn)
        try:
            conn.execute("BEGIN")
            conn.execute("UPDATE items SET state='pending',attempts=0,shard_id=NULL,shard_row=NULL,embedding_checksum=NULL,error_type=NULL,error_message=NULL,finished_at=NULL,updated_at=? WHERE shard_id=?", (utc_now(), shard_id))
            conn.execute("DELETE FROM shards WHERE shard_id=?", (shard_id,))
            conn.commit()
        finally:
            conn.close()

    def _pipeline(self, items: list[ManifestItem], ledger: Ledger, contract: dict[str, Any], first_shard: int) -> None:
        if not items:
            return
        # Hold the host-wide lease from before the first model constructor until
        # every provider reference has been released. Per-call locking alone
        # permits another process to retain a second Metal model concurrently.
        with metal_owner():
            try:
                self._pipeline_locked(items, ledger, contract, first_shard)
            finally:
                self._providers.clear()
                import gc
                gc.collect()

    def _pipeline_locked(self, items: list[ManifestItem], ledger: Ledger, contract: dict[str, Any], first_shard: int) -> None:
        if not items:
            return
        self._preflight(contract)
        decoded: queue.Queue[_Decoded | None] = queue.Queue(maxsize=max(1, self.config.batch_size * self.config.prepared_queue_batches))
        futures: dict[Future, ManifestItem] = {}
        cancel = self.cancel
        dispatcher_errors: list[BaseException] = []
        ahead_limit = max(1, self.config.batch_size * (self.config.prepared_queue_batches + 1) + self.config.decode_workers)
        ahead_slots = threading.BoundedSemaphore(ahead_limit)
        def decode_dispatch() -> None:
            try:
                with ThreadPoolExecutor(max_workers=self.config.decode_workers,
                                        thread_name_prefix="ten-signal-decode") as pool:
                    iterator = iter(items)
                    exhausted = False
                    while (futures or not exhausted) and not cancel.is_set():
                        while not exhausted and len(futures) < self.config.decode_workers and not cancel.is_set():
                            blocked_since = time.perf_counter()
                            acquired = ahead_slots.acquire(timeout=0.05)
                            waited = time.perf_counter() - blocked_since
                            if not acquired or waited > 0.05:
                                self._progress.decode_backpressure_seconds += waited
                            if not acquired:
                                break
                            try: item = next(iterator)
                            except StopIteration:
                                ahead_slots.release(); exhausted = True
                            else:
                                ledger.emit("state", image_id=item.image_id, state="decoding")
                                futures[pool.submit(self._decode, item)] = item
                                self._progress.max_decode_ahead = max(self._progress.max_decode_ahead,
                                    len(futures) + decoded.qsize())
                        if not futures:
                            continue
                        done, _ = wait(futures, return_when=FIRST_COMPLETED)
                        for future in done:
                            item = futures.pop(future)
                            try: result = future.result()
                            except BaseException as exc: result = _Decoded(item, error=exc, elapsed=0)
                            if not _queue_put(decoded, result, cancel):
                                ahead_slots.release()
                                return
            except BaseException as exc:
                self._progress.last_error = str(exc)[:500]
                dispatcher_errors.append(exc)
                cancel.set()
            finally:
                # The consumer also observes ``cancel`` with a bounded get,
                # so a saturated queue cannot prevent shutdown notification.
                with __import__("contextlib").suppress(queue.Full):
                    decoded.put_nowait(None)

        thread = threading.Thread(target=decode_dispatch, name="ten-signal-decode-dispatcher", daemon=True)
        thread.start()
        inference_failed = False
        try:
            self._infer_and_write(items, decoded, ledger, contract, first_shard, dispatcher_errors, ahead_slots)
        except BaseException:
            inference_failed = True
            raise
        finally:
            # An inference/provider failure must unblock a bounded decoder.
            self.cancel.set()
            thread.join(timeout=30)
            self._providers.clear()
            if thread.is_alive():
                raise RuntimeError("decoder thread did not stop after cancellation")
            if not inference_failed:
                self.cancel.clear()

    def _preflight(self, contract: dict[str, Any]) -> None:
        if self.batch_extractor is not None:
            return
        dimensions = {name: int(contract["features"][name]["dimension"]) for name in self.feature_names}
        self._providers.clear()
        for name in self.feature_names:
            try:
                with metal_owner():
                    provider = self.provider_factories[name]()
                if getattr(provider, "name", name) != name:
                    raise ValueError(f"provider name mismatch: {name}")
                if int(getattr(provider, "dimension", dimensions[name])) != dimensions[name]:
                    raise ValueError(f"provider {name} dimension mismatch")
                self._providers[name] = provider
            except Exception as exc:
                self._providers.clear()
                raise ProviderUnavailable(f"provider {name} unavailable: {exc}") from exc

    def _decode(self, item: ManifestItem) -> _Decoded:
        started = time.perf_counter()
        path = Path(item.source_path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        stat = path.stat()
        if stat.st_size != item.source_size or stat.st_mtime_ns != item.source_mtime_ns:
            raise ValueError(f"source changed after manifest creation: {path}")
        if item.source_fingerprint:
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024): digest.update(chunk)
            if digest.hexdigest() != item.source_fingerprint:
                raise ValueError(f"source bytes changed after manifest creation: {path}")
        image, decoder = self.prepare_fn(path, self.config.image_size)
        return _Decoded(item, image, decoder, elapsed=time.perf_counter() - started)

    def _infer_and_write(self, items: list[ManifestItem], decoded: queue.Queue, ledger: Ledger,
                         contract: dict[str, Any], first_shard: int,
                         dispatcher_errors: list[BaseException], ahead_slots: threading.BoundedSemaphore) -> None:
        np = _np(); expected = [x.manifest_position for x in items]; next_pos = 0; waiting = []
        dimensions = {name: int(contract["features"][name]["dimension"]) for name in self.feature_names}
        specs = contract["features"]
        cache_identity = ledger.cache_identity()
        try:
            # Provider instances are created and released one at a time below.
            self._progress.phase = "embedding"
            shard_id = first_shard; shard_items: list[ManifestItem] = []; shard_rows: list[Any] = []

            def flush() -> None:
                nonlocal shard_id, shard_items, shard_rows
                if not shard_items: return
                matrix = np.asarray(shard_rows, dtype=np.float32)
                path = self.config.cache_dir / f"shard-{shard_id:05d}.safetensors"
                checksum, _ = write_atomic(path, matrix,
                    [x.manifest_position for x in shard_items], [x.image_id for x in shard_items],
                    cache_schema_version=self.config.cache_schema_version,
                metadata_extra={"unit_norm": False, "signal_names": list(self.feature_names),
                                    "signal_dimensions": dimensions, "signal_specs": {name: specs[name] for name in self.feature_names},
                                    "cache_identity": cache_identity,
                                    "source_fingerprints_sha256": self._source_fingerprints_sha256,
                                    "manifest_sha256": ledger.cache_metadata().get("manifest_sha256"),
                                    "ordered_ids_sha256": ledger.cache_metadata().get("ordered_ids_sha256"),
                                    "item_count": ledger.cache_metadata().get("item_count")})
                records = [{"image_id": x.image_id, "position": x.manifest_position,
                            "checksum": hashlib.sha256(matrix[i].tobytes()).hexdigest()}
                           for i, x in enumerate(shard_items)]
                ledger.emit("commit_shard", wait=True, shard_id=shard_id, path=str(path), row_count=len(records),
                            sha256=checksum, min_position=min(r["position"] for r in records),
                            max_position=max(r["position"] for r in records), items=records)
                self._progress.completed_images += len(records)
                shard_items.clear(); shard_rows.clear(); shard_id += 1

            ended = False
            while next_pos < len(expected):
                batch = []
                # Fill a batch in manifest order.  We block for the next
                # position when decode completion arrives out of order.
                while len(batch) < self.config.batch_size and next_pos < len(expected):
                    while not waiting or waiting[0][0] != expected[next_pos]:
                        try:
                            result = decoded.get(timeout=0.1)
                        except queue.Empty:
                            if not self.cancel.is_set():
                                continue
                            if dispatcher_errors:
                                raise RuntimeError("decoder dispatcher failed") from dispatcher_errors[0]
                            ended = True
                            break
                        if result is None:
                            ended = True
                            break
                        heapq.heappush(waiting, (result.item.manifest_position, result))
                        self._progress.reorder_depth = len(waiting)
                        self._progress.max_reorder_depth = max(self._progress.max_reorder_depth, len(waiting))
                    if not waiting or waiting[0][0] != expected[next_pos]:
                        break
                    _, current = heapq.heappop(waiting); next_pos += 1
                    ahead_slots.release()
                    self._progress.reorder_depth = len(waiting)
                    batch.append(current)
                if not batch:
                    if ended: break
                    continue
                good = []
                for current in batch:
                    self._progress.decode_seconds += current.elapsed
                    self._progress.decoded_images += 1
                    if current.error:
                        transient = not _decode_error_is_permanent(current.error)
                        ledger.emit("state", image_id=current.item.image_id,
                                    state="retryable_error" if transient else "permanent_error",
                                    max_attempts=self.config.max_attempts,
                                    error_type=type(current.error).__name__, error_message=str(current.error)[:500])
                        self._progress.failed_images += 1
                    else:
                        ledger.emit("state", image_id=current.item.image_id, state="decoded", decoder=current.decoder)
                        good.append(current)
                if not good: continue
                for current in good:
                    ledger.emit("state", image_id=current.item.image_id, state="embedding")
                images = np.stack([x.image for x in good], axis=0)
                outputs = []
                if self.batch_extractor is not None:
                    started = time.perf_counter(); memory_before = _memory_gb()
                    with metal_owner():
                        values = self.batch_extractor.extract(images)
                    composite_elapsed = time.perf_counter() - started
                    stage_timings = getattr(self.batch_extractor, "last_timings", {})
                    for name in self.feature_names:
                        if name not in values:
                            raise ValueError(f"composite provider omitted declared signal {name}")
                        value = _validate_provider_output(name, self.batch_extractor, values[name], len(good), dimensions[name])
                        _validate_signal_values(name, value, specs[name])
                        elapsed = float(stage_timings.get(name, 0.0))
                        if elapsed < 0:
                            raise ValueError(f"composite provider reported negative timing for {name}")
                        # Test composites need not report timings; attribute
                        # their whole call once rather than inventing a split.
                        if not stage_timings and name == self.feature_names[0]:
                            elapsed = composite_elapsed
                        self._progress.provider_seconds[name] = self._progress.provider_seconds.get(name, 0.0) + elapsed
                        self._progress.provider_memory_gb[name] = max(self._progress.provider_memory_gb.get(name, 0.0), _memory_gb() - memory_before)
                        self._progress.provider_images[name] = self._progress.provider_images.get(name, 0) + len(good)
                        outputs.append(value)
                for name in (() if self.batch_extractor is not None else self.feature_names):
                    started = time.perf_counter(); memory_before = _memory_gb()
                    try:
                        value = _extract_one(self._providers[name], name, images, dimensions[name])
                        _validate_signal_values(name, value, specs[name])
                    except BaseException as exc:
                        self._progress.provider_failures[name] = self._progress.provider_failures.get(name, 0) + len(good)
                        for x in good:
                            state = "permanent_error" if _provider_error_is_permanent(exc) else "retryable_error"
                            ledger.emit("state", image_id=x.item.image_id, state=state,
                                        max_attempts=self.config.max_attempts if state == "retryable_error" else None,
                                        error_type=type(exc).__name__, error_message=f"{name}: {exc}"[:500])
                        self._progress.failed_images += len(good)
                        outputs = []
                        break
                    else:
                        elapsed = time.perf_counter() - started
                        self._progress.provider_seconds[name] = self._progress.provider_seconds.get(name, 0.0) + elapsed
                        self._progress.provider_memory_gb[name] = max(self._progress.provider_memory_gb.get(name, 0.0), _memory_gb() - memory_before)
                        self._progress.provider_images[name] = self._progress.provider_images.get(name, 0) + len(good)
                        outputs.append(value)
                if not outputs: continue
                packed = np.concatenate(outputs, axis=1)
                for row, x in enumerate(good):
                    ledger.emit("state", image_id=x.item.image_id, state="staged")
                    shard_items.append(x.item); shard_rows.append(packed[row])
                    if len(shard_items) >= self.config.shard_size: flush()
                self._progress.inference_batches += 1
            flush()
        except BaseException as exc:
            self._progress.last_error = str(exc)[:500]
            self.cancel.set()
            raise
