"""Resumable AVA SigLIP2 embedding cache generation.

The implementation intentionally has one owner for the MLX model and uses
threads only for native image I/O/preparation.  SQLite and committed shard
files are authoritative; ``progress.json`` is an atomic disposable view.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import queue
import signal
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .backend import SigLIP2Embedder
from .ledger import Ledger, LedgerLock, utc_now
from .preprocess import ManifestItem, prepare_image, preprocessing_contract
from .shards import read_shard, sha256_file, verify_shard, write_atomic


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("cache generation requires numpy") from exc
    return np


@dataclass
class CacheConfig:
    cache_dir: Path
    model_id: str = "google/siglip2-base-patch16-224"
    model_revision: str = "REQUIRED_PINNED_REVISION"
    image_size: int = 224
    embedding_dim: int = 768
    batch_size: int = 32
    decode_workers: int = 8
    prepared_queue_batches: int = 3
    shard_size: int = 4096
    max_attempts: int = 3
    retry_permanent: bool = False
    queue_capacity: int = 8192
    cache_schema_version: int = 1
    content_fingerprint: bool = False
    progress_interval_seconds: float = 5.0

    def contract(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "preprocessing": preprocessing_contract(self.image_size),
            "embedding_dim": self.embedding_dim,
            "embedding_dtype": "float32",
            "cache_schema_version": self.cache_schema_version,
        }

    def preprocessing_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.contract()["preprocessing"], sort_keys=True).encode()).hexdigest()


@dataclass
class DecodeResult:
    item: ManifestItem
    image: Any | None = None
    decoder: str | None = None
    error: BaseException | None = None
    traceback_text: str | None = None
    elapsed_seconds: float = 0.0


@dataclass
class EmbeddingBatch:
    items: list[ManifestItem]
    embeddings: Any


@dataclass
class Progress:
    run_id: str
    total_images: int
    batch_size: int
    decode_workers: int
    prepared_queue_capacity: int
    started_at: str = field(default_factory=utc_now)
    state: str = "running"
    phase: str = "decode"
    completed_images: int = 0
    failed_images: int = 0
    completed_shards: int = 0
    total_shards: int = 0
    current_shard: int | None = None
    gpu_starvation_percent: float = 0.0
    last_checkpoint: str | None = None
    last_error: str | None = None
    decoded_images: int = 0
    inference_batches: int = 0
    inference_images: int = 0
    decode_seconds: float = 0.0
    inference_seconds: float = 0.0
    writer_seconds: float = 0.0
    decode_backpressure_seconds: float = 0.0
    inference_backpressure_seconds: float = 0.0
    inference_wait_seconds: float = 0.0
    writer_wait_seconds: float = 0.0

    def snapshot(self, started_monotonic: float, queue_size: int, pid: int | None = None,
                 embedding_queue_size: int = 0) -> dict[str, Any]:
        elapsed = max(0.0, time.monotonic() - started_monotonic)
        rate = self.completed_images / elapsed if elapsed else 0.0
        remaining = max(0, self.total_images - self.completed_images - self.failed_images)
        inference_total = self.inference_seconds + self.inference_wait_seconds
        gpu_starvation = (100.0 * self.inference_wait_seconds / inference_total) if inference_total else 0.0
        return {
            "schema_version": 1, "run_id": self.run_id, "state": self.state, "phase": self.phase,
            "total_images": self.total_images, "completed_images": self.completed_images,
            "failed_images": self.failed_images, "images_per_second": rate,
            "elapsed_seconds": elapsed, "eta_seconds": remaining / rate if rate else None,
            "current_shard": self.current_shard, "completed_shards": self.completed_shards,
            "total_shards": self.total_shards, "batch_size": self.batch_size,
            "decode_workers": self.decode_workers, "prepared_queue_size": queue_size,
            "prepared_queue_capacity": self.prepared_queue_capacity,
            "embedding_queue_size": embedding_queue_size,
            "gpu_starvation_percent": gpu_starvation,
            "decoded_images": self.decoded_images, "inference_batches": self.inference_batches,
            "inference_images": self.inference_images,
            "decode_seconds": self.decode_seconds, "inference_seconds": self.inference_seconds,
            "writer_seconds": self.writer_seconds,
            "decode_backpressure_seconds": self.decode_backpressure_seconds,
            "inference_backpressure_seconds": self.inference_backpressure_seconds,
            "inference_wait_seconds": self.inference_wait_seconds,
            "writer_wait_seconds": self.writer_wait_seconds,
            "resident_memory_gb": _resident_memory_gb(), "swap_delta_gb": 0.0,
            "started_at": self.started_at, "updated_at": utc_now(),
            "last_checkpoint": self.last_checkpoint, "last_error": self.last_error,
            "pid": pid or os.getpid(), "heartbeat_at": utc_now(),
        }


def _resident_memory_gb() -> float:
    try:
        import resource
        # macOS returns bytes, Linux returns KiB.
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.uname().sysname == "Darwin":
            return value / (1024 ** 3)
        return value / (1024 ** 2)
    except (ImportError, AttributeError, OSError):
        return 0.0


def _write_progress(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _queue_put(target: queue.Queue, value: Any, cancel: threading.Event) -> bool:
    """Bounded put that unblocks a producer when a downstream stage fails."""
    while True:
        try:
            target.put(value, timeout=0.2)
            return True
        except queue.Full:
            if cancel.is_set():
                return False


class CacheGenerator:
    """Generate a deterministic cache from a manifest and injected embedder."""

    def __init__(self, config: CacheConfig, *, embedder_factory: Callable[[], Any] | None = None,
                 prepare_fn: Callable[[Path, int], tuple[Any, str]] = prepare_image):
        self.config = config
        self.embedder_factory = embedder_factory or (lambda: SigLIP2Embedder.production(config.model_id, config.model_revision))
        self.prepare_fn = prepare_fn
        self.cancel = threading.Event()
        self._counter_lock = threading.Lock()
        self._started_monotonic = 0.0
        self._progress: Progress | None = None
        self._progress_thread: threading.Thread | None = None
        self._ledger: Ledger | None = None
        self._prepared_queue: queue.Queue | None = None
        self._embedding_queue: queue.Queue | None = None

    @property
    def progress_path(self) -> Path:
        return self.config.cache_dir / "progress.json"

    @property
    def manifest_path(self) -> Path:
        return self.config.cache_dir / "manifest.json"

    def cancel_run(self) -> None:
        self.cancel.set()

    def run(self, manifest: Iterable[ManifestItem], *, run_id: str | None = None) -> dict[str, Any]:
        self.cancel.clear()
        if self.config.model_revision == "REQUIRED_PINNED_REVISION":
            raise ValueError("model_revision must be an immutable Hugging Face revision, not a floating default")
        items = sorted(list(manifest), key=lambda item: item.manifest_position)
        if len({item.image_id for item in items}) != len(items):
            raise ValueError("manifest contains duplicate image IDs")
        if any(items[i].manifest_position >= items[i + 1].manifest_position for i in range(len(items) - 1)):
            raise ValueError("manifest positions must be strictly increasing")
        run_id = run_id or uuid.uuid4().hex
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        lock = LedgerLock(self.config.cache_dir / "cache.lock")
        lock.acquire()
        ledger = Ledger(self.config.cache_dir / "work.sqlite3", queue_capacity=self.config.queue_capacity)
        self._ledger = ledger
        progress = Progress(run_id, len(items), self.config.batch_size, self.config.decode_workers,
                            self.config.batch_size * self.config.prepared_queue_batches,
                            total_shards=(max((item.manifest_position for item in items), default=-1) // self.config.shard_size + 1))
        self._progress = progress
        self._started_monotonic = time.monotonic()
        final_state = "completed"
        error_message = None
        dataset_fingerprint: str | None = None
        previous_handlers: dict[int, Any] = {}
        try:
            ledger.open()
            dataset_fingerprint = self._dataset_fingerprint(items)
            self._write_manifest(items, dataset_fingerprint)
            ledger.seed_items([item.as_ledger_row() for item in items])
            self._reconcile(ledger, items)
            counts = ledger.counts()
            progress.completed_images = counts.get("completed", 0)
            progress.failed_images = counts.get("permanent_error", 0) + counts.get("retryable_error", 0)
            ledger.start_run(run_id, model_revision=self.config.model_revision,
                             preprocessing_hash=self.config.preprocessing_hash(),
                             dataset_fingerprint=dataset_fingerprint,
                             cache_schema_version=self.config.cache_schema_version,
                             config=asdict(self.config) | {"cache_dir": str(self.config.cache_dir)})
            ledger.start_writer()
            self._start_progress_writer(items)
            if threading.current_thread() is threading.main_thread():
                for signum in (signal.SIGINT, signal.SIGTERM):
                    previous_handlers[signum] = signal.getsignal(signum)
                    signal.signal(signum, lambda *_: self.cancel_run())
            pending = ledger.pending_items(self.config.max_attempts, self.config.retry_permanent)
            by_id = {item.image_id: item for item in items}
            work = [by_id[row["image_id"]] for row in pending]
            self._run_pipeline(work, ledger, self._next_shard_id(ledger))
            if self.cancel.is_set():
                final_state = "cancelled"
        except KeyboardInterrupt:
            self.cancel_run()
            final_state = "cancelled"
        except BaseException as exc:
            final_state = "failed"
            error_message = str(exc)[:500]
            progress.last_error = error_message
            raise
        finally:
            self.cancel.set()
            if self._progress_thread:
                self._progress_thread.join(timeout=10)
            progress.state = final_state
            _write_progress(self.progress_path, progress.snapshot(self._started_monotonic, 0))
            if dataset_fingerprint is not None:
                self._write_manifest(items, dataset_fingerprint)
            try:
                if ledger._thread:
                    ledger.stop(final_state, error_message)
            finally:
                for signum, handler in previous_handlers.items():
                    with contextlib_suppress():
                        signal.signal(signum, handler)
                lock.release()
        return progress.snapshot(self._started_monotonic, 0)

    def _dataset_fingerprint(self, items: list[ManifestItem]) -> str:
        digest = hashlib.sha256()
        for item in items:
            digest.update(f"{item.image_id}\0{item.manifest_position}\0{item.source_size}\0{item.source_mtime_ns}\0{item.source_fingerprint or ''}\n".encode())
        return digest.hexdigest()

    def _write_manifest(self, items: list[ManifestItem], fingerprint: str) -> None:
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        def version(name: str) -> str | None:
            try:
                from importlib.metadata import version as package_version
                return package_version(name)
            except Exception:
                return None
        contract = self.config.contract()
        if self.manifest_path.exists():
            try:
                with self.manifest_path.open("r", encoding="utf-8") as handle:
                    previous = json.load(handle)
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"cannot validate existing cache manifest: {self.manifest_path}") from exc
            if previous.get("dataset_fingerprint") != fingerprint or previous.get("contract") != contract:
                raise ValueError(
                    "existing cache manifest identity differs (dataset/model/preprocessing/schema); "
                    "use a new cache version instead of mixing embeddings"
                )
        payload = {"schema_version": self.config.cache_schema_version, "dataset_fingerprint": fingerprint,
                   "contract": contract, "items": [asdict(item) for item in items],
                   "provenance": {"python": __import__("platform").python_version(), "numpy": version("numpy"),
                                  "transformers": version("transformers"), "mlx": version("mlx"),
                                  "pillow": version("Pillow"), "decoder_preference": ["macos_imageio", "pillow"]}}
        if self._progress is not None:
            elapsed = max(0.0, time.monotonic() - self._started_monotonic)
            payload["extraction"] = {"state": self._progress.state, "completed_images": self._progress.completed_images,
                                     "failed_images": self._progress.failed_images, "elapsed_seconds": elapsed}
        temporary = self.manifest_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, self.manifest_path)

    def _reconcile(self, ledger: Ledger, items: list[ManifestItem]) -> None:
        """Reset orphaned in-flight rows and quarantine invalid shard files."""
        # Any in-flight row is unsafe after an interrupted process.  Valid
        # committed shards remain completed; invalid ones are reset below.
        ledger._conn = None
        # This event runs synchronously before the writer starts by using a
        # short-lived direct transaction; no inference thread exists yet.
        conn = __import__("sqlite3").connect(ledger.path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("UPDATE items SET state='pending',shard_id=NULL,shard_row=NULL,embedding_checksum=NULL WHERE state IN ('decoding','decoded','embedding','staged')")
        conn.commit(); conn.close()
        for path in sorted(self.config.cache_dir.glob("shard-*.safetensors")):
            shard_id = int(path.stem.split("-")[-1])
            try:
                metadata = verify_shard(path, self.config.embedding_dim)
            except Exception:
                quarantine = path.with_suffix(path.suffix + f".corrupt-{int(time.time())}")
                os.replace(path, quarantine)
                sidecar = path.with_suffix(path.suffix + ".json")
                if sidecar.exists(): os.replace(sidecar, quarantine.with_suffix(quarantine.suffix + ".json"))
                conn = __import__("sqlite3").connect(ledger.path)
                conn.execute("UPDATE items SET state='pending',shard_id=NULL,shard_row=NULL,embedding_checksum=NULL WHERE shard_id=?", (shard_id,))
                conn.execute("DELETE FROM shards WHERE shard_id=?", (shard_id,))
                conn.commit(); conn.close()
                continue
            known = ledger.read("SELECT 1 FROM shards WHERE shard_id=?", (shard_id,))
            if not known:
                # Crash after rename but before SQLite commit: adopt only if all
                # IDs/positions still match the current manifest.
                by_pos = {item.manifest_position: item for item in items}
                if all(pos in by_pos and by_pos[pos].image_id == image_id for pos, image_id in zip(metadata["positions"], metadata["image_ids"])):
                    np = _numpy(); embeddings, _ = read_shard(path)
                    rows = [{"image_id": image_id, "position": pos, "checksum": hashlib.sha256(embeddings[row].tobytes()).hexdigest()} for row, (pos, image_id) in enumerate(zip(metadata["positions"], metadata["image_ids"]))]
                    ledger.open() if False else None
                    # Direct adoption is safe before a writer is started.
                    conn = __import__("sqlite3").connect(ledger.path)
                    conn.execute("BEGIN")
                    sid = shard_id
                    conn.execute("INSERT OR REPLACE INTO shards VALUES(?,?,?,?,?,?,?)", (sid, str(path), len(rows), metadata["sha256"], min(metadata["positions"]), max(metadata["positions"]), utc_now()))
                    for row, value in enumerate(rows):
                        conn.execute("UPDATE items SET state='completed',shard_id=?,shard_row=?,embedding_checksum=?,updated_at=?,finished_at=? WHERE image_id=? AND manifest_position=?", (sid,row,value["checksum"],utc_now(),utc_now(),value["image_id"],value["position"]))
                    conn.commit(); conn.close()

    def _start_progress_writer(self, items: list[ManifestItem]) -> None:
        assert self._progress is not None
        def loop() -> None:
            is_tty = sys.stderr.isatty()
            try:
                from tqdm import tqdm  # type: ignore
                bar = tqdm(total=self._progress.total_images, initial=self._progress.completed_images + self._progress.failed_images,
                           unit="image", dynamic_ncols=True, disable=not is_tty)
            except ImportError:  # pragma: no cover - minimal environments
                bar = None
            last_bar = self._progress.completed_images + self._progress.failed_images
            try:
                while not self.cancel.wait(self.config.progress_interval_seconds):
                    with self._counter_lock:
                        value = self._progress.snapshot(
                            self._started_monotonic,
                            self._prepared_queue.qsize() if self._prepared_queue else 0,
                            embedding_queue_size=self._embedding_queue.qsize() if self._embedding_queue else 0,
                        )
                        current = self._progress.completed_images + self._progress.failed_images
                    if bar is not None and current != last_bar:
                        bar.update(current - last_bar); last_bar = current
                        bar.set_postfix_str(f"{value['images_per_second']:.1f}/s shard={value['current_shard']}")
                    elif bar is None or not is_tty:
                        print(f"progress: {current}/{value['total_images']} images, {value['images_per_second']:.1f}/s, shard={value['current_shard']}", flush=True)
                    if self._ledger is not None and self._ledger._thread is not None and self._progress is not None:
                        # Heartbeat is durable SQLite state, independent of the
                        # disposable progress snapshot.
                        self._ledger.emit("heartbeat", run_id=self._progress.run_id)
                    _write_progress(self.progress_path, value)
            finally:
                if bar is not None: bar.close()
        self._progress_thread = threading.Thread(target=loop, name="embedding-progress", daemon=True)
        self._progress_thread.start()

    def _next_shard_id(self, ledger: Ledger) -> int:
        """Allocate after reconciliation, including any adopted orphan files."""
        db_next = ledger.next_shard_id()
        file_ids = []
        for path in self.config.cache_dir.glob("shard-*.safetensors"):
            try:
                file_ids.append(int(path.stem.split("-")[-1]))
            except ValueError:
                continue
        return max([db_next, *(value + 1 for value in file_ids)], default=db_next)

    def _run_pipeline(self, items: list[ManifestItem], ledger: Ledger, first_shard_id: int) -> None:
        if not items:
            return
        prepared_capacity = max(1, self.config.batch_size * self.config.prepared_queue_batches)
        decoded_q: queue.Queue[DecodeResult | None] = queue.Queue(maxsize=prepared_capacity)
        self._prepared_queue = decoded_q
        batches_q: queue.Queue[EmbeddingBatch | None] = queue.Queue(maxsize=max(2, self.config.prepared_queue_batches))
        self._embedding_queue = batches_q
        dispatcher = threading.Thread(target=self._decode_dispatcher, args=(items, decoded_q, ledger), name="image-decode-dispatcher", daemon=True)
        inference = threading.Thread(target=self._inference_worker, args=(items, decoded_q, batches_q, ledger), name="siglip2-inference", daemon=True)
        writer = threading.Thread(target=self._writer_worker, args=(batches_q, ledger, first_shard_id), name="shard-writer", daemon=True)
        dispatcher.start(); inference.start(); writer.start()
        dispatcher.join(); inference.join(); writer.join()
        if self._progress and self._progress.last_error:
            raise RuntimeError(self._progress.last_error)

    def _decode_dispatcher(self, items: list[ManifestItem], output: queue.Queue, ledger: Ledger) -> None:
        max_outstanding = max(self.config.decode_workers * 2, self.config.decode_workers)
        iterator = iter(items); futures: dict[Future, ManifestItem] = {}
        try:
            with ThreadPoolExecutor(max_workers=self.config.decode_workers, thread_name_prefix="decode") as pool:
                while not self.cancel.is_set() and len(futures) < max_outstanding:
                    try: item = next(iterator)
                    except StopIteration: break
                    ledger.emit("state", image_id=item.image_id, state="decoding")
                    futures[pool.submit(self._decode_one, item)] = item
                while futures and not self.cancel.is_set():
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        item = futures.pop(future)
                        try: result = future.result()
                        except BaseException as exc: result = DecodeResult(item, error=exc, traceback_text=traceback.format_exc())
                        put_started = time.perf_counter()
                        if not _queue_put(output, result, self.cancel):
                            return
                        if self._progress:
                            with self._counter_lock:
                                self._progress.decode_backpressure_seconds += time.perf_counter() - put_started
                                if not result.error:
                                    self._progress.decoded_images += 1
                                    self._progress.decode_seconds += result.elapsed_seconds
                        while len(futures) < max_outstanding:
                            try: next_item = next(iterator)
                            except StopIteration: break
                            ledger.emit("state", image_id=next_item.image_id, state="decoding")
                            futures[pool.submit(self._decode_one, next_item)] = next_item
        except BaseException as exc:
            if self._progress: self._progress.last_error = str(exc)[:500]
            self.cancel.set()
        finally:
            _queue_put(output, None, self.cancel)

    def _decode_one(self, item: ManifestItem) -> DecodeResult:
        started = time.perf_counter()
        path = Path(item.source_path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        stat = path.stat()
        if stat.st_size != item.source_size or stat.st_mtime_ns != item.source_mtime_ns:
            raise ValueError(f"source changed after manifest creation: {path}")
        if self.config.content_fingerprint and item.source_fingerprint:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024): digest.update(chunk)
            if digest.hexdigest() != item.source_fingerprint:
                raise ValueError(f"source content changed after manifest creation: {path}")
        image, decoder = self.prepare_fn(path, self.config.image_size)
        return DecodeResult(item, image=image, decoder=decoder,
                            elapsed_seconds=time.perf_counter() - started)

    def _inference_worker(self, items: list[ManifestItem], input_q: queue.Queue, output_q: queue.Queue, ledger: Ledger) -> None:
        expected = [item.manifest_position for item in items]; expected_index = 0; waiting: list[tuple[int, DecodeResult]] = []
        embedder = None
        batch_items: list[ManifestItem] = []
        batch_images: list[Any] = []

        def emit_batch() -> None:
            nonlocal batch_items, batch_images
            if not batch_items:
                return
            try:
                for item in batch_items: ledger.emit("state", image_id=item.image_id, state="embedding")
                infer_started = time.perf_counter()
                embeddings = embedder.embed(_numpy().stack(batch_images, axis=0))
                infer_elapsed = time.perf_counter() - infer_started
                if embeddings.shape[0] != len(batch_items):
                    raise ValueError("embedder changed batch row count")
                put_started = time.perf_counter()
                _queue_put(output_q, EmbeddingBatch(batch_items, embeddings), self.cancel)
                if self._progress:
                    with self._counter_lock:
                        self._progress.inference_batches += 1
                        self._progress.inference_images += len(batch_items)
                        self._progress.inference_seconds += infer_elapsed
                        self._progress.inference_backpressure_seconds += time.perf_counter() - put_started
            except BaseException as exc:
                for item in batch_items:
                    ledger.emit("state", image_id=item.image_id, state="retryable_error", max_attempts=self.config.max_attempts,
                                error_type=type(exc).__name__, error_message=str(exc)[:500])
                    with self._counter_lock:
                        if self._progress: self._progress.failed_images += 1
            batch_items = []
            batch_images = []

        try:
            embedder = self.embedder_factory()  # sole owner, created on this thread
            if self._progress:
                with self._counter_lock:
                    self._progress.phase = "embedding"
            while not self.cancel.is_set():
                wait_started = time.perf_counter()
                result = input_q.get()
                if self._progress:
                    with self._counter_lock:
                        self._progress.inference_wait_seconds += time.perf_counter() - wait_started
                if result is None: break
                heapq.heappush(waiting, (result.item.manifest_position, result))
                while waiting and expected_index < len(expected) and waiting[0][0] == expected[expected_index]:
                    _, current = heapq.heappop(waiting); expected_index += 1
                    if current.error:
                        permanent = isinstance(current.error, (FileNotFoundError, ValueError))
                        state = "permanent_error" if permanent else ("retryable_error" if current.item else "permanent_error")
                        ledger.emit("state", image_id=current.item.image_id, state=state, max_attempts=self.config.max_attempts,
                                    error_type=type(current.error).__name__, error_message=str(current.error)[:500])
                        with self._counter_lock:
                            if self._progress: self._progress.failed_images += 1
                        continue
                    ledger.emit("state", image_id=current.item.image_id, state="decoded", decoder=current.decoder)
                    batch_items.append(current.item); batch_images.append(current.image)
                    if len(batch_items) >= self.config.batch_size:
                        emit_batch()
            # Flush the final partial batch after the dispatcher sentinel.
            while waiting and expected_index < len(expected) and waiting[0][0] == expected[expected_index]:
                _, current = heapq.heappop(waiting); expected_index += 1
                if current.error:
                    permanent = isinstance(current.error, (FileNotFoundError, ValueError))
                    state = "permanent_error" if permanent else "retryable_error"
                    ledger.emit("state", image_id=current.item.image_id, state=state, max_attempts=self.config.max_attempts,
                                error_type=type(current.error).__name__, error_message=str(current.error)[:500])
                    with self._counter_lock:
                        if self._progress: self._progress.failed_images += 1
                else:
                    ledger.emit("state", image_id=current.item.image_id, state="decoded", decoder=current.decoder)
                    batch_items.append(current.item); batch_images.append(current.image)
                    if len(batch_items) >= self.config.batch_size: emit_batch()
            emit_batch()
        except BaseException as exc:
            if self._progress: self._progress.last_error = str(exc)[:500]
            self.cancel.set()
        finally:
            _queue_put(output_q, None, self.cancel)

    def _writer_worker(self, input_q: queue.Queue, ledger: Ledger, first_shard_id: int) -> None:
        np = _numpy(); current_shard: int | None = None; logical_shard: int | None = None
        next_shard_id = first_shard_id; shard_items: list[ManifestItem] = []; vectors: list[Any] = []
        def flush() -> None:
            nonlocal current_shard, shard_items, vectors
            if current_shard is None or not vectors: return
            write_started = time.perf_counter()
            path = self.config.cache_dir / f"shard-{current_shard:05d}.safetensors"
            matrix = np.asarray(vectors, dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1)
            if matrix.ndim != 2 or matrix.shape[1] != self.config.embedding_dim or not np.isfinite(matrix).all() or (len(norms) and not np.allclose(norms, 1.0, atol=5e-3)):
                raise ValueError(f"invalid embedding batch for shard {current_shard}")
            checksum, _ = write_atomic(path, matrix, [x.manifest_position for x in shard_items], [x.image_id for x in shard_items], cache_schema_version=self.config.cache_schema_version)
            rows = [{"image_id": item.image_id, "position": item.manifest_position, "checksum": hashlib.sha256(matrix[i].tobytes()).hexdigest()} for i, item in enumerate(shard_items)]
            ledger.emit("commit_shard", wait=True, shard_id=current_shard, path=str(path), row_count=len(rows), sha256=checksum,
                        min_position=min(x["position"] for x in rows), max_position=max(x["position"] for x in rows), items=rows)
            with self._counter_lock:
                if self._progress:
                    self._progress.completed_images += len(rows); self._progress.completed_shards += 1; self._progress.current_shard = current_shard
                    self._progress.last_checkpoint = str(path)
                    self._progress.writer_seconds += time.perf_counter() - write_started
            current_shard = None; shard_items = []; vectors = []
        try:
            if self._progress:
                with self._counter_lock:
                    self._progress.phase = "writing"
            while not self.cancel.is_set():
                wait_started = time.perf_counter()
                batch = input_q.get()
                if self._progress:
                    with self._counter_lock:
                        self._progress.writer_wait_seconds += time.perf_counter() - wait_started
                if batch is None: break
                for item, vector in zip(batch.items, batch.embeddings):
                    logical = item.manifest_position // self.config.shard_size
                    if logical_shard is None:
                        logical_shard = logical
                        current_shard = next_shard_id
                        next_shard_id += 1
                    elif logical != logical_shard:
                        flush()
                        logical_shard = logical
                        current_shard = next_shard_id
                        next_shard_id += 1
                    ledger.emit("state", image_id=item.image_id, state="staged")
                    shard_items.append(item); vectors.append(vector)
            flush()
        except BaseException as exc:
            if self._progress: self._progress.last_error = str(exc)[:500]
            self.cancel.set()


class contextlib_suppress:
    def __enter__(self): return self
    def __exit__(self, *_): return True
