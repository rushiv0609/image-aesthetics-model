"""Durable single-writer SQLite ledger for embedding extraction.

The ledger is deliberately independent from the MLX and image-decoding code.
Decoder/GPU threads enqueue tiny events; one thread owns the write connection.
That makes status transitions deterministic and avoids SQLite connections being
shared across worker threads.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # macOS/Linux; the fallback keeps unit tests portable.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


ITEM_STATES = {
    "pending", "decoding", "decoded", "embedding", "staged", "completed",
    "retryable_error", "permanent_error",
}
TERMINAL_STATES = {"completed", "retryable_error", "permanent_error"}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


@dataclass
class LedgerEvent:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    ack: queue.Queue | None = None


class LedgerLock:
    """An exclusive per-cache lock that survives process crashes."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._handle = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+")
        if fcntl is not None:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                self._handle.close()
                self._handle = None
                raise RuntimeError(f"cache is already locked: {self.path}") from exc

    def release(self) -> None:
        if self._handle is not None:
            if fcntl is not None:
                with contextlib.suppress(OSError):
                    fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "LedgerLock":
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


class Ledger:
    """SQLite-backed work ledger with a dedicated asynchronous writer.

    ``open`` and all read methods use short-lived connections.  Once ``start``
    is called, item updates must go through ``emit``; the writer thread is the
    sole owner of the write connection.
    """

    def __init__(self, path: Path, queue_capacity: int = 8192, batch_size: int = 512,
                 flush_interval: float = 1.0):
        self.path = Path(path)
        self.queue_capacity = queue_capacity
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._events: queue.Queue[LedgerEvent | None] = queue.Queue(maxsize=queue_capacity)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._writer_error: BaseException | None = None
        self._conn: sqlite3.Connection | None = None
        self._run_id: str | None = None

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=5)
        self._configure(conn)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, state TEXT NOT NULL,
                model_revision TEXT NOT NULL, preprocessing_hash TEXT NOT NULL,
                dataset_fingerprint TEXT, cache_schema_version INTEGER NOT NULL,
                started_at TEXT NOT NULL, heartbeat_at TEXT NOT NULL,
                finished_at TEXT, config_json TEXT NOT NULL, error_message TEXT
            );
            CREATE TABLE IF NOT EXISTS items (
                image_id TEXT PRIMARY KEY, manifest_position INTEGER NOT NULL UNIQUE,
                source_path TEXT NOT NULL, source_size INTEGER NOT NULL,
                source_mtime_ns INTEGER NOT NULL, source_fingerprint TEXT,
                state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                shard_id INTEGER, shard_row INTEGER, embedding_checksum TEXT,
                decoder TEXT, error_type TEXT, error_message TEXT,
                first_started_at TEXT, last_started_at TEXT, finished_at TEXT,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS shards (
                shard_id INTEGER PRIMARY KEY, path TEXT NOT NULL,
                row_count INTEGER NOT NULL, sha256 TEXT NOT NULL,
                min_position INTEGER NOT NULL, max_position INTEGER NOT NULL,
                committed_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_items_state ON items(state);
            CREATE INDEX IF NOT EXISTS idx_items_shard ON items(shard_id);
            CREATE INDEX IF NOT EXISTS idx_items_retry ON items(state, attempts);
            """
        )
        conn.commit()
        conn.close()

    @staticmethod
    def _configure(conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")

    def seed_items(self, items: list[dict[str, Any]], allow_source_update: bool = False) -> None:
        """Insert manifest rows and invalidate rows whose source identity changed."""
        conn = sqlite3.connect(self.path, timeout=5)
        self._configure(conn)
        now = utc_now()
        for item in items:
            existing = conn.execute(
                "SELECT source_path,source_size,source_mtime_ns,source_fingerprint,state FROM items WHERE image_id=?",
                (item["image_id"],),
            ).fetchone()
            if existing and (existing[0], existing[1], existing[2], existing[3]) != (
                item["source_path"], item["source_size"], item["source_mtime_ns"], item.get("source_fingerprint")
            ):
                if not allow_source_update:
                    raise ValueError(f"source changed for image {item['image_id']}; use a new cache version")
                conn.execute(
                    "UPDATE items SET state='pending',shard_id=NULL,shard_row=NULL,embedding_checksum=NULL,error_type=NULL,error_message=NULL,updated_at=? WHERE image_id=?",
                    (now, item["image_id"]),
                )
            else:
                conn.execute(
                    """INSERT OR IGNORE INTO items
                    (image_id,manifest_position,source_path,source_size,source_mtime_ns,source_fingerprint,state,updated_at)
                    VALUES (?,?,?,?,?,?, 'pending',?)""",
                    (item["image_id"], item["manifest_position"], item["source_path"], item["source_size"],
                     item["source_mtime_ns"], item.get("source_fingerprint"), now),
                )
        conn.commit()
        conn.close()

    def start_run(self, run_id: str, *, model_revision: str, preprocessing_hash: str,
                  dataset_fingerprint: str, cache_schema_version: int, config: dict[str, Any]) -> None:
        conn = sqlite3.connect(self.path, timeout=5)
        self._configure(conn)
        now = utc_now()
        conn.execute(
            """INSERT INTO runs(run_id,state,model_revision,preprocessing_hash,dataset_fingerprint,
               cache_schema_version,started_at,heartbeat_at,config_json)
               VALUES(?, 'running', ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(run_id) DO UPDATE SET state='running',heartbeat_at=?,error_message=NULL""",
            (run_id, model_revision, preprocessing_hash, dataset_fingerprint, cache_schema_version,
             now, now, json.dumps(config, sort_keys=True), now),
        )
        conn.commit()
        conn.close()
        self._run_id = run_id

    def start_writer(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._writer_error = None
        self._thread = threading.Thread(target=self._writer_loop, name="embedding-ledger", daemon=True)
        self._thread.start()

    def emit(self, kind: str, *, wait: bool = False, timeout: float = 30.0, **payload: Any) -> Any:
        if self._writer_error:
            raise RuntimeError("ledger writer failed") from self._writer_error
        ack: queue.Queue | None = queue.Queue(maxsize=1) if wait else None
        event = LedgerEvent(kind, payload, ack)
        # Blocking here is intentional: terminal states must never be dropped.
        self._events.put(event, timeout=timeout)
        if ack is not None:
            result = ack.get(timeout=timeout)
            if isinstance(result, BaseException):
                raise result
            return result
        return None

    def stop(self, state: str = "completed", error_message: str | None = None) -> None:
        if self._thread is None:
            return
        if not self._run_id:
            raise RuntimeError("cannot stop ledger before start_run")
        self.emit("run_state", run_id=self._run_id, state=state, error_message=error_message, wait=True)
        self._stop.set()
        self._events.put(None)
        self._thread.join(timeout=30)
        self._thread = None
        if self._writer_error:
            raise RuntimeError("ledger writer failed") from self._writer_error

    def _writer_loop(self) -> None:
        conn = sqlite3.connect(self.path, timeout=5)
        self._configure(conn)
        self._conn = conn
        pending: list[LedgerEvent] = []
        try:
            while not (self._stop.is_set() and self._events.empty() and not pending):
                try:
                    event = self._events.get(timeout=self.flush_interval)
                except queue.Empty:
                    event = None
                if event is not None:
                    pending.append(event)
                if pending and (len(pending) >= self.batch_size or event is None or self._stop.is_set()):
                    self._apply_batch(conn, pending)
                    pending.clear()
            if pending:
                self._apply_batch(conn, pending)
        except BaseException as exc:  # surface on the producer at stop/emit.
            self._writer_error = exc
            for event in pending:
                if event.ack:
                    event.ack.put(exc)
        finally:
            conn.close()
            self._conn = None

    def _apply_batch(self, conn: sqlite3.Connection, events: list[LedgerEvent]) -> None:
        results: list[tuple[LedgerEvent, Any]] = []
        try:
            conn.execute("BEGIN")
            for event in events:
                results.append((event, self._apply_event(conn, event)))
            conn.commit()
        except BaseException as exc:
            conn.rollback()
            for event in events:
                if event.ack:
                    event.ack.put(exc)
            raise
        for event, result in results:
            if event.ack:
                event.ack.put(result)

    def _apply_event(self, conn: sqlite3.Connection, event: LedgerEvent) -> Any:
        p = event.payload
        now = utc_now()
        if event.kind == "state":
            state = p["state"]
            if state not in ITEM_STATES:
                raise ValueError(f"invalid item state: {state}")
            fields = ["state=?", "updated_at=?"]
            values: list[Any] = [state, now]
            if state == "decoding":
                fields += ["attempts=attempts+1", "last_started_at=?", "first_started_at=COALESCE(first_started_at,?)"]
                values += [now, now]
            if state in TERMINAL_STATES:
                fields.append("finished_at=?")
                values.append(now)
            if state == "retryable_error" and p.get("max_attempts") is not None:
                # Three transient failures become a terminal permanent error;
                # the next resume therefore cannot spin forever on one file.
                fields[0] = "state=CASE WHEN attempts >= ? THEN 'permanent_error' ELSE ? END"
                values[0:1] = [p["max_attempts"], state]
            for name in ("decoder", "error_type", "error_message", "shard_id", "shard_row", "embedding_checksum"):
                if name in p:
                    fields.append(f"{name}=?")
                    values.append(p[name])
            values.append(p["image_id"])
            conn.execute(f"UPDATE items SET {','.join(fields)} WHERE image_id=?", values)
        elif event.kind == "heartbeat":
            conn.execute("UPDATE runs SET heartbeat_at=? WHERE run_id=?", (now, p["run_id"]))
        elif event.kind == "run_state":
            conn.execute("UPDATE runs SET state=?,finished_at=?,heartbeat_at=?,error_message=? WHERE run_id=?",
                         (p["state"], now, now, p.get("error_message"), p["run_id"]))
        elif event.kind == "commit_shard":
            conn.execute(
                "INSERT OR REPLACE INTO shards(shard_id,path,row_count,sha256,min_position,max_position,committed_at) VALUES(?,?,?,?,?,?,?)",
                (p["shard_id"], p["path"], p["row_count"], p["sha256"], p["min_position"], p["max_position"], now),
            )
            for row, item in enumerate(p["items"]):
                conn.execute(
                    "UPDATE items SET state='completed',shard_id=?,shard_row=?,embedding_checksum=?,updated_at=?,finished_at=? WHERE image_id=? AND manifest_position=?",
                    (p["shard_id"], row, item["checksum"], now, now, item["image_id"], item["position"]),
                )
        elif event.kind == "reset_orphans":
            conn.execute("UPDATE items SET state='pending',shard_id=NULL,shard_row=NULL,embedding_checksum=NULL,updated_at=? WHERE state IN ('decoding','decoded','embedding','staged')", (now,))
        else:
            raise ValueError(f"unknown ledger event: {event.kind}")
        return True

    def read(self, sql: str, params: tuple = ()) -> list[tuple]:
        conn = sqlite3.connect(self.path, timeout=5)
        self._configure(conn)
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return rows

    def pending_items(self, max_attempts: int = 3, retry_permanent: bool = False) -> list[dict[str, Any]]:
        states = "('pending','retryable_error')" + (" OR state='permanent_error'" if retry_permanent else "")
        rows = self.read(f"SELECT image_id,manifest_position,source_path,source_size,source_mtime_ns,source_fingerprint,state,attempts FROM items WHERE (state IN {states}) AND attempts < ? ORDER BY manifest_position", (max_attempts,))
        keys = ("image_id", "manifest_position", "source_path", "source_size", "source_mtime_ns", "source_fingerprint", "state", "attempts")
        return [dict(zip(keys, row)) for row in rows]

    def counts(self) -> dict[str, int]:
        return {row[0]: row[1] for row in self.read("SELECT state,COUNT(*) FROM items GROUP BY state")}

    def status_counts(self) -> dict[str, int]:
        return dict(sorted(self.counts().items()))

    def failures(self) -> list[dict[str, Any]]:
        keys = ("image_id", "state", "error_type", "error_message")
        return [dict(zip(keys, row)) for row in self.read("SELECT image_id,state,error_type,error_message FROM items WHERE state LIKE '%error' ORDER BY image_id")]

    def completed_records(self) -> list[dict[str, Any]]:
        keys=("image_id","manifest_position","shard_id","shard_row","embedding_checksum")
        return [dict(zip(keys,row)) for row in self.read("SELECT image_id,manifest_position,shard_id,shard_row,embedding_checksum FROM items WHERE state='completed' ORDER BY manifest_position")]

    def committed_shards(self) -> list[dict[str, Any]]:
        keys=("shard_id","path","row_count","sha256")
        return [dict(zip(keys,row)) for row in self.read("SELECT shard_id,path,row_count,sha256 FROM shards ORDER BY shard_id")]

    def next_shard_id(self) -> int:
        """Return the next unused committed shard ID.

        Callers must hold the cache run lock.  The cache generator also checks
        the filesystem for orphan files during reconciliation before allocating
        IDs, so a crash cannot cause a later retry to overwrite a shard.
        """
        rows = self.read("SELECT COALESCE(MAX(shard_id), -1) FROM shards")
        return int(rows[0][0]) + 1
