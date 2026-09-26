"""Machine-wide serialization for Metal model construction and inference."""
from __future__ import annotations

import contextlib
import os
import threading
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows is not a supported target.
    fcntl = None

_process_lock = threading.RLock()
_local = threading.local()
_LOCK_PATH = Path(os.environ.get("IMAGE_AESTHETICS_METAL_LOCK", "/tmp/image-aesthetics-metal.lock"))


@contextlib.contextmanager
def metal_owner():
    """Take one process and host-wide lease before loading or using Metal."""
    depth = getattr(_local, "depth", 0)
    if depth:
        _local.depth = depth + 1
        try:
            yield
        finally:
            _local.depth -= 1
        return
    with _process_lock:
        _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK_PATH.open("a+") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            _local.depth = 1
            try:
                yield
            finally:
                _local.depth = 0
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
