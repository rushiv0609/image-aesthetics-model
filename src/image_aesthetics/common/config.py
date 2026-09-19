"""Small dependency-free configuration helpers used by cache commands.

The data root is intentionally outside the repository.  ``IAM_DATA_ROOT`` is
the preferred way to select it; command-line ``--data-root`` always wins.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def data_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Return the configured external data root without creating it."""

    value = explicit or os.environ.get("IAM_DATA_ROOT")
    return Path(value).expanduser() if value else Path.home() / "datasets" / "image-aesthetics"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
