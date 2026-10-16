"""Verified transfer of a matching fold/seed F2 scalar logit into F5."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def initialize_from_fold_f2(
    model: torch.nn.Module,
    checkpoint_path: Path,
    *,
    fold: int,
    seed: int,
    split_manifest: Path,
    normalization_path: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    """Validate provenance and copy a best F2 checkpoint's scalar-logit weights.

    The expected artifact layout is ``<run>/checkpoints/best.pt``. The sibling
    run config and checkpoint metadata bind the source to its fold, seed, split,
    normalizer, and architecture. No dataset rows, including locked-test rows,
    are loaded by this helper.
    """
    path = Path(checkpoint_path)
    if path.name != "best.pt" or path.parent.name != "checkpoints" or not path.is_file():
        raise ValueError("F5 initialization requires an existing <F2-run>/checkpoints/best.pt")
    run_dir = path.parent.parent
    config_path = run_dir / "config.json"
    metadata_path = path.parent / "checkpoint.json"
    if not config_path.is_file() or not metadata_path.is_file():
        raise ValueError("F2 checkpoint is missing its run config or checkpoint metadata")
    config = json.loads(config_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    payload = config.get("identity_payload") or {}
    spec = payload.get("spec") or {}
    source_model = spec.get("model", config.get("model"))
    source_fold = spec.get("fold", config.get("fold", -1))
    source_seed = spec.get("seed", config.get("seed", -1))
    if config.get("model") != "F2" or source_model != "F2":
        raise ValueError("initialization source is not an F2 run")
    if config.get("locked_test_accessed") is not False:
        raise ValueError("F2 initialization source does not certify locked-test isolation")
    if int(source_fold) != fold or int(source_seed) != seed:
        raise ValueError("F2 initialization fold/seed does not match the requested F5 run")
    source_limit = payload.get("limit", config.get("limit"))
    if source_limit != limit:
        raise ValueError("F2 initialization subset limit does not match the requested F5 run")
    source_split_sha = payload.get("split_sha256")
    if source_split_sha is None:
        source_split_sha = (payload.get("split") or {}).get("split_manifest_sha256")
    if source_split_sha != sha256_file(Path(split_manifest)):
        raise ValueError("F2 initialization uses a different split manifest")
    if payload.get("normalize", config.get("normalize")) is not True:
        raise ValueError("F2 initialization was not trained with fold-fitted normalization")
    source_normalization_sha = payload.get("normalization_stats_sha256") or config.get("normalization_stats_sha256")
    if source_normalization_sha is None:
        for candidate in (run_dir / "normalization.json", run_dir / "normalization-smoke.json"):
            if candidate.is_file():
                source_normalization_sha = sha256_file(candidate)
                break
    if source_normalization_sha != sha256_file(Path(normalization_path)):
        raise ValueError("F2 initialization uses different fold normalization statistics")
    if metadata.get("identity") != config.get("identity"):
        raise ValueError("F2 checkpoint identity does not match its run config")

    # These checkpoints are locally produced optimizer/resume files. Loading
    # them with weights_only=False is needed for their RNG state; callers must
    # supply the project-generated, provenance-checked artifact above.
    state = torch.load(path, map_location="cpu", weights_only=False)
    model_state = state.get("model") if isinstance(state, dict) else None
    if not isinstance(model_state, dict):
        raise ValueError("F2 checkpoint is missing model state")
    model.initialize_from_f2(model_state)
    return {
        "model_id": "F2",
        "run_id": config.get("run_id"),
        "identity": config.get("identity"),
        "checkpoint_sha256": sha256_file(path),
        "fold": fold,
        "seed": seed,
        "source_best_epoch": metadata.get("best_epoch"),
        "locked_test_accessed": False,
    }
