"""Training dataset over identity-checked local-five, remaining-four, and NIMA shards."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file
from torch.utils.data import Dataset

FEATURE_DIMENSION = 1544
GROUP_DIMENSIONS = {"local_five": 772, "remaining_four": 771, "nima": 1}


class JoinedTenDataset(Dataset):
    """Loads a fixed-order 1,544-float feature row without copying merged shards."""

    def __init__(self, join_dir: Path, *, label_manifest: Path | None = None,
                 max_open_shards: int = 8, eligible_only: bool = False):
        self.join_dir = Path(join_dir)
        audit = json.loads((self.join_dir / "join-audit.json").read_text())
        if audit.get("active_signal_groups") != ["local_five", "remaining_four", "nima"]:
            raise ValueError("joined-ten audit does not contain all three signal groups")
        self.rows = [json.loads(line) for line in (self.join_dir / "join.jsonl").read_text().splitlines() if line]
        if eligible_only:
            self.rows = [row for row in self.rows if row.get("eligible_for_fusion") and not row.get("quarantine")]
        roots = audit.get("cache_roots")
        if not isinstance(roots, dict):
            raise ValueError("joined-ten audit is missing cache_roots")
        self.cache_roots = {name: Path(path) for name, path in roots.items()}
        self.max_open_shards = max_open_shards
        self._cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self.targets = None
        if label_manifest:
            labels = {str(row["image_id"]): row for row in
                      (json.loads(line) for line in Path(label_manifest).read_text().splitlines() if line)}
            self.targets = labels

    def __getstate__(self):
        """Do not serialize open shard arrays into spawned DataLoader workers.

        macOS uses ``spawn`` rather than ``fork``.  Keeping the per-process LRU
        cache out of the pickle avoids a costly copy of already-open safetensor
        shards; every worker repopulates its own bounded cache lazily.
        """
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state

    def __len__(self) -> int:
        return len(self.rows)

    def _shard(self, group: str, shard_id: int) -> np.ndarray:
        key = (group, shard_id)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        path = self.cache_roots[group] / f"shard-{shard_id:05d}.safetensors"
        value = load_file(str(path))["embeddings"]
        if value.ndim != 2 or value.shape[1] != GROUP_DIMENSIONS[group] or not np.isfinite(value).all():
            raise ValueError(f"invalid {group} shard: {path} {value.shape}")
        self._cache[key] = value
        while len(self._cache) > self.max_open_shards:
            self._cache.popitem(last=False)
        return value

    def __getitem__(self, index: int):
        row = self.rows[index]
        pieces = []
        for group in ("local_five", "remaining_four", "nima"):
            coordinate = row[group]
            pieces.append(self._shard(group, int(coordinate["shard_id"]))[int(coordinate["row"])])
        features = np.concatenate(pieces).astype(np.float32, copy=False)
        if features.shape != (FEATURE_DIMENSION,) or not np.isfinite(features).all():
            raise ValueError(f"invalid joined feature row for {row['image_id']}")
        result = {"features": torch.from_numpy(features), "image_id": str(row["image_id"]),
                  "dataset_id": str(row["dataset_id"]), "metadata": row}
        if self.targets is not None:
            target = self.targets.get(str(row["image_id"]))
            if target is None:
                raise ValueError(f"missing label row for {row['image_id']}")
            result["target"] = target
        return result
