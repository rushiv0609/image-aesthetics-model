import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from image_aesthetics.data.joined_ten_dataset import FEATURE_DIMENSION, JoinedTenDataset


def test_joined_ten_loads_fixed_feature_order(tmp_path: Path):
    roots = {}
    dimensions = {"local_five": 772, "remaining_four": 771, "nima": 1}
    values = {"local_five": 1.0, "remaining_four": 2.0, "nima": 3.0}
    for group, dimension in dimensions.items():
        root = tmp_path / group
        root.mkdir()
        save_file({"embeddings": np.full((1, dimension), values[group], dtype=np.float32)},
                  str(root / "shard-00000.safetensors"))
        roots[group] = str(root)
    join = tmp_path / "join"
    join.mkdir()
    (join / "join-audit.json").write_text(json.dumps({
        "active_signal_groups": ["local_five", "remaining_four", "nima"],
        "cache_roots": roots,
    }))
    row = {"dataset_id": "fixture", "image_id": "one", "eligible_for_fusion": True,
           "quarantine": False, **{group: {"shard_id": 0, "row": 0} for group in dimensions}}
    (join / "join.jsonl").write_text(json.dumps(row) + "\n")
    item = JoinedTenDataset(join)[0]
    assert tuple(item["features"].shape) == (FEATURE_DIMENSION,)
    assert np.all(item["features"][:772].numpy() == 1)
    assert np.all(item["features"][772:-1].numpy() == 2)
    assert item["features"][-1].item() == 3
