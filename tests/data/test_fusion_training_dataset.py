import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file
from torch.utils.data import DataLoader

from image_aesthetics.data.fusion_training_dataset import DatasetBalancedBatchSampler, FusionTrainingDataset


def fixture(tmp_path: Path):
    join_dirs = {}
    split_rows = []
    for dataset_id, count in (("small", 2), ("large", 4)):
        roots = {}
        for group, dimension in (("local_five", 772), ("remaining_four", 771), ("nima", 1)):
            root = tmp_path / dataset_id / group
            root.mkdir(parents=True)
            save_file({"embeddings": np.ones((count, dimension), dtype=np.float32)}, str(root / "shard-00000.safetensors"))
            roots[group] = str(root)
        join = tmp_path / dataset_id / "join"
        join.mkdir()
        (join / "join-audit.json").write_text(json.dumps({"active_signal_groups": ["local_five", "remaining_four", "nima"], "cache_roots": roots}))
        joined = []
        for i in range(count):
            image_id = f"{dataset_id}:{i}"
            joined.append({"dataset_id": dataset_id, "image_id": image_id,
                           **{g: {"shard_id": 0, "row": i} for g in roots}})
            split_rows.append({"dataset_id": dataset_id, "image_id": image_id, "feature_id": image_id,
                               "native_score": i + 1, "native_score_range": [1, 5],
                               "locked_test": i == 0, "development_fold": None if i == 0 else i % 5,
                               "group_id": f"group-{dataset_id}-{i}"})
        (join / "join.jsonl").write_text("".join(json.dumps(row) + "\n" for row in joined))
        join_dirs[dataset_id] = join
    split = tmp_path / "split.jsonl"
    split.write_text("".join(json.dumps(row) + "\n" for row in split_rows))
    return split, join_dirs


def test_training_view_excludes_locked_and_validation_rows(tmp_path):
    split, joins = fixture(tmp_path)
    dataset = FusionTrainingDataset(split, joins, partition="train", validation_fold=1)
    assert all(not sample[2]["locked_test"] and sample[2]["development_fold"] != 1 for sample in dataset.samples)
    assert dataset[0]["features"].shape == (1544,)


def test_locked_test_requires_explicit_opt_in(tmp_path):
    split, joins = fixture(tmp_path)
    with pytest.raises(PermissionError):
        FusionTrainingDataset(split, joins, partition="locked_test")
    dataset = FusionTrainingDataset(split, joins, partition="locked_test", allow_locked_test=True)
    assert len(dataset) == 2
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch["development_fold"].tolist() == [-1, -1]


def test_balanced_batches_are_dataset_homogeneous(tmp_path):
    split, joins = fixture(tmp_path)
    dataset = FusionTrainingDataset(split, joins, partition="train", validation_fold=4)
    sampler = DatasetBalancedBatchSampler(dataset, 2)
    for batch in sampler:
        assert len({dataset.samples[index][0] for index in batch}) == 1
