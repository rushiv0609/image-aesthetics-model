import pytest
import torch
from torch.utils.data import Dataset

from image_aesthetics.models.fusion import F5R, RegularizedLinear
from image_aesthetics.training.fusion_trainer import train_fusion


class _Rows(Dataset):
    def __init__(self, seed):
        generator = torch.Generator().manual_seed(seed)
        self.features = torch.randn(8, 1544, generator=generator)
        self.targets = torch.sigmoid(self.features[:, 1540] * 0.2)
        self.samples = [
            ("toy", i, {"native_score_range": [0.0, 1.0], "group_id": f"g{i}"},
             float(self.targets[i])) for i in range(8)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return {
            "features": self.features[index], "target": self.targets[index],
            "native_target": self.targets[index], "dataset_id": "toy",
            "image_id": str(index), "group_id": f"g{index}",
            "pairing_group": "toy",
        }


def test_f5_resume_requires_stable_variant_and_f2_identity(tmp_path):
    train_rows, validation_rows = _Rows(11), _Rows(12)
    torch.manual_seed(20)
    source_f2 = RegularizedLinear()
    model = F5R()
    model.initialize_from_f2(source_f2.state_dict())
    config = {
        "identity": "F5-R-v1-fold0-seed1-sourceabc",
        "model": "F5-R",
        "model_config": model.architecture_config(),
        "f2_initialization": {"identity": "sourceabc", "fold": 0, "seed": 1},
        "locked_test_accessed": False,
    }
    run_dir = tmp_path / "run"
    train_fusion(model, train_rows, validation_rows, run_dir, config=config,
                 epochs=1, min_epochs=1, patience=1, batch_size=4, seed=44,
                 ranking_weight=0, status_interval_batches=1, device="cpu")
    checkpoint = run_dir / "checkpoints"
    assert (checkpoint / "last.pt").is_file()

    resumed = F5R()
    train_fusion(resumed, train_rows, validation_rows, run_dir, config=config,
                 epochs=2, min_epochs=2, patience=1, batch_size=4, seed=44,
                 ranking_weight=0, status_interval_batches=1, device="cpu",
                 resume=checkpoint)
    assert resumed(torch.randn(3, 1544)).shape == (3,)

    changed_identity = {**config, "identity": "different-F5-X-or-F2-source"}
    with pytest.raises(ValueError, match="identity mismatch"):
        train_fusion(F5R(), train_rows, validation_rows, tmp_path / "other-run",
                     config=changed_identity, epochs=2, min_epochs=2, patience=1,
                     batch_size=4, seed=44, ranking_weight=0,
                     status_interval_batches=1, device="cpu", resume=checkpoint)
