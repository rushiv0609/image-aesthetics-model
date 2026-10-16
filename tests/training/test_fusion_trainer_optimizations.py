import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from image_aesthetics.models.fusion import ScalarMLP
from image_aesthetics.training.fusion_trainer import _collect, _metrics, train_fusion
from scripts.training.benchmark_fusion_backends import ProvisionalF5


class _TinyValidationDataset(Dataset):
    def __init__(self):
        self.samples = [("a" if index % 2 else "b", index, {"group_id": str(index)}, 0.0) for index in range(9)]
        self.features = torch.randn(9, 1544)
        self.targets = torch.linspace(0.1, 0.9, 9)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return {
            "features": self.features[index], "target": self.targets[index],
            "native_target": self.targets[index], "dataset_id": self.samples[index][0],
            "image_id": str(index), "group_id": str(index),
            "pairing_group": self.samples[index][0], "development_fold": 0,
        }


def test_collect_batched_host_transfer_preserves_metrics():
    torch.manual_seed(17)
    dataset = _TinyValidationDataset()
    model = ScalarMLP().eval()
    loader = DataLoader(dataset, batch_size=3, shuffle=False)
    report = _collect(model, loader, torch.device("cpu"))
    with torch.no_grad():
        prediction = model(dataset.features).tolist()
    expected = _metrics(prediction, dataset.targets.tolist())
    assert report["overall"] == pytest.approx(expected)
    assert report["loss"] == pytest.approx(torch.nn.functional.huber_loss(torch.tensor(prediction), dataset.targets).item())


def test_status_interval_is_validated_and_keeps_final_status(tmp_path):
    dataset = _TinyValidationDataset()
    with pytest.raises(ValueError, match="status_interval_batches"):
        train_fusion(ScalarMLP(), dataset, dataset, tmp_path / "bad", config={"identity": "x"},
                     epochs=1, batch_size=3, device="cpu", status_interval_batches=0)
    result = train_fusion(ScalarMLP(), dataset, dataset, tmp_path / "run", config={"identity": "x"},
                          epochs=1, batch_size=3, device="cpu", status_interval_batches=100)
    assert result["state"] == "completed"
    assert result["global_step"] == 4  # two homogeneous dataset turns, twice.


def test_provisional_f5_benchmark_matches_gated_residual_input_contract():
    model = ProvisionalF5().eval()
    features = torch.zeros(3, 1544)
    scores = model(features)
    assert scores.shape == (3,)
    assert torch.all((scores >= 0) & (scores <= 1))
