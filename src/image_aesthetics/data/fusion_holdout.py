"""Dataset views for leakage-safe leave-one-dataset-out experiments."""

from __future__ import annotations

from typing import Any

from image_aesthetics.data.fusion_training_dataset import FusionTrainingDataset
from image_aesthetics.data.normalization import FeatureNormalizer


class DatasetExclusionView(FusionTrainingDataset):
    """A fusion view whose membership is explicitly constrained by dataset.

    The base constructor still enforces the development/locked-test boundary.
    Filtering happens before normalization is fitted, batching is constructed,
    or calibration ranges are derived.  Its identity records the exact policy,
    preventing ordinary fold normalizers from being reused in LODO runs.
    """

    def __init__(self, *args: Any, excluded_dataset: str | None = None,
                 only_dataset: str | None = None, **kwargs: Any):
        if bool(excluded_dataset) == bool(only_dataset):
            raise ValueError("specify exactly one of excluded_dataset or only_dataset")
        super().__init__(*args, **kwargs)
        self.excluded_dataset = str(excluded_dataset) if excluded_dataset else None
        self.only_dataset = str(only_dataset) if only_dataset else None
        if self.excluded_dataset:
            self.samples = [sample for sample in self.samples
                            if sample[0] != self.excluded_dataset]
        else:
            self.samples = [sample for sample in self.samples
                            if sample[0] == self.only_dataset]
        if not self.samples:
            raise ValueError("dataset exclusion policy produced an empty view")

    def normalization_identity(self) -> dict[str, Any]:
        identity = super().normalization_identity()
        identity.update({
            "schema_version": "fusion-training-lodo-view-v1",
            "excluded_dataset": self.excluded_dataset,
            "only_dataset": self.only_dataset,
            "visible_dataset_ids": sorted({sample[0] for sample in self.samples}),
        })
        return identity

    def normalized(self, normalizer: FeatureNormalizer):
        # Keep the established lazy normalization wrapper and exposed samples.
        return super().normalized(normalizer)
