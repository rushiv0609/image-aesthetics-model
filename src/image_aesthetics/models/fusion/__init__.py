"""Small, query-independent fusion baselines for the ten-signal scorer.

The package deliberately keeps dataset identifiers and calibration heads out of
production inputs: calibration is a training-only view of the shared score.
"""
from .contract import (FEATURE_DIM, EMBEDDING_DIM, SCALAR_DIM, SCALAR_NAMES,
                       SIGLIP_SLICE, CLIP_SLICE, SCALAR_INDICES, split_features,
                       flatten_features)
from .baselines import (WeightedScalarScore, ScalarEnsemble, RegularizedLinear,
                        ScalarMLP, FlatFeatureMLP, MOMENTS_F0_WEIGHTS, build_baseline)
from .calibration import MonotonicAffineCalibrator, DatasetCalibrationHeads
from .losses import calibrated_huber_loss, pairwise_ranking_loss, technical_auxiliary_loss

__all__ = ["FEATURE_DIM", "EMBEDDING_DIM", "SCALAR_DIM", "SCALAR_NAMES",
           "SIGLIP_SLICE", "CLIP_SLICE", "SCALAR_INDICES", "split_features",
           "flatten_features", "WeightedScalarScore", "ScalarEnsemble",
           "RegularizedLinear", "ScalarMLP", "FlatFeatureMLP", "build_baseline",
           "MOMENTS_F0_WEIGHTS",
           "MonotonicAffineCalibrator", "DatasetCalibrationHeads",
           "calibrated_huber_loss", "pairwise_ranking_loss", "technical_auxiliary_loss"]
