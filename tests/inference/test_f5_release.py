import json

import pytest
import torch

from image_aesthetics.inference.f5_release import F5ReleaseScorer


def test_feature_tensor_is_strict_and_uses_contract_order():
    scalars = {name: (index + 1) / 10 for index, name in enumerate((
        "a1_siglip", "laion_clip", "nima", "musiq_aesthetic",
        "musiq_technical", "sharpness", "contrast", "colorfulness"))}
    features = F5ReleaseScorer.feature_tensor(torch.zeros(768), torch.ones(768), scalars)
    assert features.shape == (1544,)
    assert features[768].item() == pytest.approx(scalars["a1_siglip"])
    assert features[772].item() == 1.0
    assert features[1543].item() == pytest.approx(scalars["nima"])
    with pytest.raises(ValueError, match="contract mismatch"):
        F5ReleaseScorer.feature_tensor(torch.zeros(768), torch.ones(768), {})


def test_release_loader_scores_finite_range():
    release = "fusion/releases/f5-g-v1"
    scorer = F5ReleaseScorer(release)
    scalars = {name: 0.5 for name in (
        "a1_siglip", "laion_clip", "nima", "musiq_aesthetic",
        "musiq_technical", "sharpness", "contrast", "colorfulness")}
    score = scorer.score(torch.zeros(768), torch.zeros(768), scalars)
    assert 0.0 <= score <= 1.0


def test_release_manifest_detects_tampering(tmp_path):
    source = "fusion/releases/f5-g-v1"
    manifest = json.loads(open(f"{source}/manifest.json").read())
    manifest["files"]["model-config.json"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    for name in manifest["files"]:
        (tmp_path / name).symlink_to((__import__("pathlib").Path(source) / name).resolve())
    with pytest.raises(ValueError, match="SHA-256"):
        F5ReleaseScorer(tmp_path)
