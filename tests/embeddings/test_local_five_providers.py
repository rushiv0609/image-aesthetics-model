from __future__ import annotations

import numpy as np
import pytest
import os
from pathlib import Path

from image_aesthetics.embeddings.local_five_providers import LocalFiveExtractor, scalar_metrics


def test_scalar_metrics_frozen_goldens_and_boundary_behavior():
    images = np.zeros((3, 224, 224, 3), dtype=np.uint8)
    images[1, 0, 0] = 20  # Exercises Laplacian BORDER_REFLECT_101 boundary.
    images[2, :, 112:] = (255, 0, 0)
    result = scalar_metrics(images)
    assert result["sharpness"].dtype == np.float32
    assert result["sharpness"][:, 0] == pytest.approx([0.0, 0.00011957855167565867, 0.04297618940472603], abs=1e-9)
    assert result["contrast"][:, 0] == pytest.approx([0.0, 0.0, 0.34545454382896423], abs=1e-7)
    # Half black/half red: Hasler--Süsstrunk opponent metric, normalized by 100.
    assert result["colorfulness"][:, 0] == pytest.approx([0.0, 0.0, 1.0], abs=1e-7)


def test_scalar_metrics_batch_contract_and_determinism():
    rng = np.random.default_rng(20260925)
    images = rng.integers(0, 256, (4, 31, 47, 3), dtype=np.uint8)
    first, second = scalar_metrics(images), scalar_metrics(images.copy())
    assert set(first) == {"sharpness", "contrast", "colorfulness"}
    for name in first:
        assert first[name].shape == (4, 1) and first[name].dtype == np.float32
        assert np.all(np.isfinite(first[name])) and np.all((0 <= first[name]) & (first[name] <= 1))
        np.testing.assert_array_equal(first[name], second[name])


def test_a1_hash_mismatch_fails_closed(tmp_path):
    bad = tmp_path / "a1.safetensors"
    bad.write_bytes(b"not-the-locked-checkpoint")
    with pytest.raises(RuntimeError, match="checkpoint hash mismatch"):
        LocalFiveExtractor(a1_path=bad)


def test_local_five_shares_one_siglip_batch_and_a1_expected_rating():
    import torch
    class FakeSigLIP:
        def __init__(self): self.calls = 0
        def embed(self, images):
            self.calls += 1
            out = np.ones((len(images), 768), dtype=np.float32)
            return out / np.linalg.norm(out, axis=1, keepdims=True)
    class FakeA1:
        def __call__(self, embeddings):
            logits = torch.full((len(embeddings), 10), -20.0)
            logits[:, 9] = 20.0  # rating ten => normalized expected score one
            return logits
    siglip = FakeSigLIP()
    images = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    output = LocalFiveExtractor(siglip=siglip, a1=FakeA1()).extract(images)
    assert siglip.calls == 1
    assert output["siglip_embedding"].shape == (2, 768)
    assert output["a1_siglip"][:, 0] == pytest.approx([1.0, 1.0], abs=1e-6)
    assert set(output) == {"siglip_embedding", "a1_siglip", "sharpness", "contrast", "colorfulness"}


def test_scalar_metrics_match_opencv_reference_when_available():
    cv2=pytest.importorskip("cv2")
    rng=np.random.default_rng(42)
    images=rng.integers(0,256,(3,224,224,3),dtype=np.uint8)
    actual=scalar_metrics(images)
    expected={"sharpness":[],"contrast":[],"colorfulness":[]}
    for pixels in images:
        gray=cv2.cvtColor(pixels,cv2.COLOR_RGB2GRAY)
        lap=cv2.Laplacian(gray,cv2.CV_64F)
        expected["sharpness"].append(np.clip(lap.var()/1200.,0.,1.))
        p5,p95=np.percentile(gray,[5,95])
        expected["contrast"].append(np.clip((p95-p5)/220.,0.,1.))
        r,g,b=(pixels[...,i].astype(np.float32) for i in range(3))
        rg=np.abs(r-g); yb=np.abs(.5*(r+g)-b)
        metric=np.sqrt(rg.std()**2+yb.std()**2)+.3*np.sqrt(rg.mean()**2+yb.mean()**2)
        expected["colorfulness"].append(np.clip(metric/100.,0.,1.))
    for name, values in expected.items():
        np.testing.assert_allclose(actual[name][:,0],values,rtol=0,atol=1e-7)


def _parity_fixtures():
    n=224; rng=np.random.default_rng(20260926); y,x=np.mgrid[:n,:n]
    return np.stack([
        np.full((n,n,3),(90,120,140),np.uint8),
        np.repeat((105+((x+y)%10))[...,None].astype(np.uint8),3,axis=2),
        np.stack((100+(x%31),110+(y%27),120+((x+y)%23)),axis=-1).astype(np.uint8),
        rng.integers(100,110,(n,n,3),dtype=np.uint8),
        rng.integers(90,161,(n,n,3),dtype=np.uint8),
        np.repeat((110+((x+y)%2))[...,None].astype(np.uint8),3,axis=2),
    ])


def test_unsaturated_gradient_flat_and_noisy_scalar_goldens():
    actual=scalar_metrics(_parity_fixtures())
    # Frozen from Moments Generator's cv2 5.0.0 implementation, with local
    # scalar outputs cast to the cache's required float32 representation.
    goldens={
        "sharpness":[0.,.06688031554222107,.023473422974348068,.06365733593702316,1.,.013333333656191826],
        "contrast":[0.,.040909089148044586,.08181817829608917,.03181818127632141,.20000000298023224,.004545454401522875],
        "colorfulness":[.13829317688941956,0.,.1680363267660141,.044117361307144165,.3123946189880371,0.],
    }
    for name, expected in goldens.items():
        np.testing.assert_allclose(actual[name][:,0],expected,rtol=0,atol=1e-8)


def test_reproduces_and_fixes_half_luminance_checker_mismatch():
    cv2=pytest.importorskip("cv2")
    n=224; y,x=np.mgrid[:n,:n]
    a=np.array([45,215,60],dtype=np.uint8)   # 146.5 under the specified RGB weights
    b=np.array([45,216,60],dtype=np.uint8)
    pixels=np.where(((x//80+y//80)%2)[...,None],a,b).astype(np.uint8)
    gray=cv2.cvtColor(pixels,cv2.COLOR_RGB2GRAY)
    reference=np.clip(cv2.Laplacian(gray,cv2.CV_64F).var()/1200.,0.,1.)

    # The former float32 + ties-to-even implementation produced about
    # 3.03e-5 despite OpenCV seeing a constant uint8-gray image (score 0).
    old=pixels.astype(np.float32)
    old_gray=np.rint(.299*old[...,0]+.587*old[...,1]+.114*old[...,2]).astype(np.float32)
    old_pad=np.pad(old_gray,1,mode="reflect")
    old_lap=old_pad[:-2,1:-1]+old_pad[2:,1:-1]+old_pad[1:-1,:-2]+old_pad[1:-1,2:]-4*old_gray
    old_score=np.clip(old_lap.var()/1200.,0.,1.)
    assert abs(float(old_score)-float(reference))==pytest.approx(3.0293368e-5,abs=2e-9)

    fixed=float(scalar_metrics(pixels[None])["sharpness"][0,0])
    assert float(reference)==0.0
    assert fixed==float(reference)


def test_low_contrast_sharpness_tracks_moments_cv2_with_normalization_bound():
    cv2=pytest.importorskip("cv2")
    images=_parity_fixtures()
    actual=scalar_metrics(images)["sharpness"][:,0]
    expected=[]
    for pixels in images:
        gray=cv2.cvtColor(pixels,cv2.COLOR_RGB2GRAY)
        expected.append(np.clip(cv2.Laplacian(gray,cv2.CV_64F).var()/1200.,0.,1.))
    # The golden is checked to tight float32 output precision after /1200.
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-8)


def test_seeded_gaussian_sweep_matches_opencv_rgb2gray_exactly():
    cv2=pytest.importorskip("cv2")
    rng=np.random.default_rng(20260926)
    max_gray_mismatches=0
    max_score_delta=0.0
    reproduced_old_delta=None
    # Reproduce the independent reviewer sweep, including the failing index 8.
    for i in range(501):
        scale=(i%30)+1
        pixels=np.clip(127+rng.normal(0,scale,(224,224,3)),0,255).astype(np.uint8)
        expected_gray=cv2.cvtColor(pixels,cv2.COLOR_RGB2GRAY)
        # This was the former float luminance + half-up implementation.
        f=pixels.astype(np.float64)
        old_gray=np.clip(np.floor(.299*f[...,0]+.587*f[...,1]+.114*f[...,2]+.5),0,255).astype(np.uint8)
        if i==8:
            count=int(np.count_nonzero(old_gray!=expected_gray))
            old_lap=cv2.Laplacian(old_gray,cv2.CV_64F)
            expected_lap=cv2.Laplacian(expected_gray,cv2.CV_64F)
            reproduced_old_delta=abs(old_lap.var()/1200.-expected_lap.var()/1200.)
            assert count==34
            assert reproduced_old_delta==pytest.approx(6.22497e-5,abs=3e-9)

        # The production fixed-point grayscale kernel must match cvtColor pixel-for-pixel.
        rgb=pixels.astype(np.int32)
        fixed=((9798*rgb[...,0]+19235*rgb[...,1]+3735*rgb[...,2]+(1<<14))>>15).astype(np.uint8)
        gray_mismatches=int(np.count_nonzero(fixed!=expected_gray))
        max_gray_mismatches=max(max_gray_mismatches,gray_mismatches)
        assert gray_mismatches==0
        reference_score=np.clip(cv2.Laplacian(expected_gray,cv2.CV_64F).var()/1200.,0.,1.)
        actual_score=float(scalar_metrics(pixels[None])['sharpness'][0,0])
        max_score_delta=max(max_score_delta,abs(actual_score-float(np.float32(reference_score))))

    assert max_gray_mismatches==0
    # Grayscale is exact for all pixels; this bound is solely the final
    # float32 cache representation compared to OpenCV's float64 variance
    # rounded to the same contract dtype.
    assert max_score_delta<=1e-8
    assert reproduced_old_delta is not None and reproduced_old_delta>6.2e-5


def test_a1_derivation_matches_released_ava_embedding_fixture():
    fixture_root=Path(os.environ.get("AVA_EMBEDDING_FIXTURE_ROOT",
        "/Users/rushivyas/datasets/image-aesthetics/ava/embeddings/siglip2-base-patch16-224"))
    checkpoint=Path("/Users/rushivyas/datasets/image-aesthetics/ava/releases/siglip2-aesthetic/a1-v1/model.safetensors")
    shard=fixture_root/"full-v1"/"shard-00000.safetensors"
    if not shard.is_file() or not checkpoint.is_file(): pytest.skip("released AVA embedding/A1 artifacts unavailable")
    from image_aesthetics.embeddings.shards import read_shard
    embeddings,metadata=read_shard(shard)
    assert metadata["image_ids"][:3]==["953417","953777","953756"]
    class ReleasedSigLIP:
        def embed(self,images): return embeddings[:len(images)].copy()
    images=np.zeros((3,8,8,3),dtype=np.uint8)
    values=LocalFiveExtractor(siglip=ReleasedSigLIP(),a1_path=checkpoint).extract(images)["a1_siglip"][:,0]
    np.testing.assert_allclose(values,[.5590007901191711,.5637840628623962,.5465470552444458],rtol=0,atol=2e-6)


def test_siglip_released_cache_profiles_match_when_metal_is_not_used():
    fixture_root=Path(os.environ.get("AVA_EMBEDDING_FIXTURE_ROOT",
        "/Users/rushivyas/datasets/image-aesthetics/ava/embeddings/siglip2-base-patch16-224"))
    left=fixture_root/"profile-b32-w8"/"shard-00000.safetensors"
    right=fixture_root/"profile-b64-w8"/"shard-00000.safetensors"
    if not left.is_file() or not right.is_file(): pytest.skip("released SigLIP cache profiles unavailable")
    from image_aesthetics.embeddings.shards import read_shard
    a,ma=read_shard(left); b,mb=read_shard(right)
    assert ma["image_ids"]==mb["image_ids"] and ma["positions"]==mb["positions"]
    np.testing.assert_array_equal(a,b)
