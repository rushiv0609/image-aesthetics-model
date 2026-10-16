import pytest
import torch

from image_aesthetics.models.fusion import (
    CorrectedScalarMLP, F5G, F5R, F5X, RegularizedLinear, ScalarMLP,
    SCALAR_INDICES, build_baseline, flatten_features,
)


def _features(batch=8):
    generator = torch.Generator().manual_seed(604)
    return torch.randn(batch, 1544, generator=generator)


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [(F5R, 163_755), (F5G, 297_555), (F5X, 246_635)],
)
def test_f5_parameter_counts_and_output_contract(model_type, expected):
    model = model_type(technical_head=True)
    scores = model(_features())
    assert model.param_count == expected
    assert scores.shape == (8,)
    assert torch.all((scores >= 0) & (scores <= 1))
    assert model.architecture_config()["feature_contract"] == "joined-ten-v1"


@pytest.mark.parametrize("name", ["F5-R", "F5-G", "F5-X"])
def test_zero_residual_is_exact_fold_f2_score(name):
    torch.manual_seed(91)
    f2 = RegularizedLinear()
    f5 = build_baseline(name)
    f5.initialize_from_f2(f2.state_dict())
    assert f5.f2_initialization_loaded
    assert torch.equal(f5.bounded_residual(_features()), torch.zeros(8))
    assert torch.equal(f5(_features()), f2(_features()))


def test_f3c_is_explicit_corrected_control_and_historical_f3_is_unchanged():
    old = ScalarMLP()
    corrected = build_baseline("F3C")
    assert isinstance(old.net[0], torch.nn.LayerNorm)
    assert isinstance(corrected, CorrectedScalarMLP)
    assert isinstance(corrected.net[0], torch.nn.Linear)
    assert corrected.architecture_config()["model_id"] == "F3C-no-per-image-layernorm-v1"
    assert corrected(_features()).shape == (8,)


def test_f5_g_gate_starts_at_one_and_remains_bounded():
    model = F5G()
    x = _features()
    assert torch.equal(model.gate_values(x), torch.ones(8, 8))
    with torch.no_grad():
        model.gate[-1].weight.normal_(0, 0.4)
        model.gate[-1].bias.normal_(0, 0.4)
    values = model.gate_values(x)
    assert values.min() >= 0.75 and values.max() <= 1.25


@pytest.mark.parametrize("model_type", [F5R, F5G, F5X])
def test_residual_bound_and_embedding_gradient_flow(model_type):
    model = model_type()
    with torch.no_grad():
        model.residual_head[-1].weight.fill_(0.02)
    x = _features()
    assert model.bounded_residual(x).abs().max() <= model.residual_bound
    model(x).sum().backward()
    assert model.scalar_base.weight.grad is not None
    assert model.residual_head[-1].weight.grad is not None
    assert model.siglip_branch[1].weight.grad is not None
    assert model.clip_branch[1].weight.grad is not None


def test_f5_uses_immutable_feature_order_and_has_no_dataset_input():
    siglip = torch.full((2, 768), 3.0)
    clip = torch.full((2, 768), 5.0)
    scalars = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    features = flatten_features(siglip, clip, scalars)
    model = F5R()
    f2 = RegularizedLinear()
    model.initialize_from_f2(f2.state_dict())
    assert torch.equal(model(features), f2(features))
    with pytest.raises(TypeError):
        model(features, dataset_ids=["a", "b"])


def test_f5_scalar_base_uses_joined_ten_contract_order():
    model = F5R()
    with torch.no_grad():
        model.scalar_base.weight.copy_(torch.arange(1, 9, dtype=torch.float32).view(1, 8))
        model.scalar_base.bias.zero_()
    row = torch.arange(1544, dtype=torch.float32).view(1, -1) / 1544
    ordered_scalars = row[:, list(SCALAR_INDICES)]
    expected = torch.sigmoid((ordered_scalars * torch.arange(1, 9)).sum(dim=-1))
    torch.testing.assert_close(model(row), expected)


def test_f5_initialization_is_deterministic():
    torch.manual_seed(818)
    first = F5G()
    torch.manual_seed(818)
    second = F5G()
    assert all(torch.equal(first.state_dict()[key], value)
               for key, value in second.state_dict().items())
    assert first.architecture_config() == second.architecture_config()


def test_optional_technical_head_is_isolated_from_aesthetic_logit():
    torch.manual_seed(113)
    aesthetic_only = F5R()
    torch.manual_seed(113)
    multitask = F5R(technical_head=True)
    assert torch.equal(aesthetic_only(_features()), multitask(_features()))
    with pytest.raises(RuntimeError, match="technical head is disabled"):
        aesthetic_only.forward_with_aux(_features())
    q_aes, q_tech = multitask.forward_with_aux(_features())
    assert q_aes.shape == q_tech.shape == (8,)
    q_tech.sum().backward()
    assert multitask.technical_head.weight.grad is not None
    assert multitask.scalar_base.weight.grad is None
    assert multitask.residual_head[-1].weight.grad is None


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="PyTorch MPS unavailable")
@pytest.mark.parametrize("model_type", [F5R, F5G, F5X])
def test_cpu_mps_numerical_parity_when_available(model_type):
    torch.manual_seed(72)
    cpu = model_type().eval()
    mps = model_type().eval()
    mps.load_state_dict(cpu.state_dict())
    with torch.no_grad():
        expected = cpu(_features())
        actual = mps.to("mps")(_features().to("mps")).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
