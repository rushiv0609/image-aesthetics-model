import pytest
import torch

from image_aesthetics.models.fusion import *

def features(n=6): return torch.arange(n * FEATURE_DIM, dtype=torch.float32).reshape(n, FEATURE_DIM) / 10000

def test_feature_contract_order_and_dimensions():
    x=features(2); s,c,z=split_features(x)
    assert s.shape==(2,768) and c.shape==(2,768) and z.shape==(2,8)
    assert torch.equal(flatten_features(s,c,z),x)
    assert SCALAR_NAMES == ("a1_siglip","laion_clip","nima","musiq_aesthetic","musiq_technical","sharpness","contrast","colorfulness")
    with pytest.raises(ValueError): split_features(torch.zeros(2,10))

def test_f0_weights_follow_contract_and_f1_ensembles():
    x=flatten_features(torch.zeros(2,768),torch.zeros(2,768),torch.arange(8).float().repeat(2,1))
    m=WeightedScalarScore({name: i+1 for i,name in enumerate(SCALAR_NAMES)})
    assert torch.allclose(m(x),torch.tensor([168.,168.]))
    assert ScalarEnsemble("individual",signal="nima")(x).shape==(2,)
    assert torch.allclose(ScalarEnsemble("mean")(x),torch.tensor([3.5,3.5]))
    assert torch.allclose(ScalarEnsemble("median")(x),torch.tensor([3.5,3.5]))

def test_f2_f3_f4_shapes_and_budget():
    x=features()
    for model in (RegularizedLinear(),ScalarMLP(),FlatFeatureMLP()):
        assert model(x).shape==(6,)
    assert FlatFeatureMLP().param_count < 1_000_000
    assert RegularizedLinear().regularization_loss().item() >= 0

def test_factory():
    assert [type(build_baseline(x)).__name__ for x in ("F0","F1","F2","F3","F4")] == ["WeightedScalarScore","ScalarEnsemble","RegularizedLinear","ScalarMLP","FlatFeatureMLP"]

def test_monotonic_dataset_calibration_is_training_only():
    cal=DatasetCalibrationHeads({"aadb":(1,5),"para":(1,5)})
    q=torch.tensor([.1,.2,.1]); y=cal(q,["aadb","aadb","para"])
    assert y.shape==(3,) and cal.heads["aadb"].slope.item()>0
    assert cal(torch.tensor([.1,.9]),["aadb","aadb"])[1] > cal(torch.tensor([.1,.9]),["aadb","aadb"])[0]
    with pytest.raises(ValueError): cal(q,["unknown"]*3)

def test_calibrated_huber_ranking_and_technical_losses():
    cal=DatasetCalibrationHeads({"aadb":(1,5)})
    q=torch.tensor([.2,.8],requires_grad=True); target=torch.tensor([2.,4.])
    loss,pred=calibrated_huber_loss(q,target,["aadb","aadb"],cal); rank=pairwise_ranking_loss(q,target); tech=technical_auxiliary_loss(q,target/5)
    (loss+rank+tech).backward()
    assert loss.item()>=0 and rank.item()>=0 and q.grad is not None
    assert technical_auxiliary_loss(q.detach(),target,valid=torch.zeros(2,dtype=torch.bool)).item()==0
    assert pairwise_ranking_loss(torch.ones(2),torch.ones(2)).item()==0
    assert pairwise_ranking_loss(torch.tensor([0.,1.]),torch.tensor([0.,1.]),groups=['a','b']).item()==0
