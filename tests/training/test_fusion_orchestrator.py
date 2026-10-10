import json
from pathlib import Path
from scripts.training.run_fusion_baselines import matrix_specs, _identity, _matrix_status

def test_matrix_contains_direct_frozen_and_trainable_runs():
    specs=matrix_specs(range(5),(1,2,3))
    assert len(specs)==100
    assert sum(not s.trainable for s in specs)==55
    assert sum(s.trainable for s in specs)==45
    assert {s.f1_signal for s in specs if s.model=='F1' and s.f1_mode=='individual'} == {'laion_clip','a1_siglip','nima','musiq_aesthetic','musiq_technical','sharpness','contrast','colorfulness'}
    assert sum(s.model=='F0' for s in specs)==5

def test_identity_changes_with_fold_seed_and_normalizer():
    split=Path(__file__).resolve(); from scripts.training.run_fusion_baselines import RunSpec; s=RunSpec('x','F3',0,1); a,_=_identity(s,split,True,'a',None); b,_=_identity(s,split,True,'b',None)
    assert a != b

def test_status_accepts_structured_failures(tmp_path):
    specs=matrix_specs(range(1),(1,))[:2]
    value=_matrix_status(tmp_path/'status.json',specs,failed=[{'run_id':specs[0].run_id,'error':'boom'}])
    assert specs[0].run_id not in value['pending'] and specs[1].run_id in value['pending']
