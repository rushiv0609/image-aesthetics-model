import json
import torch
from torch.utils.data import Dataset
import pytest
from image_aesthetics.models.fusion import ScalarMLP
from image_aesthetics.training.fusion_trainer import train_fusion

class TinyFusion(Dataset):
    def __init__(self, n=12):
        self.samples=[]; self.x=torch.randn(n,1544)
        for i in range(n): self.samples.append(('a' if i%2==0 else 'b',i,{'group_id':str(i)},float(self.x[i,1536:].mean().sigmoid())) )
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        ds,_,row,target=self.samples[i]
        return {'features':self.x[i],'target':target,'native_target':target,'dataset_id':ds,'image_id':str(i),'group_id':str(i),'development_fold':0}

def test_fusion_trainer_status_metrics_and_resume(tmp_path):
    ds=TinyFusion(); out=tmp_path/'run'; cfg={'identity':'tiny'}
    result=train_fusion(ScalarMLP(),ds,ds,out,config=cfg,epochs=2,batch_size=2,device='cpu',min_epochs=1,patience=2)
    assert result['state']=='completed' and (out/'checkpoints/last.pt').exists() and (out/'checkpoints/best.pt').exists()
    assert 'per_dataset' in result['best_metrics']['val'] and result['global_step']>0
    resumed=train_fusion(ScalarMLP(),ds,ds,tmp_path/'resume',config=cfg,epochs=2,batch_size=2,device='cpu',min_epochs=1,resume=out/'checkpoints')
    assert resumed['state']=='completed'

def test_fusion_trainer_rejects_identity_mismatch(tmp_path):
    ds=TinyFusion(); out=tmp_path/'run'; train_fusion(ScalarMLP(),ds,ds,out,config={'identity':'a'},epochs=1,batch_size=2,device='cpu',min_epochs=1)
    with pytest.raises(ValueError): train_fusion(ScalarMLP(),ds,ds,tmp_path/'other',config={'identity':'b'},epochs=1,batch_size=2,device='cpu',resume=out/'checkpoints')

def test_optional_auxiliary_hook_is_supported(tmp_path):
    ds=TinyFusion(); seen=[]
    def hook(**kwargs):
        seen.append(kwargs['batch']['features'].shape[0]); return torch.zeros((),requires_grad=True)
    train_fusion(ScalarMLP(),ds,ds,tmp_path/'run',config={'identity':'aux'},epochs=1,batch_size=2,device='cpu',min_epochs=1,technical_loader=hook)
    assert seen
