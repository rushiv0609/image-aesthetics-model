import json, sqlite3
import numpy as np
import torch
from safetensors.numpy import save_file
from image_aesthetics.data.training_dataset import CachedAVADataset
from image_aesthetics.models.siglip_head.models import build_head
from image_aesthetics.models.siglip_head.losses import distribution_loss
from image_aesthetics.evaluation.metrics import metric_report

def test_architectures_and_budget():
    assert build_head('A0').param_count < 1_000_000
    assert build_head('A1').param_count < 1_000_000
    assert build_head('A2').param_count < 1_000_000
    assert any(isinstance(x,torch.nn.LayerNorm) for x in build_head('A2').net)
    assert [x.out_features for x in build_head('A2').net if isinstance(x,torch.nn.Linear)] == [512,128,10]

def test_loss_gradients_and_components():
    logits=torch.randn(4,10,requires_grad=True); target=torch.zeros(4,10); target[:,4]=1
    loss, parts=distribution_loss(logits,target); loss.backward()
    assert loss.item()>0 and logits.grad is not None and set(parts)=={'emd','huber'}

def test_metrics_perfect():
    t=np.eye(10,dtype=float)[[1,3,5,7]]
    m=metric_report(t,t,pair_samples=1000)
    assert m['spearman'] > .99 and m['pairwise_accuracy'] > .99 and m['mae'] < 1e-8

def test_dataset_join_and_label_normalization(tmp_path):
    cache=tmp_path/'cache'; cache.mkdir(); arr=np.zeros((2,768),np.float32); arr[:,0]=1
    shard=cache/'shard-00000.safetensors'; save_file({'embeddings':arr},str(shard))
    db=sqlite3.connect(cache/'work.sqlite3'); db.executescript('CREATE TABLE items(image_id TEXT,shard_id INTEGER,shard_row INTEGER,state TEXT); CREATE TABLE shards(shard_id INTEGER,path TEXT,sha256 TEXT);')
    db.execute("insert into items values('a',0,0,'completed')"); db.execute("insert into items values('b',0,1,'completed')"); db.execute("insert into shards values(0,?, '')",(str(shard),)); db.commit(); db.close()
    p=tmp_path/'train.jsonl'; p.write_text(json.dumps({'image_id':'a','vote_counts':[0,1,0,0,0,0,0,0,0,0]})+'\n')
    ds=CachedAVADataset(p,cache); x,y,i=ds[0]; assert x.shape==(768,) and torch.isclose(y.sum(),torch.tensor(1.)) and i=='a'

def test_locked_guard(tmp_path):
    p=tmp_path/'locked-test.jsonl'; p.write_text('')
    try: CachedAVADataset(p,tmp_path)
    except PermissionError: return
    assert False
