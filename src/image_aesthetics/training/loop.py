"""Reproducible cached-feature training loop with resumable checkpoints."""
from __future__ import annotations
import json, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from safetensors.torch import save_file, load_file
from image_aesthetics.evaluation.metrics import metric_report
from image_aesthetics.models.siglip_head.losses import distribution_loss

def atomic_json(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)); os.replace(tmp,path)

def atomic_weights(path, state):
    tmp=path.with_suffix(path.suffix+'.tmp'); save_file(state,str(tmp)); os.replace(tmp,path)

def train(model, train_ds, val_ds, out_dir, *, config, device=None, epochs=50, batch_size=512, patience=7, min_epochs=8, seed=20260920, resume=None):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device=torch.device(device or ('mps' if torch.backends.mps.is_available() else 'cpu')); model.to(device)
    out=Path(out_dir); ck=out/'checkpoints'; ck.mkdir(parents=True,exist_ok=True)
    generator=torch.Generator().manual_seed(seed)
    tr=DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=0,generator=generator); va=DataLoader(val_ds,batch_size=batch_size,shuffle=False,num_workers=0)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4); total=max(1,epochs*len(tr)); warm=max(1,int(.05*total))
    def lrfn(step): return min(1.,(step+1)/warm)*(.5*(1+np.cos(np.pi*max(0,step-warm)/(max(1,total-warm)))))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lrfn); best=-1e9; patience_best=-1e9; best_epoch=0; bad=0; start=0; t0=time.time()
    if resume:
        rp=Path(resume); meta=json.loads((rp/'checkpoint.json').read_text())
        if meta.get('identity') != config.get('identity'): raise ValueError('checkpoint identity mismatch')
        model.load_state_dict(load_file(str(rp/'last.safetensors')))
        state=torch.load(rp/'optimizer.pt',map_location='cpu'); opt.load_state_dict(state['optimizer']); sched.load_state_dict(state['scheduler'])
        start=int(meta['epoch'])+1; best=float(meta['best']); patience_best=float(meta.get('patience_best',best)); best_epoch=int(meta['best_epoch']); bad=int(meta['bad'])
    status=out/'training-status.json'; metrics=out/'epoch-metrics.jsonl'
    atomic_json(status,{"state":"running","epoch":0,"total_epochs":epochs,"batch":0,"batches_total":len(tr),"best_epoch":0,"patience_remaining":patience})
    for epoch in range(start,epochs):
        model.train(); losses=[]; emds=[]; hubs=[]
        for bi,(x,y,_) in enumerate(tr):
            x=x.to(device); y=y.to(device); opt.zero_grad(set_to_none=True); loss,parts=distribution_loss(model(x),y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step(); losses.append(float(loss.detach())); emds.append(parts['emd']); hubs.append(parts['huber'])
            atomic_json(status,{"state":"running","epoch":epoch+1,"total_epochs":epochs,"batch":bi+1,"batches_total":len(tr),"loss":float(np.mean(losses)),"emd":float(np.mean(emds)),"huber":float(np.mean(hubs)),"lr":opt.param_groups[0]['lr'],"best_epoch":best_epoch,"patience_remaining":patience-bad,"elapsed_seconds":time.time()-t0})
        model.eval(); pp=[]; yy=[]; val_loss=[]
        with torch.no_grad():
            for x,y,_ in va:
                logits=model(x.to(device)); l,_=distribution_loss(logits,y.to(device)); val_loss.append(float(l)); pp.append(logits.softmax(-1).cpu().numpy()); yy.append(y.numpy())
        report=metric_report(np.concatenate(pp),np.concatenate(yy)); score=report['spearman']; rec={"epoch":epoch+1,"train_loss":float(np.mean(losses)),"train_emd":float(np.mean(emds)),"train_huber":float(np.mean(hubs)),"val_loss":float(np.mean(val_loss)),**{f"val_{k}":v for k,v in report.items()},"lr":opt.param_groups[0]['lr'],"elapsed_seconds":time.time()-t0}
        with metrics.open('a') as f: f.write(json.dumps(rec)+'\n')
        atomic_weights(ck/'last.safetensors',{k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()})
        if score > best:
            best=score; best_epoch=epoch+1; atomic_weights(ck/'best.safetensors',{k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()})
        if score > patience_best + .001: patience_best=score; bad=0
        else: bad+=1
        state={"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"epoch":epoch,"best":best,"patience_best":patience_best,"best_epoch":best_epoch,"bad":bad,"rng_torch":torch.get_rng_state(),"rng_numpy":np.random.get_state(),"rng_python":random.getstate()}
        tmpopt=ck/'optimizer.pt.tmp'; torch.save(state,tmpopt); os.replace(tmpopt,ck/'optimizer.pt')
        atomic_json(ck/'checkpoint.json',{"identity":config['identity'],"epoch":epoch,"best":best,"patience_best":patience_best,"best_epoch":best_epoch,"bad":bad,"global_step":(epoch+1)*len(tr)})
        atomic_json(status,{"state":"running","epoch":epoch+1,"total_epochs":epochs,"batch":len(tr),"batches_total":len(tr),**rec,"best_epoch":best_epoch,"patience_remaining":patience-bad,"checkpoint_last":str(ck/'last.safetensors'),"checkpoint_best":str(ck/'best.safetensors'),"elapsed_seconds":time.time()-t0})
        if epoch+1>=min_epochs and bad>=patience: break
    rows=[json.loads(x) for x in metrics.read_text().splitlines() if x.strip()]
    best_row=max(rows,key=lambda x:x.get('val_spearman',-1)) if rows else {}
    atomic_json(status,{"state":"completed","epoch":epoch+1,"total_epochs":epochs,"best_epoch":best_epoch,"best_spearman":best,"best_metrics":best_row,"final_metrics":rows[-1] if rows else {},"reason_stopped":"early_stopping" if epoch+1<epochs else "max_epochs","global_step":(epoch+1)*len(tr),"batches_total":len(tr),"checkpoint_last":str(ck/'last.safetensors'),"checkpoint_best":str(ck/'best.safetensors'),"elapsed_seconds":time.time()-t0})
    return json.loads(status.read_text())
