"""Training/evaluation loop for leakage-safe F1-F4 fusion baselines."""
from __future__ import annotations
import json, os, random, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from image_aesthetics.data.fusion_training_dataset import DatasetBalancedBatchSampler
from image_aesthetics.models.fusion.losses import calibrated_huber_loss, pairwise_ranking_loss

def _atomic_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,indent=2,sort_keys=True)); os.replace(tmp,path)

def _atomic_torch(path: Path, value: Any):
    tmp=path.with_suffix(path.suffix+'.tmp'); torch.save(value,tmp); os.replace(tmp,path)

def _rank(x):
    x=np.asarray(x); order=np.argsort(x,kind='mergesort'); result=np.empty(len(x),float); result[order]=np.arange(len(x),dtype=float)
    for value in np.unique(x):
        indexes=np.flatnonzero(x==value)
        if len(indexes)>1: result[indexes]=result[indexes].mean()
    return result

def _metrics(pred: list[float], target: list[float]):
    p=np.asarray(pred,float); y=np.asarray(target,float)
    if len(p)<2 or p.std()==0 or y.std()==0: spear=0.0
    else: spear=float(np.corrcoef(_rank(p),_rank(y))[0,1])
    rng=np.random.default_rng(20260920); n=len(p); m=min(100000,n*(n-1)//2)
    if m:
        i=rng.integers(0,n,m); j=rng.integers(0,n,m); keep=(i!=j)&(y[i]!=y[j]); pair=float((np.sign(p[i]-p[j])[keep]==np.sign(y[i]-y[j])[keep]).mean()) if keep.any() else 0.0
    else: pair=0.0
    out={'spearman':spear,'pairwise_accuracy':pair,'mae':float(np.mean(abs(p-y))) if len(p) else 0.0,'rmse':float(np.sqrt(np.mean((p-y)**2))) if len(p) else 0.0}
    for q in (.01,.05,.10):
        k=max(1,int(np.ceil(len(p)*q))); a=np.argsort(-p)[:k]; b=np.argsort(-y)[:k]
        out[f'top{int(q*100)}_lift']=float(y[a].mean()-y.mean()) if len(p) else 0.0
        out[f'top{int(q*100)}_overlap']=float(len(set(a)&set(b))/k) if len(p) else 0.0
    return out

def _collect(model, loader, device, calibrators=None):
    model.eval(); all_pred=[]; all_target=[]; by={}; losses=[]
    calibrated_by={}
    with torch.no_grad():
        for batch in loader:
            x=batch['features'].to(device).float(); y=batch['target'].to(device).float(); pred=model(x)
            if pred.ndim>1: pred=pred.mean(-1)
            all_pred.extend(pred.cpu().tolist()); all_target.extend(y.cpu().tolist())
            for ds,p,t in zip(batch['dataset_id'],pred.cpu().tolist(),y.cpu().tolist()): by.setdefault(str(ds),[[],[]]); by[str(ds)][0].append(p); by[str(ds)][1].append(t)
            if calibrators is not None:
                native=batch['native_target'].to(device).float(); calibrated=calibrators(pred,list(batch['dataset_id']))
                losses.append(float(torch.nn.functional.huber_loss(calibrated,native)))
                for ds,p,t in zip(batch['dataset_id'],calibrated.cpu().tolist(),native.cpu().tolist()): calibrated_by.setdefault(str(ds),[[],[]]); calibrated_by[str(ds)][0].append(p); calibrated_by[str(ds)][1].append(t)
            else:
                losses.append(float(torch.nn.functional.huber_loss(pred,y)))
    report={'macro':{},'overall':_metrics(all_pred,all_target),'loss':float(np.mean(losses)) if losses else 0.0,'per_dataset':{ds:_metrics(v[0],v[1]) for ds,v in sorted(by.items())}}
    if report['per_dataset']:
        for key in report['overall']:
            vals=[v[key] for v in report['per_dataset'].values() if key in v]; report['macro'][key]=float(np.mean(vals)) if vals else 0.0
    if calibrated_by:
        report['native_calibration']={ds:{'mae':float(np.mean(np.abs(np.asarray(v[0])-np.asarray(v[1])))),'rmse':float(np.sqrt(np.mean((np.asarray(v[0])-np.asarray(v[1]))**2)))} for ds,v in sorted(calibrated_by.items())}
    return report

def train_fusion(model, train_dataset, validation_dataset, output_dir: Path, *, config: dict[str,Any], epochs=50, batch_size=256, seed=20260920, device=None, patience=7, min_epochs=8, ranking_weight=.1, calibrators=None, technical_loader=None, resume: Path | None = None):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device=torch.device(device or ('mps' if torch.backends.mps.is_available() else 'cpu')); model.to(device); calibrators=calibrators.to(device) if calibrators is not None else None; out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    sampler=DatasetBalancedBatchSampler(train_dataset,batch_size,seed=seed); train_loader=DataLoader(train_dataset,batch_sampler=sampler,num_workers=0); val_loader=DataLoader(validation_dataset,batch_size=batch_size,shuffle=False,num_workers=0)
    params=[p for p in model.parameters() if p.requires_grad] + ([p for p in calibrators.parameters() if p.requires_grad] if calibrators is not None else []); opt=torch.optim.AdamW(params,lr=3e-4,weight_decay=1e-4) if params else None; total=max(1,epochs*len(train_loader)); warm=max(1,int(.05*total)); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,total-warm)) if opt else None
    status=out/'training-status.json'; metrics_path=out/'epoch-metrics.jsonl'; ck=out/'checkpoints'; ck.mkdir(exist_ok=True); best=-1e9; patience_best=-1e9; bad=0; best_epoch=0; step=0; start=0; t0=time.time()
    if resume:
        rp=Path(resume); meta=json.loads((rp/'checkpoint.json').read_text())
        if meta.get('identity') != config.get('identity'): raise ValueError('checkpoint identity mismatch')
        state=torch.load(rp/'last.pt',map_location='cpu',weights_only=False); model.load_state_dict(state['model']);
        if calibrators is not None:
            if 'calibrators' not in state: raise ValueError('checkpoint is missing calibration heads')
            calibrators.load_state_dict(state['calibrators'])
        if opt: opt.load_state_dict(state['optimizer']); sched.load_state_dict(state['scheduler'])
        if opt:
            for optimizer_state in opt.state.values():
                for key,value in optimizer_state.items():
                    if torch.is_tensor(value): optimizer_state[key]=value.to(device)
        start=int(state['epoch'])+1; best=float(meta['best']); patience_best=float(meta.get('patience_best',best)); best_epoch=int(meta['best_epoch']); bad=int(meta['bad']); step=(start)*len(train_loader)
        torch.set_rng_state(state['rng_torch']); np.random.set_state(state['rng_numpy']); random.setstate(state['rng_python'])
        source_metrics=rp.parent/'epoch-metrics.jsonl'
        if source_metrics.exists() and not metrics_path.exists(): metrics_path.write_text(source_metrics.read_text())
    _atomic_json(status,{'state':'running','epoch':0,'batch':0,'batches_total':len(train_loader),'best_epoch':0,'patience_remaining':patience,'config':config})
    for epoch in range(start,epochs):
        sampler.set_epoch(epoch); model.train(); train_losses=[]
        for bi,batch in enumerate(train_loader):
            x=batch['features'].to(device).float(); y=batch['target'].to(device).float(); pred=model(x); pred=pred.mean(-1) if pred.ndim>1 else pred
            if calibrators is not None:
                loss,_=calibrated_huber_loss(pred,batch['native_target'].to(device).float(),list(batch['dataset_id']),calibrators)
            else:
                loss=torch.nn.functional.huber_loss(pred,y)
            if ranking_weight: loss=loss+ranking_weight*pairwise_ranking_loss(pred,y,groups=batch.get('pairing_group'))
            if hasattr(model,'regularization_loss'): loss=loss+model.regularization_loss()
            # Optional SPAQ adapter: callers may provide a callable returning a
            # scalar auxiliary loss for the current batch.  The baseline models
            # do not require or assume a technical head.
            if callable(technical_loader):
                auxiliary=technical_loader(batch=batch, model=model, device=device)
                if auxiliary is not None: loss=loss+auxiliary
            if opt:
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1
                if step>warm: sched.step()
            else: step+=1
            train_losses.append(float(loss.detach())); _atomic_json(status,{'state':'running','epoch':epoch+1,'batch':bi+1,'batches_total':len(train_loader),'loss':float(np.mean(train_losses)),'lr':opt.param_groups[0]['lr'] if opt else 0.0,'best_epoch':best_epoch,'patience_remaining':patience-bad,'global_step':step,'elapsed_seconds':time.time()-t0})
        report=_collect(model,val_loader,device,calibrators); score=report['macro'].get('spearman',report['overall']['spearman']); row={'epoch':epoch+1,'train_loss':float(np.mean(train_losses)),'val_loss':report['loss'],'val':report,'lr':opt.param_groups[0]['lr'] if opt else 0.0,'elapsed_seconds':time.time()-t0}
        with metrics_path.open('a') as f: f.write(json.dumps(row)+'\n')
        improved=score>best
        if improved: best=score; best_epoch=epoch+1
        if score>patience_best+0.001: patience_best=score; bad=0
        else: bad+=1
        state={'model':model.state_dict(),'calibrators':calibrators.state_dict() if calibrators is not None else None,'optimizer':opt.state_dict() if opt else {},'scheduler':sched.state_dict() if sched else {},'epoch':epoch,'best':best,'patience_best':patience_best,'best_epoch':best_epoch,'bad':bad,'rng_torch':torch.get_rng_state(),'rng_numpy':np.random.get_state(),'rng_python':random.getstate()}; _atomic_torch(ck/'last.pt',state)
        if improved: _atomic_torch(ck/'best.pt',state)
        _atomic_json(ck/'checkpoint.json',{'identity':config.get('identity'),'epoch':epoch,'best':best,'patience_best':patience_best,'best_epoch':best_epoch,'bad':bad})
        _atomic_json(status,{'state':'running','epoch':epoch+1,'batch':len(train_loader),'batches_total':len(train_loader),'best_epoch':best_epoch,'best_macro_spearman':best,'patience_remaining':patience-bad,'global_step':step,'last_metrics':row,'elapsed_seconds':time.time()-t0})
        if epoch+1>=min_epochs and bad>=patience: break
    rows=[json.loads(x) for x in metrics_path.read_text().splitlines() if x.strip()]; best_row=max(rows,key=lambda r:r['val']['macro'].get('spearman',-1)); completed_epoch=(epoch+1) if 'epoch' in locals() else start; _atomic_json(status,{'state':'completed','epoch':completed_epoch,'best_epoch':best_epoch,'best_macro_spearman':best,'best_metrics':best_row,'final_metrics':rows[-1],'reason_stopped':'early_stopping' if completed_epoch<epochs else 'max_epochs','global_step':step,'checkpoint_last':str(ck/'last.pt'),'checkpoint_best':str(ck/'best.pt'),'elapsed_seconds':time.time()-t0})
    return json.loads(status.read_text())
