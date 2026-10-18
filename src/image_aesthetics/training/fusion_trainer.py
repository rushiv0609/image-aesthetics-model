"""Training/evaluation loop for leakage-safe F1-F5 fusion baselines."""
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
    """Evaluate while avoiding one accelerator-to-host synchronization per batch.

    Validation metrics are host-side NumPy metrics, but a validation epoch can
    contain hundreds of small batches.  Keeping the compact prediction/target
    tensors on the selected device until the loader is exhausted makes that
    boundary explicit and is materially cheaper on MPS.  The result is exactly
    the same set of predictions and labels as the prior per-batch transfer.
    """
    model.eval(); pred_batches=[]; target_batches=[]; dataset_ids=[]
    calibrated_pred_batches=[]; native_target_batches=[]; loss_total=None; loss_count=0
    with torch.no_grad():
        for batch in loader:
            x=batch['features'].to(device=device,dtype=torch.float32); y=batch['target'].to(device=device,dtype=torch.float32); pred=model(x)
            if pred.ndim>1: pred=pred.mean(-1)
            pred_batches.append(pred.detach()); target_batches.append(y.detach()); dataset_ids.extend(map(str, batch['dataset_id']))
            if calibrators is not None:
                native=batch['native_target'].to(device=device,dtype=torch.float32); calibrated=calibrators(pred,list(batch['dataset_id']))
                batch_loss=torch.nn.functional.huber_loss(calibrated,native)
                calibrated_pred_batches.append(calibrated.detach()); native_target_batches.append(native.detach())
            else:
                batch_loss=torch.nn.functional.huber_loss(pred,y)
            loss_total=batch_loss.detach() if loss_total is None else loss_total+batch_loss.detach(); loss_count+=1
    if pred_batches:
        # One host transfer for normalized prediction/target pairs.
        normalized_pairs=torch.stack((torch.cat(pred_batches),torch.cat(target_batches)),dim=-1).cpu().tolist()
    else:
        normalized_pairs=[]
    by={}
    for ds,(prediction,target) in zip(dataset_ids,normalized_pairs):
        by.setdefault(ds,[[],[]]); by[ds][0].append(prediction); by[ds][1].append(target)
    calibrated_by={}
    if calibrated_pred_batches:
        # Calibration diagnostics use a second compact transfer only when
        # calibration heads are enabled.
        calibrated_pairs=torch.stack((torch.cat(calibrated_pred_batches),torch.cat(native_target_batches)),dim=-1).cpu().tolist()
        for ds,(prediction,target) in zip(dataset_ids,calibrated_pairs):
            calibrated_by.setdefault(ds,[[],[]]); calibrated_by[ds][0].append(prediction); calibrated_by[ds][1].append(target)
    all_pred=[pair[0] for pair in normalized_pairs]; all_target=[pair[1] for pair in normalized_pairs]
    report={'macro':{},'overall':_metrics(all_pred,all_target),'loss':float((loss_total/loss_count).cpu()) if loss_count else 0.0,'per_dataset':{ds:_metrics(v[0],v[1]) for ds,v in sorted(by.items())}}
    if report['per_dataset']:
        for key in report['overall']:
            vals=[v[key] for v in report['per_dataset'].values() if key in v]; report['macro'][key]=float(np.mean(vals)) if vals else 0.0
    if calibrated_by:
        report['native_calibration']={ds:{'mae':float(np.mean(np.abs(np.asarray(v[0])-np.asarray(v[1])))),'rmse':float(np.sqrt(np.mean((np.asarray(v[0])-np.asarray(v[1]))**2)))} for ds,v in sorted(calibrated_by.items())}
    return report

def train_fusion(model, train_dataset, validation_dataset, output_dir: Path, *, config: dict[str,Any], epochs=50, batch_size=256, seed=20260920, device=None, patience=7, min_epochs=8, ranking_weight=.1, ranking_min_target_gap: float = 0.0, calibrators=None, technical_loader=None, resume: Path | None = None, status_interval_batches: int = 25, num_workers: int = 0):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if status_interval_batches < 1: raise ValueError('status_interval_batches must be positive')
    if num_workers < 0: raise ValueError('num_workers must be non-negative')
    if ranking_min_target_gap < 0: raise ValueError('ranking_min_target_gap must be non-negative')
    device=torch.device(device or ('mps' if torch.backends.mps.is_available() else 'cpu')); model.to(device); calibrators=calibrators.to(device) if calibrators is not None else None; out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    sampler=DatasetBalancedBatchSampler(train_dataset,batch_size,seed=seed)
    # macOS workers are opt-in: their startup and per-worker shard caches cost
    # memory, but M5 measurements show that four persistent workers can remove
    # the dominant cached-feature loading bottleneck for 1,024-row batches.
    loader_options={'num_workers':num_workers}
    if num_workers:
        loader_options.update({'persistent_workers':True,'prefetch_factor':2})
    train_loader=DataLoader(train_dataset,batch_sampler=sampler,**loader_options)
    val_loader=DataLoader(validation_dataset,batch_size=batch_size,shuffle=False,**loader_options)
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
        sampler.set_epoch(epoch); model.train(); train_loss_total=None
        for bi,batch in enumerate(train_loader):
            x=batch['features'].to(device=device,dtype=torch.float32); y=batch['target'].to(device=device,dtype=torch.float32); pred=model(x); pred=pred.mean(-1) if pred.ndim>1 else pred
            if calibrators is not None:
                loss,_=calibrated_huber_loss(pred,batch['native_target'].to(device=device,dtype=torch.float32),list(batch['dataset_id']),calibrators)
            else:
                loss=torch.nn.functional.huber_loss(pred,y)
            if ranking_weight:
                # DatasetBalancedBatchSampler emits homogeneous batches.  In
                # that common case every pair is compatible, so avoid the
                # expensive Python O(B^2) compatibility matrix construction.
                pairing_groups=batch.get('pairing_group')
                groups=None if pairing_groups is not None and len(set(pairing_groups)) == 1 else pairing_groups
                loss=loss+ranking_weight*pairwise_ranking_loss(
                    pred,y,groups=groups,min_target_gap=ranking_min_target_gap)
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
            train_loss_total=loss.detach() if train_loss_total is None else train_loss_total+loss.detach()
            # Frequent JSON writes dominate tiny-model training and also force
            # a device synchronization to report the running loss.  Always
            # write the final batch of an epoch for accurate resumable state.
            if (bi+1) % status_interval_batches == 0 or bi+1 == len(train_loader):
                _atomic_json(status,{'state':'running','epoch':epoch+1,'batch':bi+1,'batches_total':len(train_loader),'loss':float((train_loss_total/(bi+1)).cpu()),'lr':opt.param_groups[0]['lr'] if opt else 0.0,'best_epoch':best_epoch,'patience_remaining':patience-bad,'global_step':step,'elapsed_seconds':time.time()-t0})
        train_loss=float((train_loss_total/len(train_loader)).cpu()) if train_loss_total is not None else 0.0
        report=_collect(model,val_loader,device,calibrators); score=report['macro'].get('spearman',report['overall']['spearman']); row={'epoch':epoch+1,'train_loss':train_loss,'val_loss':report['loss'],'val':report,'lr':opt.param_groups[0]['lr'] if opt else 0.0,'elapsed_seconds':time.time()-t0}
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
