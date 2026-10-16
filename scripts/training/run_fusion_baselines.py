#!/usr/bin/env python3
"""Plan or execute the resumable F0-F4 fusion baseline matrix.

Planning is the default and never opens a dataset.  Execution is intentionally
foreground-friendly, e.g. ``caffeinate -dimsu .venv/bin/python ... --execute``.
Only development rows are loaded; this command has no locked-test code path.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from image_aesthetics.data.fusion_training_dataset import FusionTrainingDataset
from image_aesthetics.data.normalization import FeatureNormalizer
from image_aesthetics.models.fusion import MOMENTS_F0_WEIGHTS, DatasetCalibrationHeads, build_baseline
from image_aesthetics.training.fusion_trainer import _atomic_json, _collect, train_fusion
from torch.utils.data import DataLoader

SCALARS=("laion_clip","a1_siglip","nima","musiq_aesthetic","musiq_technical","sharpness","contrast","colorfulness")
DEFAULT_SEEDS=(20260920,20260921,20260922)

@dataclass(frozen=True)
class RunSpec:
    run_id: str
    model: str
    fold: int
    seed: int | None
    f1_mode: str | None = None
    f1_signal: str | None = None
    trainable: bool = True

def matrix_specs(folds=range(5), seeds=DEFAULT_SEEDS):
    result=[]
    for fold in folds:
        result.append(RunSpec(f"F0-fold{fold}","F0",fold,None,trainable=False))
        for mode,signal,label in [("mean",None,"mean"),("median",None,"median")]+[("individual",s,s) for s in SCALARS]:
            result.append(RunSpec(f"F1-{label}-fold{fold}","F1",fold,None,mode,signal,False))
        for model in ("F2","F3","F4"):
            for seed in seeds: result.append(RunSpec(f"{model}-fold{fold}-seed{seed}",model,fold,seed))
    return result

def _identity(spec, split_path, normalize, stats_hash=None, limit=None):
    payload={'spec':asdict(spec),'split_sha256':hashlib.sha256(Path(split_path).read_bytes()).hexdigest(),'normalize':normalize,'normalization_stats_sha256':stats_hash,'limit':limit}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),payload

def _matrix_status(path, specs, *, current=None, completed=None, failed=None, started_at=None):
    completed=completed or []; failed=failed or []; failed_ids={item['run_id'] if isinstance(item,dict) else item for item in failed}; done=set(completed)|failed_ids
    pending=[s.run_id for s in specs if s.run_id not in done]
    value={'state':'running','started_at':started_at,'updated_at':time.time(),'current_run':current,'total':len(specs),'completed':completed,'failed':failed,'pending':pending}
    _atomic_json(path,value); return value

def _ranges(dataset):
    out={}
    source=dataset.dataset if hasattr(dataset,'dataset') else dataset
    for ds,_,row,_ in source.samples:
        bounds=tuple(map(float,row['native_score_range']))
        if ds in out and out[ds]!=bounds: raise ValueError(f'inconsistent native range for {ds}')
        out[ds]=bounds
    return out

def execute_matrix(args, specs):
    from scripts.training.train_fusion_baseline import balanced_limit
    out=Path(args.output_root); out.mkdir(parents=True,exist_ok=True); status_path=out/'matrix-status.json'; started=time.time(); completed=[]; failed=[]; summaries={}; fold_cache={}
    joins={name:Path(path) for name,path in (v.split('=',1) for v in args.join)}
    for spec in specs[:args.max_runs or len(specs)]:
        run=out/spec.run_id; run.mkdir(parents=True,exist_ok=True); cfg_path=run/'config.json'; status_path_run=run/'training-status.json'; normalize=spec.model in {'F2','F3','F4'}
        stats_path=out/'normalization'/f'fold{spec.fold}.json'
        # Fit exactly one fold-local stats file, reused across models and seeds.
        if spec.fold not in fold_cache:
            base_train=FusionTrainingDataset(args.split_manifest,joins,partition='train',validation_fold=spec.fold); base_val=FusionTrainingDataset(args.split_manifest,joins,partition='validation',validation_fold=spec.fold)
            if args.limit:
                balanced_limit(base_train,args.limit); balanced_limit(base_val,max(len(joins),args.limit//5))
            fold_cache[spec.fold]=(base_train,base_val)
        train_ds,val_ds=fold_cache[spec.fold]
        if normalize:
            expected=train_ds.normalization_identity()
            norm=FeatureNormalizer.from_file(stats_path,expected_identity=expected) if stats_path.exists() else train_ds.fit_normalizer(stats_path)
            train_ds=train_ds.normalized(norm); val_ds=val_ds.normalized(norm)
        stats_hash=hashlib.sha256(stats_path.read_bytes()).hexdigest() if normalize else None; ident,payload=_identity(spec,args.split_manifest,normalize,stats_hash,args.limit)
        cfg={'run_id':spec.run_id,'model':spec.model,'fold':spec.fold,'seed':spec.seed,'f1_mode':spec.f1_mode,'f1_signal':spec.f1_signal,'identity':ident,'identity_payload':payload,'epochs':args.epochs,'batch_size':args.batch_size,'num_workers':args.num_workers,'patience':args.patience,'min_epochs':args.min_epochs,'locked_test_accessed':False}
        if status_path_run.exists() and cfg_path.exists():
            try:
                if json.loads(status_path_run.read_text()).get('state')=='completed' and json.loads(cfg_path.read_text()).get('identity')==ident:
                    completed.append(spec.run_id); summaries[spec.run_id]=json.loads(status_path_run.read_text()).get('best_metrics',{}); _matrix_status(status_path,specs,current=spec.run_id,completed=completed,failed=failed,started_at=started); continue
            except Exception: pass
        if cfg_path.exists():
            try:
                old_identity=json.loads(cfg_path.read_text()).get('identity')
                if old_identity != ident:
                    failed.append({'run_id':spec.run_id,'error':'incompatible existing run identity','timestamp':time.time()}); _matrix_status(status_path,specs,current=spec.run_id,completed=completed,failed=failed,started_at=started); continue
            except Exception: pass
        cfg_path.write_text(json.dumps(cfg,indent=2,sort_keys=True)); _matrix_status(status_path,specs,current=spec.run_id,completed=completed,failed=failed,started_at=started)
        try:
            model=build_baseline(spec.model,**({'mode':spec.f1_mode,'signal':spec.f1_signal} if spec.model=='F1' else ({'weights':MOMENTS_F0_WEIGHTS} if spec.model=='F0' else {})))
            if not spec.trainable:
                report=_collect(model,DataLoader(val_ds,batch_size=args.batch_size,shuffle=False),torch.device(args.device or 'cpu'))
                summary={'state':'completed','run_id':spec.run_id,'best_metrics':report,'reason_stopped':'direct_evaluation','locked_test_accessed':False}
                _atomic_json(status_path_run,summary); summaries[spec.run_id]=report
                _atomic_json(run/'checkpoints'/'eval.json',{'identity':ident,'run_id':spec.run_id,'evaluation_only':True,'metrics':report})
            else:
                ranges=_ranges(train_ds); cal=DatasetCalibrationHeads(ranges)
                resume_dir=run/'checkpoints' if (run/'checkpoints/checkpoint.json').exists() and status_path_run.exists() else None
                result=train_fusion(model,train_ds,val_ds,run,config=cfg,epochs=args.epochs,batch_size=args.batch_size,seed=spec.seed,device=args.device,patience=args.patience,min_epochs=args.min_epochs,calibrators=cal,resume=resume_dir,num_workers=args.num_workers)
                summaries[spec.run_id]=result.get('best_metrics',{})
            completed.append(spec.run_id)
        except Exception as exc:
            failed.append({'run_id':spec.run_id,'error':repr(exc),'timestamp':time.time()}); _atomic_json(status_path_run,{'state':'failed','run_id':spec.run_id,'error':repr(exc),'locked_test_accessed':False});
        _matrix_status(status_path,specs,current=spec.run_id,completed=completed,failed=failed,started_at=started)
    aggregate={'state':'completed' if not failed else 'completed_with_failures','completed':completed,'failed':failed,'pending':[s.run_id for s in specs if s.run_id not in completed and not any(f.get('run_id')==s.run_id for f in failed)],'summaries':summaries,'updated_at':time.time()}; _atomic_json(out/'matrix-status.json',aggregate); _atomic_json(out/'aggregate-metrics.json',aggregate)
    return aggregate

def main():
    p=argparse.ArgumentParser(); p.add_argument('--execute',action='store_true'); p.add_argument('--smoke',action='store_true'); p.add_argument('--max-runs',type=int); p.add_argument('--data-root',type=Path,default=Path('/Users/rushivyas/datasets/image-aesthetics')); p.add_argument('--output-root',type=Path); p.add_argument('--split-manifest',type=Path); p.add_argument('--join',action='append'); p.add_argument('--folds',default='0,1,2,3,4'); p.add_argument('--seeds',default=','.join(map(str,DEFAULT_SEEDS))); p.add_argument('--epochs',type=int,default=50); p.add_argument('--patience',type=int,default=7); p.add_argument('--min-epochs',type=int,default=8); p.add_argument('--batch-size',type=int,default=256); p.add_argument('--num-workers',type=int,default=0); p.add_argument('--limit',type=int); p.add_argument('--device',default='cpu')
    args=p.parse_args()
    args.output_root=args.output_root or args.data_root/'fusion/experiments'/('baseline-matrix-v6-smoke' if args.smoke else 'baseline-matrix-v6')
    args.split_manifest=args.split_manifest or args.data_root/'fusion/prepared/v6/samples.jsonl'
    args.join=args.join or [f'{d}={args.data_root}/{d}/features/joined-ten-v1' for d in ('aadb','para','flickr_aes','tad66k')]
    args.folds=[int(x) for x in args.folds.split(',') if x!='']; args.seeds=tuple(int(x) for x in args.seeds.split(',') if x!='')
    if args.smoke: args.limit=args.limit or 40; args.epochs=min(args.epochs,2); args.patience=min(args.patience,1); args.min_epochs=min(args.min_epochs,1); args.max_runs=args.max_runs or 3
    specs=matrix_specs(args.folds,args.seeds)
    if not args.execute:
        args.output_root.mkdir(parents=True,exist_ok=True)
        _atomic_json(args.output_root/'matrix-status.json',{'state':'planned','created_at':time.time(),'current_run':None,'total':len(specs),'completed':[],'failed':[],'pending':[s.run_id for s in specs]})
        print(json.dumps({'output_root':str(args.output_root),'split_manifest':str(args.split_manifest),'runs':[asdict(s) for s in specs],'count':len(specs),'normalization_files':[str(args.output_root/'normalization'/f'fold{f}.json') for f in args.folds]},indent=2)); return
    result=execute_matrix(args,specs)
    print(json.dumps({'state':result['state'],'output_root':str(args.output_root),'completed':len(result['completed']),'failed':len(result['failed']),'pending':len(result['pending'])},indent=2))
if __name__=='__main__': main()
