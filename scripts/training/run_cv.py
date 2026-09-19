#!/usr/bin/env python3
"""Run development-only grouped CV for A1/A2; never loads locked-test."""
from __future__ import annotations
import argparse, json, os, time, hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[2]/'src'))
import numpy as np
from image_aesthetics.data.training_dataset import CachedAVADataset
from image_aesthetics.models.siglip_head.models import build_head
from image_aesthetics.training.loop import train, atomic_json

SEEDS=(20260920,20260921,20260922)

def main():
 p=argparse.ArgumentParser(); p.add_argument('--data-root',default='/Users/rushivyas/datasets/image-aesthetics/ava'); p.add_argument('--output-root',default='/Users/rushivyas/datasets/image-aesthetics/ava/experiments/siglip2-aesthetic-cv-v1'); p.add_argument('--epochs',type=int,default=50); p.add_argument('--batch-size',type=int,default=512); p.add_argument('--device',default=None); args=p.parse_args()
 root=Path(args.data_root); split=root/'prepared/v1/splits'; cache=root/'embeddings/siglip2-base-patch16-224/full-v1'; out=Path(args.output_root); out.mkdir(parents=True,exist_ok=True)
 # Explicitly read only development + CV manifests.  No locked-test path is opened.
 labels=[json.loads(x) for x in (split/'development.jsonl').read_text().splitlines() if x.strip()]
 folds={str(r['image_id']):int(r['fold']) for r in (json.loads(x) for x in (split/'cv-assignments.jsonl').read_text().splitlines() if x.strip())}
 if set(folds)!={str(r['image_id']) for r in labels}: raise ValueError('CV assignment/development IDs mismatch')
 manifest_dir=out/'manifests'; manifest_dir.mkdir(exist_ok=True)
 aggregate={'state':'running','root':str(out),'models':['A1','A2'],'seeds':list(SEEDS),'folds':5,'completed':0,'total':30,'runs':[]}; atomic_json(out/'experiment-status.json',aggregate)
 for fold in range(5):
  valid=[r for r in labels if folds[str(r['image_id'])]==fold]; train_rows=[r for r in labels if folds[str(r['image_id'])]!=fold]
  vp=manifest_dir/f'fold-{fold}-validation.jsonl'; tp=manifest_dir/f'fold-{fold}-train.jsonl'
  if not vp.exists(): vp.write_text(''.join(json.dumps(r)+'\n' for r in valid))
  if not tp.exists(): tp.write_text(''.join(json.dumps(r)+'\n' for r in train_rows))
  for seed in SEEDS:
   for arch in ('A1','A2'):
    run=out/f'{arch}-fold{fold}-seed{seed}'; cfgbase={'architecture':arch,'fold':fold,'seed':seed,'cache_manifest_sha256':'90e691fdc96473decca29075193b62e30d8767e9c3d6c8ca8adb43a2d32358c1','cv_assignments_sha256':'97d2990cbf7778d5790aa74286a445b21d020478029088f18aa141a9296a8cfc','loss':'squared_emd+0.25_huber','optimizer':'AdamW','schedule':'warmup5pct_cosine'}
    ident=hashlib.sha256(json.dumps(cfgbase,sort_keys=True).encode()).hexdigest(); cfg={**cfgbase,'identity':ident,'batch_size':args.batch_size,'epochs':args.epochs,'param_count':build_head(arch).param_count}
    status=run/'training-status.json'
    if status.exists():
     try:
      old=json.loads(status.read_text())
      if old.get('state')=='completed' and json.loads((run/'config.json').read_text()).get('identity')==ident:
       aggregate['completed']+=1; aggregate['runs'].append({'run':run.name,'status':'skipped'}); atomic_json(out/'experiment-status.json',aggregate); continue
     except Exception: pass
    if status.exists():
     try:
      if json.loads(status.read_text()).get('state') != 'completed':
       preserved=run.with_name(run.name+f'-interrupted-{int(time.time())}'); os.rename(run,preserved)
     except Exception: pass
    run.mkdir(parents=True,exist_ok=True); (run/'config.json').write_text(json.dumps(cfg,indent=2))
    tr=CachedAVADataset(tp,cache,max_shards=64); va=CachedAVADataset(vp,cache,max_shards=64)
    result=None; last_exc=None
    for attempt in range(1,4):
     try:
      result=train(build_head(arch),tr,va,run,config=cfg,device=args.device,epochs=args.epochs,batch_size=args.batch_size,seed=seed)
      vals=[result.get('best_spearman'),result.get('elapsed_seconds')]
      if not all(v is not None and np.isfinite(v) for v in vals): raise ValueError('non-finite or incomplete training status')
      break
     except Exception as exc:
      last_exc=exc; atomic_json(run/'failure.json',{'attempt':attempt,'error':repr(exc)})
      if attempt<3: time.sleep(min(30,5*attempt))
    if result is None:
     aggregate['runs'].append({'run':run.name,'status':'failed','error':repr(last_exc)}); aggregate['state']='failed'; atomic_json(out/'experiment-status.json',aggregate); raise last_exc
    aggregate['completed']+=1; aggregate['runs'].append({'run':run.name,'status':'completed','best_spearman':result.get('best_spearman'),'duration':result.get('elapsed_seconds')})
    atomic_json(out/'experiment-status.json',aggregate)
 aggregate['state']='completed'; atomic_json(out/'experiment-status.json',aggregate)
 # Aggregate numerical metrics by architecture across all fold/seed completed runs.
 rows=[]
 for run in out.glob('A?-fold*-seed*/epoch-metrics.jsonl'):
  status=json.loads((run.parent/'training-status.json').read_text()); best=status.get('best_metrics',{})
  if best: rows.append({'run':run.parent.name,'architecture':run.parent.name.split('-')[0],**best})
 report={'runs':rows,'summary':{},'paired':{}}
 for arch in ('A1','A2'):
  ar=[r for r in rows if r['architecture']==arch]; report['summary'][arch]={}
  for key in ('val_spearman','val_pairwise_accuracy','val_emd','val_mae','val_rmse','val_top1_lift','val_top5_lift','val_top10_lift','elapsed_seconds','epoch'):
   vals=np.asarray([r[key] for r in ar],float); report['summary'][arch][key]={'mean':float(vals.mean()),'std':float(vals.std(ddof=1)),'ci95':float(1.96*vals.std(ddof=1)/np.sqrt(len(vals)))}
 pairs={r['run'].replace('A1-',''):r for r in rows if r['architecture']=='A1'}
 for r in rows:
  if r['architecture']=='A2':
   key=r['run'].replace('A2-',''); a=pairs.get(key)
   if a: report['paired'][key]={'spearman_delta_A2_minus_A1':r['val_spearman']-a['val_spearman'],'pairwise_delta_A2_minus_A1':r['val_pairwise_accuracy']-a['val_pairwise_accuracy'],'emd_delta_A2_minus_A1':r['val_emd']-a['val_emd']}
 deltas=[v['spearman_delta_A2_minus_A1'] for v in report['paired'].values()]
 report['practical_conclusion']='A2 is preferred only if its paired CV gains justify its extra capacity; otherwise A1 is the simpler choice.'
 if deltas: report['paired_summary']={'n':len(deltas),'spearman_delta_mean':float(np.mean(deltas)),'spearman_delta_std':float(np.std(deltas,ddof=1)),'a2_wins':int(sum(x>0 for x in deltas))}
 atomic_json(out/'cv-aggregate-report.json',report)

if __name__=='__main__': main()
