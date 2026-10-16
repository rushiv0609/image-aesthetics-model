#!/usr/bin/env python3
"""Train one F1-F4 fusion baseline on a prepared development fold."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[2]/'src'))
from image_aesthetics.data.fusion_training_dataset import FusionTrainingDataset
from image_aesthetics.data.normalization import FeatureNormalizer
from image_aesthetics.models.fusion import DatasetCalibrationHeads, build_baseline
from image_aesthetics.training.fusion_trainer import train_fusion

def parse_join(values):
    result={}
    for value in values:
        if '=' not in value: raise argparse.ArgumentTypeError('join must be dataset_id=join_dir')
        name,path=value.split('=',1); result[name]=Path(path)
    if not result: raise argparse.ArgumentTypeError('at least one --join is required')
    return result

def balanced_limit(dataset, limit):
    if not limit or limit >= len(dataset.samples): return
    grouped={}
    for sample in dataset.samples: grouped.setdefault(sample[0],[]).append(sample)
    per_dataset=max(1,limit//len(grouped)); selected=[]
    for name in sorted(grouped): selected.extend(grouped[name][:per_dataset])
    remainder=limit-len(selected)
    if remainder:
        leftovers=[sample for name in sorted(grouped) for sample in grouped[name][per_dataset:]]
        selected.extend(leftovers[:remainder])
    dataset.samples=selected

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',choices=['F0','F1','F2','F3','F4'],required=True); p.add_argument('--f1-mode',choices=['individual','mean','median'],default='mean'); p.add_argument('--f1-signal',choices=['a1_siglip','laion_clip','nima','musiq_aesthetic','musiq_technical','sharpness','contrast','colorfulness']); p.add_argument('--split-manifest',type=Path,required=True); p.add_argument('--join',action='append',required=True,help='dataset_id=joined-ten directory'); p.add_argument('--fold',type=int,required=True,choices=range(5)); p.add_argument('--seed',type=int,default=20260920); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--epochs',type=int,default=50); p.add_argument('--batch-size',type=int,default=256); p.add_argument('--num-workers',type=int,default=0); p.add_argument('--limit',type=int,default=None); p.add_argument('--device',default=None); p.add_argument('--normalize',action='store_true'); p.add_argument('--normalizer-path',type=Path); p.add_argument('--resume',type=Path)
    args=p.parse_args(); joins=parse_join(args.join)
    train_ds=FusionTrainingDataset(args.split_manifest,joins,partition='train',validation_fold=args.fold); val_ds=FusionTrainingDataset(args.split_manifest,joins,partition='validation',validation_fold=args.fold)
    if args.limit:
        balanced_limit(train_ds,args.limit); balanced_limit(val_ds,max(len(joins),args.limit//5))
    if args.normalize:
        if args.model in {'F0','F1'}: raise ValueError('F0/F1 must evaluate the raw pretrained scalar scores')
        stats_path=args.normalizer_path or args.output_dir/('normalization-smoke.json' if args.limit else 'normalization.json')
        expected=train_ds.normalization_identity()
        norm=(FeatureNormalizer.from_file(stats_path,expected_identity=expected) if stats_path.exists()
              else train_ds.fit_normalizer(stats_path)); train_ds=train_ds.normalized(norm); val_ds=val_ds.normalized(norm)
    ranges={}
    source=train_ds.dataset if hasattr(train_ds,'dataset') else train_ds
    for dataset_id,_,row,_ in source.samples:
        bounds=tuple(map(float,row['native_score_range']))
        if dataset_id in ranges and ranges[dataset_id] != bounds: raise ValueError(f'inconsistent native range for {dataset_id}')
        ranges[dataset_id]=bounds
    model=build_baseline(args.model,**({'mode':args.f1_mode,'signal':args.f1_signal} if args.model=='F1' else {})); calibrators=DatasetCalibrationHeads(ranges); identity=hashlib.sha256(json.dumps({'model':args.model,'f1_mode':args.f1_mode,'f1_signal':args.f1_signal,'seed':args.seed,'fold':args.fold,'split':source.normalization_identity(),'normalize':args.normalize,'limit':args.limit,'calibration_ranges':ranges},sort_keys=True,default=str).encode()).hexdigest(); cfg={'model':args.model,'f1_mode':args.f1_mode,'f1_signal':args.f1_signal,'seed':args.seed,'fold':args.fold,'identity':identity,'batch_size':args.batch_size,'num_workers':args.num_workers,'epochs':args.epochs,'locked_test_accessed':False,'calibration_ranges':ranges,'limit':args.limit}
    args.output_dir.mkdir(parents=True,exist_ok=True); (args.output_dir/'config.json').write_text(json.dumps(cfg,indent=2,sort_keys=True)); print(f'TRAINING_BEGIN {args.model} fold={args.fold} output={args.output_dir}',flush=True)
    print(train_fusion(model,train_ds,val_ds,args.output_dir,config=cfg,epochs=args.epochs,batch_size=args.batch_size,seed=args.seed,device=args.device,calibrators=calibrators,resume=args.resume,num_workers=args.num_workers),flush=True)
if __name__=='__main__': main()
