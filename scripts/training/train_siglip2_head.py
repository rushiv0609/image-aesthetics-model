#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[2]/'src'))
from image_aesthetics.data.training_dataset import CachedAVADataset
from image_aesthetics.models.siglip_head.models import build_head
from image_aesthetics.training.loop import train

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--architecture',choices=['A0','A1','A2'],required=True)
    p.add_argument('--epochs',type=int,default=50); p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--data-root',default='/Users/rushivyas/datasets/image-aesthetics/ava')
    p.add_argument('--output-root',default='/Users/rushivyas/datasets/image-aesthetics/ava/experiments/siglip2-aesthetic')
    p.add_argument('--device',default=None); p.add_argument('--limit',type=int,default=None)
    args=p.parse_args()
    root=Path(args.data_root); split=root/'prepared/v1/splits'; cache=root/'embeddings/siglip2-base-patch16-224/full-v1'; run=Path(args.output_root)/(args.architecture+'-seed20260920')
    tr=CachedAVADataset(split/'train.jsonl',cache,max_shards=64,limit=args.limit); va=CachedAVADataset(split/'validation.jsonl',cache,max_shards=64,limit=args.limit)
    model=build_head(args.architecture)
    sm=json.loads((split/'split-metadata.json').read_text()); cm=json.loads((cache/'manifest.json').read_text())
    identity_payload={'arch':args.architecture,'seed':20260920,'cache_manifest_sha256':sm['cache_manifest_sha256'],'train_sha256':sm['outer']['train']['sha256'],'validation_sha256':sm['outer']['validation']['sha256'],'loss':'squared_emd+0.25_huber','optimizer':'AdamW','schedule':'warmup5pct_cosine'}
    identity=hashlib.sha256(json.dumps(identity_payload,sort_keys=True).encode()).hexdigest()
    cfg={**identity_payload,'identity':identity,'cache':str(cache),'split_root':str(split),'batch_size':args.batch_size,'epochs':args.epochs,'param_count':model.param_count}
    run.mkdir(parents=True,exist_ok=True); (run/'config.json').write_text(json.dumps(cfg,indent=2))
    print(f'TRAINING_BEGIN {args.architecture} output={run}',flush=True)
    print(train(model,tr,va,run,config=cfg,device=args.device,epochs=args.epochs,batch_size=args.batch_size,seed=20260920),flush=True)
if __name__=='__main__': main()
