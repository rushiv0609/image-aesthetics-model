#!/usr/bin/env python3
"""Create an immutable, group-safe SPAQ technical auxiliary split."""
from __future__ import annotations
import argparse, json, os, tempfile
from pathlib import Path

from image_aesthetics.data.fusion_training_dataset import prepare_spaq_technical_splits

def load(path): return [json.loads(line) for line in Path(path).read_text().splitlines() if line]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels',type=Path,required=True); parser.add_argument('--joined-cache',type=Path,required=True)
    parser.add_argument('--exclude-manifest',type=Path,help='JSONL whose source hashes must be excluded from SPAQ')
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--seed',type=int,default=20260920)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    labels={str(row['label_join_id']):row for row in load(args.labels)}; rows=[]; missing=[]
    excluded_hashes={str(row.get('sha256') or row.get('source_sha256')) for row in load(args.exclude_manifest)} if args.exclude_manifest else set()
    excluded=[]
    for joined in load(args.joined_cache/'join.jsonl'):
        if joined.get('quarantine') or joined.get('label_task')!='technical': continue
        label=labels.get(str(joined.get('label_join_id')))
        if label is None: missing.append(joined['image_id']); continue
        if str(joined['source_sha256']) in excluded_hashes: excluded.append(joined['image_id']); continue
        rows.append({**label,'dataset_id':'spaq','image_id':joined['image_id'],'feature_id':joined['image_id'],
                     'sha256':joined['source_sha256'],'upstream_locked_test':bool(label.get('locked_test'))})
    prepared=prepare_spaq_technical_splits(rows,seed=args.seed)
    staging=Path(tempfile.mkdtemp(prefix=f'.{args.output.name}.',dir=args.output.parent))
    try:
        with (staging/'samples.jsonl').open('w') as handle:
            for row in prepared: handle.write(json.dumps(row,sort_keys=True)+'\n')
        audit={'schema_version':'spaq-technical-v1','seed':args.seed,'records':len(prepared),
               'development':sum(not row['locked_test'] for row in prepared),
               'locked_test':sum(row['locked_test'] for row in prepared),
               'fold_counts':{str(fold):sum(not row['locked_test'] and row['development_fold']==fold for row in prepared) for fold in range(5)},
               'missing_label_count':len(missing),'missing_label_examples':missing[:20],
               'cross_corpus_excluded_count':len(excluded),'cross_corpus_excluded_examples':excluded[:20]}
        (staging/'audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n'); os.replace(staging,args.output); print(json.dumps(audit,sort_keys=True))
    except Exception:
        import shutil; shutil.rmtree(staging,ignore_errors=True); raise

if __name__=='__main__': main()
