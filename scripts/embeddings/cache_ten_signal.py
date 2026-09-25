"""Dependency-free contract/status/verification surface for ten-signal caches."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from image_aesthetics.embeddings.feature_contract import manifest_sha256, write_contract, load_contract, load_manifest
from image_aesthetics.embeddings.shards import verify_shard, read_shard
from image_aesthetics.embeddings.ledger import Ledger

def main():
 p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command",required=True)
 c=sub.add_parser("contract"); c.add_argument("--manifest",type=Path,required=True); c.add_argument("--image-root",type=Path,required=True); c.add_argument("--contract",type=Path,required=True); c.add_argument("--spec",type=Path,required=True)
 s=sub.add_parser("status"); s.add_argument("--cache-dir",type=Path,required=True)
 f=sub.add_parser("failures"); f.add_argument("--cache-dir",type=Path,required=True)
 v=sub.add_parser("verify"); v.add_argument("--cache-dir",type=Path,required=True)
 a=p.parse_args()
 if a.command=="contract":
  spec=json.loads(a.spec.read_text()); load_manifest(a.manifest,a.image_root); ident=write_contract(a.contract,input_manifest_sha256=manifest_sha256(a.manifest),contract=spec); print(json.dumps({"cache_identity":ident,"manifest_sha256":manifest_sha256(a.manifest)}))
 else:
  db=a.cache_dir/"work.sqlite3"
  if not db.exists(): raise SystemExit("ledger missing")
  ledger=Ledger(db); ledger.open()
  if a.command=="status": print(json.dumps(ledger.status_counts(), sort_keys=True))
  elif a.command=="failures": print(json.dumps(ledger.failures(), sort_keys=True))
  else:
   errors=[]; verified=0; shard_rows={}
   committed = ledger.committed_shards()
   ledger_shards = {int(row["shard_id"]): row for row in committed}
   completed=ledger.completed_records()
   shard_paths = [Path(row["path"]) for row in committed] + [p for p in sorted(a.cache_dir.glob("shard-*")) if p.suffix in (".safetensors", ".npz") and p not in {Path(r["path"]) for r in committed}]
   for shard in shard_paths:
    if not shard.exists():
     errors.append({"type":"missing_shard","shard":shard.name}); continue
    if shard.suffix in (".safetensors", ".npz"):
     try:
      emb, meta=read_shard(shard); verify_shard(shard, int(meta.get("embedding_dim", 768))); verified += 1; sid=int(shard.stem.split('-')[-1]); shard_rows[sid]=(meta,emb)
      if sid not in ledger_shards:
       errors.append({"type":"orphan_shard","shard":shard.name})
      else:
       expected = ledger_shards[sid]
       if expected["sha256"] != meta.get("sha256"):
        errors.append({"type":"checksum_mismatch","shard":shard.name,"shard_id":sid})
       if expected["row_count"] != int(meta.get("row_count", -1)):
        errors.append({"type":"row_count_mismatch","shard":shard.name,"shard_id":sid})
     except Exception as exc: errors.append({"type":"corrupt_shard", "shard": shard.name, "error": str(exc)})
   for sid, expected in sorted(ledger_shards.items()):
    if sid not in shard_rows:
     errors.append({"type":"missing_shard","shard_id":sid})
   for item in completed:
    pair=shard_rows.get(item["shard_id"]); meta=pair[0] if pair else None; emb=pair[1] if pair else None
    if meta is None: errors.append({"type":"missing_shard","image_id":item["image_id"],"shard_id":item["shard_id"]}); continue
    row=item["shard_row"]
    if row is None or row<0 or row>=meta["row_count"]: errors.append({"type":"row_out_of_range","image_id":item["image_id"],"shard_id":item["shard_id"],"shard_row":row}); continue
    if meta["image_ids"][row] != item["image_id"] or meta["positions"][row] != item["manifest_position"]: errors.append({"type":"coordinate_mismatch","image_id":item["image_id"],"shard_id":item["shard_id"],"shard_row":row})
    else:
     checksum = hashlib.sha256(emb[row].tobytes()).hexdigest()
     if item.get("embedding_checksum") and checksum != item["embedding_checksum"]:
      errors.append({"type":"checksum_mismatch","image_id":item["image_id"],"shard_id":item["shard_id"],"shard_row":row})
   errors.sort(key=lambda e: (e.get("type",e.get("error","")), e.get("image_id", ""), str(e.get("shard_id", ""))))
   print(json.dumps({"ledger":"ok","verified_shards":verified,"errors":errors}, sort_keys=True))
if __name__=="__main__": main()
