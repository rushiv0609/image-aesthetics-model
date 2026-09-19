"""Efficient AVA split-to-embedding join over immutable cached shards."""
from __future__ import annotations
import json, sqlite3
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from safetensors.numpy import load_file

class CachedAVADataset(Dataset):
    def __init__(self, manifest_path, cache_dir, *, allow_locked_test=False, max_shards=4, limit=None):
        self.manifest_path=Path(manifest_path); self.cache_dir=Path(cache_dir)
        if self.manifest_path.name == "locked-test.jsonl" and not allow_locked_test: raise PermissionError("locked test requires explicit opt-in")
        self.records=[json.loads(x) for x in self.manifest_path.read_text().splitlines() if x.strip()]
        if limit: self.records=self.records[:limit]
        self._cache=OrderedDict(); self.max_shards=max_shards
        db=self.cache_dir/"work.sqlite3"
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
            rows=con.execute("SELECT image_id,shard_id,shard_row,state FROM items WHERE state='completed'").fetchall()
            shardrows=con.execute("SELECT shard_id,path,sha256 FROM shards").fetchall()
        coords={str(a):(int(b),int(c)) for a,b,c,d in rows}; paths={int(a):(Path(b),str(c)) for a,b,c in shardrows}
        self._coords=[]
        for rec in self.records:
            iid=str(rec["image_id"])
            if iid not in coords: raise ValueError(f"missing completed cache row {iid}")
            sid,row=coords[iid]
            if sid not in paths: raise ValueError(f"missing shard {sid}")
            self._coords.append((sid,row))
        self._paths=paths
    def __len__(self): return len(self.records)
    def _shard(self,sid):
        if sid in self._cache: self._cache.move_to_end(sid); return self._cache[sid]
        arr=load_file(str(self._paths[sid][0]))["embeddings"]
        self._cache[sid]=arr; self._cache.move_to_end(sid)
        while len(self._cache)>self.max_shards: self._cache.popitem(last=False)
        return arr
    def __getitem__(self, i):
        r=self.records[i]; x=self._shard(self._coords[i][0])[self._coords[i][1]]
        target=np.asarray(r["vote_counts"],dtype=np.float32); target/=target.sum()
        return torch.from_numpy(np.asarray(x,dtype=np.float32)), torch.from_numpy(target), str(r["image_id"])
