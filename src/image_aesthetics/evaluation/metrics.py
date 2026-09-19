"""Numpy-only metrics for AVA distribution predictions."""
from __future__ import annotations
import numpy as np

def _rank(x):
    order=np.argsort(x, kind="mergesort"); out=np.empty(len(x),float); out[order]=np.arange(len(x));
    # average ties
    for v in np.unique(x):
        ix=np.flatnonzero(x==v)
        if len(ix)>1: out[ix]=out[ix].mean()
    return out
def corr(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); return float(np.corrcoef(x,y)[0,1]) if len(x)>1 and x.std()>0 and y.std()>0 else 0.0
def metric_report(pred_dist, target_dist, seed=20260920, pair_samples=200_000):
    p=np.asarray(pred_dist,float); t=np.asarray(target_dist,float); bins=np.arange(1,11)
    ps=(p*bins).sum(1); ts=(t*bins).sum(1); pn=(ps-1)/9; tn=(ts-1)/9
    spearman=corr(_rank(ps),_rank(ts)); pearson=corr(ps,ts)
    rng=np.random.default_rng(seed); n=len(ps); m=min(pair_samples,n*(n-1)//2)
    i=rng.integers(0,n,m); j=rng.integers(0,n,m); keep=i!=j; i=i[keep]; j=j[keep]
    true=np.sign(ts[i]-ts[j]); pred=np.sign(ps[i]-ps[j]); useful=true!=0
    pair=float((true[useful]==pred[useful]).mean()) if useful.any() else 0.0
    cdfp=np.cumsum(p,1); cdft=np.cumsum(t,1); emd=float(np.mean((cdfp-cdft)**2))
    out={"spearman":spearman,"pearson":pearson,"mae":float(np.mean(abs(pn-tn))),"rmse":float(np.sqrt(np.mean((pn-tn)**2))),"emd":emd,"pairwise_accuracy":pair}
    for q in (.01,.05,.10):
        k=max(1,int(np.ceil(n*q))); a=np.argsort(-ps)[:k]; b=np.argsort(-ts)[:k]
        out[f"top{int(q*100)}_quality"]=float(ts[a].mean()); out[f"top{int(q*100)}_lift"]=float(ts[a].mean()-ts.mean()); out[f"top{int(q*100)}_overlap"]=float(len(set(a)&set(b))/k)
    out["calibration_bias"]=float(np.mean(pn-tn)); out["calibration_slope"]=float(np.polyfit(tn,pn,1)[0]) if len(tn)>1 and tn.std()>0 else 0.0
    return out
