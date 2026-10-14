"""Column-level raw, rank and Rank256 audit functions."""
from __future__ import annotations
from collections import Counter
import numpy as np


def _pct(x,q): return float(np.percentile(x,q,method="linear")) if len(x) else np.nan


def audit_raw(raw,mask,feature_ids,families,site):
    raw=np.asarray(raw,float); mask=np.asarray(mask,bool); rows=[]
    for j,uid in enumerate(feature_ids):
        x=raw[mask[:,j],j]; counts=Counter(x.tolist()); max_tie=max(counts.values()) if counts else 0
        rows.append({"site":site,"feature_id":uid,"family":families[j],"n_total":len(raw),"n_valid":len(x),"n_missing":len(raw)-len(x),
                     "missing_rate":(len(raw)-len(x))/len(raw),"n_unique":len(counts),"lawful_zero_count":int(np.sum(x==0)),
                     "max_tie_count":max_tie,"max_tie_fraction":max_tie/len(x) if len(x) else 0.0,
                     "raw_min":float(np.min(x)) if len(x) else np.nan,"raw_max":float(np.max(x)) if len(x) else np.nan,
                     "raw_p05":_pct(x,5),"raw_median":_pct(x,50),"raw_p95":_pct(x,95)})
    return rows


def audit_rank(raw,percentile,quantized,source_mask,rank_mask,metadata):
    raw=np.asarray(raw,float); p=np.asarray(percentile,float); q=np.asarray(quantized,float)
    source=np.asarray(source_mask,bool); rank=np.asarray(rank_mask,bool); levels=np.asarray(metadata["levels"])
    if not(raw.shape==p.shape==q.shape==source.shape==rank.shape==levels.shape): raise ValueError("audit shapes")
    if np.any(rank&~source): raise ValueError("rank valid without source")
    if not np.isfinite(p[rank]).all() or not np.isfinite(q[rank]).all() or np.any((p[rank]<0)|(p[rank]>1)): raise ValueError("rank range")
    if not np.isnan(p[~rank]).all() or not np.isnan(q[~rank]).all(): raise ValueError("rank missing semantics")
    if np.any(levels[~rank]!=-1) or np.any((levels[rank]<0)|(levels[rank]>255)): raise ValueError("level range")
    if not np.allclose(q[rank],levels[rank]/255.0,rtol=0,atol=1e-15): raise ValueError("off-grid")
    if np.any(np.abs(q[rank]-p[rank])>1/510+1e-14): raise ValueError("quantization error")
    feature_ids=metadata["feature_ids"]; families=metadata["families"]; refn=np.asarray(metadata["reference_n"])
    refmin=np.asarray(metadata.get("reference_min",np.full(raw.shape[1],np.nan))); refmax=np.asarray(metadata.get("reference_max",np.full(raw.shape[1],np.nan)))
    rows=[]
    for j,uid in enumerate(feature_ids):
        idx=np.flatnonzero(rank[:,j]); rv=raw[idx,j]; pv=p[idx,j]; qv=q[idx,j]; kv=levels[idx,j]
        if len(idx):
            order=np.argsort(rv,kind="mergesort")
            if np.any(np.diff(pv[order])<0) or np.any(np.diff(qv[order])<0): raise ValueError(f"nonmonotone {uid}")
            for value in np.unique(rv):
                if len(np.unique(pv[rv==value]))!=1: raise ValueError(f"tie mismatch {uid}")
        pu=np.unique(pv); qu=np.unique(kv); grouped={level:len(np.unique(pv[kv==level])) for level in qu}
        error=np.abs(qv-pv)
        rows.append({"feature_id":uid,"family":families[j],"n_total":len(raw),"n_source_valid":int(source[:,j].sum()),"n_rank_valid":len(idx),
                     "reference_n":int(refn[j]),"reference_unavailable_count":int(np.sum(source[:,j]&~rank[:,j])),
                     "n_percentile_unique":len(pu),"n_quantized_unique":len(qu),"occupied_bins":len(qu),"occupied_bin_fraction":len(qu)/256,
                     "max_abs_quantization_error":float(np.max(error)) if len(error) else 0.0,"mean_abs_quantization_error":float(np.mean(error)) if len(error) else 0.0,
                     "new_tie_group_count":sum(v>1 for v in grouped.values()),"distinct_rank_levels_collapsed":len(pu)-len(qu),
                     "zero_level_count":int(np.sum(kv==0)),"top_level_count":int(np.sum(kv==255)),
                     "percentile_min":float(np.min(pv)) if len(pv) else np.nan,"percentile_max":float(np.max(pv)) if len(pv) else np.nan,
                     "quantized_min":float(np.min(qv)) if len(qv) else np.nan,"quantized_max":float(np.max(qv)) if len(qv) else np.nan,
                     "below_reference_count":int(np.sum(rv<refmin[j])) if np.isfinite(refmin[j]) else 0,
                     "above_reference_count":int(np.sum(rv>refmax[j])) if np.isfinite(refmax[j]) else 0})
    return rows


def compare_protocols(train_trans,train_ind,test_trans,test_ind,train_mask,test_mask):
    if not np.array_equal(train_trans,train_ind,equal_nan=True): raise ValueError("protocol train percentile mismatch")
    if np.asarray(train_mask).shape!=np.asarray(train_trans).shape or np.asarray(test_mask).shape!=np.asarray(test_trans).shape: raise ValueError("comparison masks")
    comparable=np.asarray(test_mask,bool)&np.isfinite(test_trans)&np.isfinite(test_ind)
    difference=np.zeros_like(comparable); difference[comparable]=test_trans[comparable]!=test_ind[comparable]
    return {"train_exact":True,"test_comparable_count":int(comparable.sum()),"test_different_count":int(difference.sum()),
            "test_mean_abs_difference":float(np.mean(np.abs(test_trans[comparable]-test_ind[comparable]))) if comparable.any() else 0.0,
            "test_max_abs_difference":float(np.max(np.abs(test_trans[comparable]-test_ind[comparable]))) if comparable.any() else 0.0}
