"""Aggregate quality-control summaries for the private v2 candidate pool."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping
import csv
import json
import numpy as np

def _validate(values, valid_mask, support_count, reason_codes):
    values=np.asarray(values, dtype=np.float64); valid=np.asarray(valid_mask,dtype=bool); support=np.asarray(support_count,dtype=np.int64); reasons=np.asarray(reason_codes,dtype=np.int64)
    if values.ndim != 2 or valid.shape != values.shape or support.shape != values.shape or reasons.shape != values.shape: raise ValueError('QC arrays must share a 2-D shape')
    if np.any(valid & ~np.isfinite(values)) or np.any((~valid) & ~np.isnan(values)): raise ValueError('values/mask disagreement')
    if np.any(support < 0): raise ValueError('support_count must be nonnegative')
    return values,valid,support,reasons

def _feature_summary(values, valid, support, reasons, feature_id):
    v=values[valid]; s=support
    counts=Counter(str(int(x)) for x in reasons[~valid].tolist())
    return {'feature_id':str(feature_id),'valid_count':int(valid.sum()),'missing_count':int((~valid).sum()),'zero_count':int(np.count_nonzero(v==0.0)),'constant_valid':bool(len(v)>0 and np.all(v==v[0])),'support_min':int(s.min()) if len(s) else 0,'support_median':float(np.median(s)) if len(s) else 0.0,'support_max':int(s.max()) if len(s) else 0,'missing_reason_counts':dict(sorted(counts.items())),'quantiles':{q:float(np.quantile(v,q)) for q in (0.01,0.25,0.5,0.75,0.99)} if len(v) else {}}

def summarize_matrix(values, valid_mask, support_count, reason_codes, feature_ids=None, groups=None):
    values,valid,support,reasons=_validate(values,valid_mask,support_count,reason_codes)
    n,f=values.shape
    ids=tuple(feature_ids or (f'feature_{i}' for i in range(f)))
    if len(ids)!=f or len(set(ids))!=f: raise ValueError('feature_ids must be unique and match columns')
    group_arr=None if groups is None else np.asarray(groups)
    if group_arr is not None and len(group_arr)!=n: raise ValueError('groups length mismatch')
    out={'overall':{},'features':{}}
    all_valid=valid
    out['overall']={'rows':n,'features':f,'valid_count':int(all_valid.sum()),'missing_count':int((~all_valid).sum()),'zero_count':int(np.count_nonzero(values[all_valid]==0.0)),'constant_feature_count':int(sum(np.count_nonzero(valid[:,j])>0 and np.all(values[valid[:,j],j]==values[valid[:,j],j][0]) for j in range(f)))}
    out['features']={ids[j]:_feature_summary(values[:,j],valid[:,j],support[:,j],reasons[:,j],ids[j]) for j in range(f)}
    if group_arr is not None:
        by_group={}
        for group in sorted({str(x) for x in group_arr.tolist()}):
            mask=group_arr.astype(str)==group
            by_group[group]={'rows':int(mask.sum()),'valid_count':int(valid[mask].sum()),'missing_count':int((~valid[mask]).sum()),'zero_count':int(np.count_nonzero(values[mask][valid[mask]]==0.0))}
        out['groups']=by_group
    return out

def public_feature_rows(summary):
    rows=[]
    for feature_id,data in summary['features'].items():
        rows.append({'feature_id':feature_id,'valid_count':data['valid_count'],'missing_count':data['missing_count'],'zero_count':data['zero_count'],'constant_valid':data['constant_valid'],'support_min':data['support_min'],'support_median':data['support_median'],'support_max':data['support_max']})
    return rows

def write_public_qc_csv(path: Path, summary):
    rows=public_feature_rows(summary); target=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
    fields=('feature_id','valid_count','missing_count','zero_count','constant_valid','support_min','support_median','support_max')
    with target.open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=fields,lineterminator='\n'); w.writeheader(); w.writerows(rows)

def write_private_qc_json(path: Path, summary):
    target=Path(path); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(summary,sort_keys=True,indent=2),encoding='utf-8')
