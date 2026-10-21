"""Aggregate privacy-safe QC summaries for the FeatureLab candidate pool."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import Mapping
import csv, json
import numpy as np


def _validate(values, valid_mask, support_count, reason_codes):
    values=np.asarray(values,dtype=np.float64); valid=np.asarray(valid_mask,dtype=bool); support=np.asarray(support_count,dtype=np.int64); reasons=np.asarray(reason_codes,dtype=np.int64)
    if values.ndim != 2 or valid.shape != values.shape or support.shape != values.shape or reasons.shape != values.shape: raise ValueError('QC arrays must share a 2-D shape')
    if np.any(valid & ~np.isfinite(values)) or np.any((~valid) & ~np.isnan(values)): raise ValueError('values/mask disagreement')
    if np.any(support < 0): raise ValueError('support_count must be nonnegative')
    return values,valid,support,reasons


def _reason_label(code, reason_dictionary):
    if reason_dictionary is None: return str(int(code))
    try: return str(reason_dictionary[int(code)])
    except (KeyError, TypeError, ValueError) as exc: raise ValueError(f'unknown reason code {code}') from exc


def _feature_summary(values, valid, support, reasons, feature_id, reason_dictionary=None):
    v=values[valid]; valid_support=support[valid]
    counts=Counter(_reason_label(x, reason_dictionary) for x in reasons[~valid].tolist())
    return {'feature_id':str(feature_id),'valid_count':int(valid.sum()),'missing_count':int((~valid).sum()),
            'n_total':int(valid.size),'valid_rate':float(valid.mean()) if valid.size else None,
            'n_unique_valid':int(np.unique(v).size) if len(v) else 0,'all_missing':bool(len(v)==0),
            'zero_count':int(np.count_nonzero(v==0.0)),'zero_rate':float(np.mean(v==0.0)) if len(v) else None,
            'constant_valid':bool(len(v)>0 and np.all(v==v[0])),
            'support_min':int(valid_support.min()) if len(valid_support) else None,
            'support_median':float(np.median(valid_support)) if len(valid_support) else None,
            'support_max':int(valid_support.max()) if len(valid_support) else None,
            'missing_reason_counts':dict(sorted(counts.items())),
            'quantiles':{str(q):float(np.quantile(v,q)) for q in (0.01,0.25,0.5,0.75,0.99)} if len(v) else {}}


def summarize_matrix(values, valid_mask, support_count, reason_codes, feature_ids=None, groups=None, reason_dictionary=None):
    values,valid,support,reasons=_validate(values,valid_mask,support_count,reason_codes); n,f=values.shape
    ids=tuple(feature_ids or (f'feature_{i}' for i in range(f)))
    if len(ids)!=f or len(set(ids))!=f: raise ValueError('feature_ids must be unique and match columns')
    group_arr=None if groups is None else np.asarray(groups)
    if group_arr is not None and len(group_arr)!=n: raise ValueError('groups length mismatch')
    out={'overall':{'rows':n,'features':f,'valid_count':int(valid.sum()),'missing_count':int((~valid).sum()),'zero_count':int(np.count_nonzero(values[valid]==0.0)),'constant_feature_count':int(sum(np.count_nonzero(valid[:,j])>0 and np.all(values[valid[:,j],j]==values[valid[:,j],j][0]) for j in range(f)))},'features':{ids[j]:_feature_summary(values[:,j],valid[:,j],support[:,j],reasons[:,j],ids[j],reason_dictionary) for j in range(f)}}
    if group_arr is not None:
        out['groups']={}
        for group in sorted({str(x) for x in group_arr.tolist()}):
            mask=group_arr.astype(str)==group
            out['groups'][group]={'rows':int(mask.sum()),'valid_count':int(valid[mask].sum()),'missing_count':int((~valid[mask]).sum()),'zero_count':int(np.count_nonzero(values[mask][valid[mask]]==0.0))}
    return out


def summarize_scopes(values, valid_mask, support_count, reason_codes, feature_ids, groups, reason_dictionary=None, site_ids=('I0002','I0006','S0001')):
    values,valid,support,reasons=_validate(values,valid_mask,support_count,reason_codes)
    group_arr=np.asarray(groups).astype(str)
    if len(group_arr)!=values.shape[0]: raise ValueError('groups length mismatch')
    scopes={'Overall':np.ones(values.shape[0],dtype=bool)}
    scopes.update({str(site):group_arr==str(site) for site in site_ids})
    out={}
    for scope,mask in scopes.items():
        out[scope]=summarize_matrix(values[mask],valid[mask],support[mask],reasons[mask],feature_ids=feature_ids,reason_dictionary=reason_dictionary)
        out[scope]['scope']=scope; out[scope]['site_id']='' if scope=='Overall' else scope
    return out


def public_feature_rows(summary):
    return [{'feature_id':feature_id,'valid_count':data['valid_count'],'missing_count':data['missing_count'],'zero_count':data['zero_count'],'constant_valid':data['constant_valid']} for feature_id,data in summary['features'].items()]


def public_scope_feature_rows(scoped_summary, family_by_feature=None, modality_by_feature=None, implementation_by_feature=None):
    rows=[]
    for scope in ('Overall','I0002','I0006','S0001'):
        summary=scoped_summary[scope]
        for feature_id,data in summary['features'].items():
            rows.append({'scope':scope,'site_id':summary['site_id'],'feature_id':feature_id,
                         'family_id':(family_by_feature or {}).get(feature_id,''),'modality':(modality_by_feature or {}).get(feature_id,''),
                         'implementation_status':(implementation_by_feature or {}).get(feature_id,''),
                         'n_total':data['n_total'],'n_valid':data['valid_count'],'n_missing':data['missing_count'],
                         'valid_rate':data['valid_rate'],'n_unique_valid':data['n_unique_valid'],'all_missing':data['all_missing'],
                         'constant_valid':data['constant_valid'],'zero_count':data['zero_count'],'zero_rate':data['zero_rate'],
                         'missing_reason_counts':json.dumps(data['missing_reason_counts'],sort_keys=True,separators=(',',':'))})
    return rows


def write_public_qc_csv(path: Path, summary):
    rows=public_feature_rows(summary); target=Path(path); target.parent.mkdir(parents=True,exist_ok=True); fields=('feature_id','valid_count','missing_count','zero_count','constant_valid')
    with target.open('w',newline='',encoding='utf-8') as h: w=csv.DictWriter(h,fieldnames=fields,lineterminator='\n'); w.writeheader(); w.writerows(rows)


def write_public_scope_qc_csv(path: Path, scoped_summary, family_by_feature=None, modality_by_feature=None, implementation_by_feature=None):
    rows=public_scope_feature_rows(scoped_summary,family_by_feature,modality_by_feature,implementation_by_feature); target=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
    fields=('scope','site_id','feature_id','family_id','modality','implementation_status','n_total','n_valid','n_missing','valid_rate','n_unique_valid','all_missing','constant_valid','zero_count','zero_rate','missing_reason_counts')
    with target.open('w',newline='',encoding='utf-8') as h: w=csv.DictWriter(h,fieldnames=fields,lineterminator='\n'); w.writeheader(); w.writerows(rows)


def write_private_qc_json(path: Path, summary):
    target=Path(path); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(summary,sort_keys=True,indent=2),encoding='utf-8')
