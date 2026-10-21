"""Private 1444-column Batch1 candidate pool assembly and public contracts."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np

from .pool_sources import Alignment, read_aligned_record

FAMILY_ORDER=("eeg_tail","eeg_temporal","eeg_coherence_stage","caisr_stage","emg_stage","resp_stage")
BASE_VIEW_DIMS={'V1':113,'V2':129,'V3':349,'V4':113,'V5':123}
NEW_VIEW_COUNTS={'V1':48,'V2':48,'V3':144,'V4':144,'V5':253}
VIEW_IDS=('V1','V2','V3','V4','V5')

def build_column_contract(old_feature_ids:Sequence[str], family_feature_ids:Mapping[str,Sequence[str]]):
    old=tuple(str(x) for x in old_feature_ids)
    if len(old)!=807 or len(set(old))!=807: raise ValueError('old feature contract must contain 807 unique IDs')
    ordered=[]
    for family in FAMILY_ORDER:
        ids=tuple(str(x) for x in family_feature_ids[family]); ordered.extend(ids)
    all_ids=old+tuple(ordered)
    if len(all_ids)!=1444 or len(set(all_ids))!=1444: raise ValueError('v2 feature IDs must be globally unique')
    return {'unique_feature_ids':all_ids,'old_feature_ids':old,'family_feature_ids':{f:tuple(str(x) for x in family_feature_ids[f]) for f in FAMILY_ORDER},'column_index':{fid:i for i,fid in enumerate(all_ids)},'experiment_sets':{'E0':807,'E1':1191,'E2':1060,'E3':1444}}

def _old_rows_from_contract(contract, old_view_rows):
    by_view={view:[] for view in VIEW_IDS}
    if old_view_rows:
        for row in old_view_rows:
            view=str(row['view_id']); fid=str(row['unique_feature_id'])
            if view not in BASE_VIEW_DIMS: raise ValueError('unknown old view')
            by_view[view].append({'view_id':view,'view_position':int(row['view_position']),'feature_id':fid,'unique_column_index':contract['column_index'][fid],'source_block':'old807'})
        return by_view
    offset=0
    for view in VIEW_IDS:
        for pos in range(BASE_VIEW_DIMS[view]):
            old_ids=contract['old_feature_ids']
            fid=old_ids[offset] if offset < len(old_ids) else old_ids[offset-len(old_ids)]
            by_view[view].append({'view_id':view,'view_position':pos,'feature_id':fid,'unique_column_index':offset,'source_block':'old807'})
            offset+=1
    return by_view
def _new_rows(contract,new_manifest_rows):
    if new_manifest_rows:
        by_view={view:[] for view in VIEW_IDS}
        for row in new_manifest_rows:
            view=str(row['proposed_view']); fid=str(row['feature_id']);
            if view in by_view: by_view[view].append((contract['column_index'][fid],fid,str(row['family_id'])))
        return by_view
    out={v:[] for v in VIEW_IDS}; offset=807
    for fam in FAMILY_ORDER:
        ids=contract['family_feature_ids'][fam]
        target={'eeg_tail':None,'eeg_temporal':'V4','eeg_coherence_stage':'V5','caisr_stage':'V5','emg_stage':'V5','resp_stage':'V5'}[fam]
        if fam=='eeg_tail':
            # synthetic fallback uses the approved segment counts, not feature semantics.
            split=(('V1',48),('V2',48),('V3',144),('V4',48))
            cursor=0
            for view,count in split:
                for fid in ids[cursor:cursor+count]: out[view].append((offset,fid,fam)); offset+=1
                cursor+=count
        elif fam=='eeg_temporal':
            for fid in ids: out['V4'].append((offset,fid,fam)); offset+=1
        else:
            for fid in ids: out['V5'].append((offset,fid,fam)); offset+=1
    return out

def build_view_manifest(contract, old_view_rows, new_manifest_rows=None):
    rows=_old_rows_from_contract(contract,old_view_rows)
    additions=_new_rows(contract,new_manifest_rows)
    for view in VIEW_IDS:
        start=len(rows[view])
        for i,(idx,fid,fam) in enumerate(additions[view]): rows[view].append({'view_id':view,'view_position':start+i,'feature_id':fid,'unique_column_index':idx,'source_block':fam})
    expected={v:BASE_VIEW_DIMS[v]+NEW_VIEW_COUNTS[v] for v in VIEW_IDS}
    if {v:len(rows[v]) for v in VIEW_IDS}!=expected: raise ValueError(f'Unexpected v2 view dimensions: { {v:len(rows[v]) for v in VIEW_IDS} }')
    return rows

def _as_matrix(array,n, f, name, dtype):
    arr=np.asarray(array,dtype=dtype)
    if arr.ndim==1 and n==1: arr=arr.reshape(1,-1)
    if arr.shape!=(n,f): raise ValueError(f'{name} shape {arr.shape} != {(n,f)}')
    return arr

def assemble_arrays(old_values,old_valid,old_support,family_blocks,old_feature_ids,family_feature_ids):
    old_values=np.asarray(old_values,dtype=np.float64); old_valid=np.asarray(old_valid,dtype=bool); old_support=np.asarray(old_support,dtype=np.int64)
    if old_values.ndim==1: old_values=old_values.reshape(1,-1); old_valid=old_valid.reshape(1,-1); old_support=old_support.reshape(1,-1)
    n=old_values.shape[0]; old_ids=tuple(old_feature_ids)
    if old_values.shape[1]!=len(old_ids): raise ValueError('old feature length mismatch')
    blocks=family_blocks if isinstance(family_blocks,Mapping) else {f:b for f,b in zip(FAMILY_ORDER,family_blocks)}
    family_ids=family_feature_ids if isinstance(family_feature_ids,Mapping) else {f:ids for f,ids in zip(FAMILY_ORDER,family_feature_ids)}
    vals=[old_values]; masks=[old_valid]; supports=[old_support]; reasons=[np.where(old_valid,0,1).astype(np.int64)]; ids=[old_ids]; reason_dict={0:'VALID',1:'OLD_SOURCE_MISSING'}
    next_code=2
    for family in FAMILY_ORDER:
        if family not in blocks: continue
        fids=tuple(family_ids[family]); b=blocks[family]
        v=_as_matrix(b['values'],n,len(fids),'values',np.float64); m=_as_matrix(b['valid_mask'],n,len(fids),'valid_mask',bool); s=_as_matrix(b['support_count'],n,len(fids),'support_count',np.int64)
        if np.any(m & ~np.isfinite(v)) or np.any((~m)&~np.isnan(v)): raise ValueError(f'{family} values/mask disagreement')
        if np.any(s<0): raise ValueError(f'{family} support negative')
        raw=b.get('missing_reason',tuple('' if ok else 'MISSING' for ok in m.reshape(-1)))
        reason_arr=np.asarray(raw,dtype=str)
        if reason_arr.ndim==1 and n==1: reason_arr=reason_arr.reshape(1,-1)
        if reason_arr.shape!=(n,len(fids)): raise ValueError(f'{family} missing_reason shape mismatch')
        codes=np.zeros((n,len(fids)),dtype=np.int64)
        for reason in sorted({str(x) for x in reason_arr[~m].tolist()}):
            if not reason: reason='MISSING'
            if reason not in reason_dict: reason_dict[reason]=next_code; next_code+=1
        for i in range(n):
            for j in range(len(fids)): codes[i,j]=0 if m[i,j] else reason_dict.get(str(reason_arr[i,j]) or 'MISSING',1)
        vals.append(v); masks.append(m); supports.append(s); reasons.append(codes); ids.append(fids)
    result={'values':np.concatenate(vals,axis=1),'valid_mask':np.concatenate(masks,axis=1),'support_count':np.concatenate(supports,axis=1),'missing_reason_codes':np.concatenate(reasons,axis=1),'missing_reason_dictionary':reason_dict,'unique_feature_ids':tuple(x for group in ids for x in group)}
    if result['values'].shape[1]!=1444: raise ValueError('assembled pool must have 1444 columns')
    return result

def assemble_candidate_pool(alignment:Alignment, output_root:Path, view_rows=None, experiment_manifest=None):
    n=len(alignment.entries); contract=build_column_contract(alignment.old_feature_ids,alignment.family_feature_ids); values=np.empty((n,1444),dtype=np.float64); valid=np.empty((n,1444),dtype=bool); support=np.empty((n,1444),dtype=np.int64); codes=np.empty((n,1444),dtype=np.int64); reason_dict={'VALID':0,'OLD_SOURCE_MISSING':1}; next_code=2
    row_keys=[]; groups=[]
    for i,entry in enumerate(alignment.entries):
        block=read_aligned_record(entry,alignment); assembled=assemble_arrays(block['old_values'],block['old_valid_mask'],block['old_support_count'],block['family_blocks'],alignment.old_feature_ids,alignment.family_feature_ids)
        if not np.array_equal(assembled['values'][0,:807], block['old_values'], equal_nan=True): raise ValueError(f'old807 equality failure at row {i}')
        cursor=807
        for family in FAMILY_ORDER:
            family_values=np.asarray(block['family_blocks'][family]['values'],dtype=np.float64).reshape(1,-1)[0]
            width=len(alignment.family_feature_ids[family])
            if not np.array_equal(assembled['values'][0,cursor:cursor+width], family_values, equal_nan=True): raise ValueError(f'{family} equality failure at row {i}')
            cursor+=width
        values[i]=assembled['values'][0]; valid[i]=assembled['valid_mask'][0]; support[i]=assembled['support_count'][0]; row_codes=assembled['missing_reason_codes'][0]
        local_dict=assembled['missing_reason_dictionary']
        remap={}
        for label,code in local_dict.items():
            if label not in reason_dict: reason_dict[label]=next_code; next_code+=1
            remap[code]=reason_dict[label]
        codes[i]=np.asarray([remap.get(int(c),0) for c in row_codes],dtype=np.int64); row_keys.append(entry.record_key); groups.append(entry.site_id)
    out=Path(output_root); out.mkdir(parents=True,exist_ok=True); np.save(out/'X_raw.npy',values); np.save(out/'valid_mask.npy',valid); np.save(out/'support_count.npy',support); np.save(out/'missing_reason_codes.npy',codes); (out/'missing_reason_dictionary.json').write_text(json.dumps({str(k):v for k,v in reason_dict.items()},sort_keys=True,indent=2),encoding='utf-8'); np.save(out/'row_record_keys.npy',np.asarray(row_keys,dtype=np.str_)); np.save(out/'row_sites.npy',np.asarray(groups,dtype=np.str_)); np.save(out/'unique_feature_ids.npy',np.asarray(contract['unique_feature_ids'],dtype=np.str_));
    views=build_view_manifest(contract,view_rows or (),experiment_manifest.get('new_manifest_rows') if experiment_manifest else None)
    with (out/'view_manifest.csv').open('w',newline='',encoding='utf-8') as h:
        fields=('view_id','view_position','feature_id','unique_column_index','source_block'); w=csv.DictWriter(h,fieldnames=fields,lineterminator='\n'); w.writeheader(); [w.writerow(row) for view in VIEW_IDS for row in views[view]]
    (out/'completion.json').write_text(json.dumps({'status':'COMPLETE','rows':n,'columns':1444,'source_identity_order':'old_g2b1_index','reason_dictionary':{str(k):v for k,v in reason_dict.items()}},sort_keys=True,indent=2),encoding='utf-8')
    return {'contract':contract,'views':views,'values':values,'valid_mask':valid,'support_count':support,'missing_reason_codes':codes,'reason_dictionary':reason_dict,'row_keys':row_keys,'groups':groups,'output_root':out}
