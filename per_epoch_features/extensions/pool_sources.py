"""Read-only alignment of frozen old and Batch1 family caches."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np

from .cache import read_cache
from .manifest import frozen_feature_ids
from .registry import get_family, list_families
from .recipe import build_recipe_identity

DEFAULT_FAMILY_IDS=("eeg_tail","eeg_temporal","eeg_coherence_stage","caisr_stage","emg_stage","resp_stage")
OLD_CACHE_SCHEMA="g2b1_full_v1"

class AlignmentError(RuntimeError): pass
class SourceIdentityMismatch(AlignmentError): pass
class CacheContractError(AlignmentError): pass

@dataclass(frozen=True)
class AlignmentEntry:
    old_index:int
    site_id:str
    record_id:str
    relative_path:str
    old_cache_path:Path
    old_source_sha256:str
    family_paths:Mapping[str,Path]
    record_key:str

@dataclass(frozen=True)
class Alignment:
    entries:tuple[AlignmentEntry,...]
    old_feature_ids:tuple[str,...]
    family_feature_ids:Mapping[str,tuple[str,...]]
    recipe_ids:Mapping[str,str]
    site_counts:Mapping[str,int]

@dataclass(frozen=True)
class RecordFeatureBlocks:
    entry:AlignmentEntry
    old_values:np.ndarray
    old_valid_mask:np.ndarray
    old_support_count:np.ndarray
    family_blocks:Mapping[str,Mapping[str,np.ndarray|tuple[str,...]]]

def parse_record_key(relative_path:str):
    path=Path(relative_path); parts=path.parts
    if len(parts)<2 or path.suffix.lower()!='.npz': raise AlignmentError(f'Invalid relative source path: {relative_path}')
    return str(parts[0]), path.stem

def _scalar(payload,key):
    arr=np.asarray(payload[key]);
    if arr.shape!=() or arr.dtype.kind not in 'US': raise CacheContractError(f'{key} must be scalar text')
    return str(arr.item())

def _load_json(payload,key):
    try: return json.loads(_scalar(payload,key))
    except Exception as exc: raise CacheContractError(f'Invalid {key}') from exc

def _old_rows(old_root:Path):
    index_path=old_root/'private_index.json'
    if not index_path.exists(): raise AlignmentError(f'Missing old private index: {index_path}')
    try: rows=json.loads(index_path.read_text(encoding='utf-8'))
    except Exception as exc: raise AlignmentError('Old private index unreadable') from exc
    if not isinstance(rows,list): raise AlignmentError('Old private index must be a list')
    seen=set(); out=[]
    for i,row in enumerate(rows):
        if not isinstance(row,dict): raise AlignmentError('Old index row malformed')
        site,record=parse_record_key(str(row.get('relative_path','')))
        key=(site,record)
        if key in seen: raise AlignmentError(f'Duplicate old identity: {key}')
        seen.add(key)
        cache_name=str(row.get('cache_name',''))
        cache=old_root/'cache'/cache_name
        if not cache.is_file(): raise AlignmentError(f'Missing old cache: {cache}')
        out.append((i,site,record,str(row['relative_path']),cache))
    return out

def _validate_old_cache(path:Path, expected_ids=None):
    try:
        with np.load(path,allow_pickle=False) as z:
            names=set(z.files)
            required={'cache_schema_version','unique_feature_ids','unique_values','unique_valid_mask','unique_support_count','source_npz_sha256'}
            if not required.issubset(names): raise CacheContractError(f'Old cache missing keys: {path}')
            if str(np.asarray(z['cache_schema_version']).item()) != OLD_CACHE_SCHEMA: raise CacheContractError(f'Old cache schema mismatch: {path}')
            ids=tuple(str(x) for x in np.asarray(z['unique_feature_ids']).tolist())
            values=np.asarray(z['unique_values'])
            valid=np.asarray(z['unique_valid_mask'])
            support=np.asarray(z['unique_support_count'])
            source=np.asarray(z['source_npz_sha256'])
    except CacheContractError: raise
    except Exception as exc: raise AlignmentError(f'Unreadable old cache: {path}') from exc
    if len(ids)!=807 or len(set(ids))!=807 or values.shape!=(807,) or valid.shape!=(807,) or support.shape!=(807,): raise CacheContractError(f'Old 807 shape/order invalid: {path}')
    if values.dtype!=np.float64 or valid.dtype!=bool or support.dtype!=np.int64: raise CacheContractError(f'Old 807 dtype invalid: {path}')
    if np.any(valid & ~np.isfinite(values)) or np.any((~valid)&~np.isnan(values)): raise CacheContractError(f'Old validity mismatch: {path}')
    if np.any(support<0): raise CacheContractError(f'Old support invalid: {path}')
    if source.shape!=() or source.dtype.kind not in 'US': raise CacheContractError(f'Old source digest invalid: {path}')
    if expected_ids is not None and ids!=tuple(expected_ids): raise CacheContractError(f'Old feature order mismatch: {path}')
    return {'unique_values':values,'unique_valid_mask':valid,'unique_support_count':support},ids,str(source.item())

def _old_feature_contract(old_rows, manifest_path:Path|None):
    p,ids,source=_validate_old_cache(old_rows[0][4])
    if manifest_path is not None:
        try:
            with Path(manifest_path).open(newline='', encoding='utf-8') as handle:
                manifest_rows=list(csv.DictReader(handle))
            ordered=[]
            for row in manifest_rows:
                feature_id=str(row.get('unique_feature_id',''))
                if feature_id and feature_id not in ordered:
                    ordered.append(feature_id)
        except Exception as exc: raise AlignmentError('Frozen old manifest unreadable') from exc
        if len(ordered)!=807 or tuple(ordered)!=tuple(ids): raise CacheContractError('Old cache IDs/order disagree with frozen 807 manifest')
    return ids

def _expected_family(family_id):
    spec=get_family(family_id); recipe=build_recipe_identity(spec,dict(spec.default_config),Path(__file__).resolve().parents[1])
    return spec,recipe.recipe_id,tuple(item.feature_id for item in spec.feature_definitions)

def _family_index(root:Path,family_id:str,recipe_id:str,feature_ids:tuple[str,...]):
    folder=root/family_id/recipe_id
    files=sorted(folder.glob('*.npz'))
    if len(files)!=6600: raise CacheContractError(f'{family_id} expected 6600 caches, found {len(files)}')
    index={}
    for path in files:
        try: payload=read_cache(path)
        except Exception as exc: raise CacheContractError(f'Invalid family cache: {path}') from exc
        if _scalar(payload,'family_id')!=family_id or _scalar(payload,'recipe_id')!=recipe_id: raise CacheContractError(f'Family identity mismatch: {path}')
        ids=tuple(str(x) for x in payload['feature_ids'].tolist())
        if ids!=feature_ids: raise CacheContractError(f'Feature order mismatch: {path}')
        ident=_load_json(payload,'source_identity_json')
        if ident.get('site_id') is None or ident.get('record_id') is None or ident.get('source_sha256') is None: raise SourceIdentityMismatch(f'Incomplete source identity: {path}')
        key=(str(ident['site_id']),str(ident['record_id']))
        if key in index: raise AlignmentError(f'Duplicate family identity {family_id}: {key}')
        index[key]=(path,ident)
    return index

def discover_alignment(old_cache_root:Path,new_cache_root:Path,old_manifest_path:Path|None=None,family_ids:Sequence[str]=DEFAULT_FAMILY_IDS):
    old_root=Path(old_cache_root); new_root=Path(new_cache_root)
    rows=_old_rows(old_root)
    if len(rows)!=6600: raise AlignmentError(f'Expected 6600 old rows, found {len(rows)}')
    old_ids=_old_feature_contract(rows,old_manifest_path)
    expected={}
    family_maps={}
    for family_id in tuple(family_ids):
        spec,recipe_id,ids=_expected_family(family_id); expected[family_id]=(recipe_id,ids); family_maps[family_id]=_family_index(new_root,family_id,recipe_id,ids)
    old_keys={(site,record) for _,site,record,_,_ in rows}
    for family_id, mapping in family_maps.items():
        keys=set(mapping)
        if keys!=old_keys: raise AlignmentError(f'Identity set mismatch for {family_id}')
    old_sources={}
    for i,site,record,relative,old_path in rows:
        _,_,source_sha=_validate_old_cache(old_path,old_ids)
        old_sources[(site,record)]=source_sha
    entries=[]; site_counts={}
    for i,site,record,relative,old_path in rows:
        old_cache_sha=old_sources[(site,record)]
        family_paths={}; record_key=''
        for family_id,mapping in family_maps.items():
            path,ident=mapping[(site,record)]
            if str(ident['source_sha256'])!=old_cache_sha: raise SourceIdentityMismatch(f'Source SHA mismatch {family_id} {site}/{record}')
            family_paths[family_id]=path
            if not record_key: record_key=str(ident.get('record_key',''))
        if not record_key.startswith('rec_'): raise SourceIdentityMismatch(f'Invalid record key: {site}/{record}')
        entries.append(AlignmentEntry(i,site,record,relative,old_path,old_cache_sha,family_paths,record_key)); site_counts[site]=site_counts.get(site,0)+1

    return Alignment(tuple(entries),tuple(old_ids),{fam:ids for fam,(_,ids) in expected.items()},{fam:rid for fam,(rid,_) in expected.items()},dict(site_counts))

def read_aligned_record(entry:AlignmentEntry,alignment:Alignment):
    p,ids,source=_validate_old_cache(entry.old_cache_path,alignment.old_feature_ids)
    if source!=entry.old_source_sha256: raise SourceIdentityMismatch(f'Old source changed: {entry.old_cache_path}')
    blocks={}
    for family_id,path in entry.family_paths.items():
        payload=read_cache(path); ident=_load_json(payload,'source_identity_json')
        if (str(ident.get('site_id')),str(ident.get('record_id')))!=(entry.site_id,entry.record_id) or str(ident.get('source_sha256'))!=entry.old_source_sha256: raise SourceIdentityMismatch(f'Family source mismatch: {path}')
        blocks[family_id]={'values':np.asarray(payload['values'],dtype=np.float64),'valid_mask':np.asarray(payload['valid_mask'],dtype=bool),'support_count':np.asarray(payload['support_count'],dtype=np.int64),'missing_reason':tuple(str(x) for x in payload['missing_reason'].tolist()),'feature_ids':tuple(str(x) for x in payload['feature_ids'].tolist())}
    return {'old_values':np.asarray(p['unique_values'],dtype=np.float64),'old_valid_mask':np.asarray(p['unique_valid_mask'],dtype=bool),'old_support_count':np.asarray(p['unique_support_count'],dtype=np.int64),'family_blocks':blocks}
