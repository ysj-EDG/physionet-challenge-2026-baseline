#!/usr/bin/env python
from __future__ import annotations
import hashlib, json
from pathlib import Path
import edfio, numpy as np, pandas as pd
from per_epoch_features.per_epoch_extractor import PerEpochExtractor

ROOT=Path(__file__).resolve().parents[2]
DATA=Path('/database2/physionet2026_kaggle/data')
OUT=ROOT/'output'/'p3_v2'/'stage_sidecar'
STAGES={1,2,3,4,5}; CODE_BY_COL=np.array([3,2,1,4,5])

def key(r): return f"{r['BidsFolder']}_ses-{r['SessionID']}"
def safe(edf,name):
 try: return str(getattr(edf,name))
 except Exception: return ''
def cache_path(split,rid): return ROOT/'npz_new'/split/f'{rid}.npz'
def stage_path(r,rid): return DATA/'algorithmic_annotations'/str(r['SiteID'])/f'{rid}_caisr_annotations.edf'
def phys_path(r,rid): return DATA/'physiological_data'/str(r['SiteID'])/f'{rid}.edf'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 if OUT.exists(): raise FileExistsError(f'Refusing to overwrite {OUT}')
 OUT.mkdir(parents=True)
 demo=pd.read_csv(DATA/'demographics.csv',dtype={'BidsFolder':str,'SessionID':str,'BDSPPatientID':str})
 demo_map={(str(r.BidsFolder),str(r.SessionID)):r for _,r in demo.iterrows()}
 extractor=PerEpochExtractor(); rows=[]
 for split in ('train','val','test','external'):
  records=json.loads((ROOT/'split'/f'{split}_records.json').read_text())
  for r in records:
   r={k:str(v) if k in ('BidsFolder','SessionID') else v for k,v in r.items()}; rid=key(r)
   cp=cache_path(split,rid)
   with np.load(cp,allow_pickle=False) as z:
    n=len(z['X_seq']); cached=np.asarray(z['X_seq'][:,470:475])
   ap=stage_path(r,rid); pp=phys_path(r,rid)
   raw=np.array([],float); stage_fs=np.nan; algo_duration=np.nan; algo_start=''; annotation='missing'
   if ap.is_file():
    ae=edfio.read_edf(ap,lazy_load_data=False); sig=next((s for s in ae.signals if s.label.lower().strip()=='stage_caisr'),None)
    if sig is not None:
     raw=np.asarray(sig.data,float).reshape(-1); stage_fs=float(sig.sampling_frequency); annotation='available'
    algo_duration=float(ae.duration); algo_start=safe(ae,'startdate')+'T'+safe(ae,'starttime')
   aligned=np.full(n,np.nan,np.float32); aligned[:min(n,len(raw))]=raw[:n]
   valid=np.isin(aligned,list(STAGES)); raw_oh=(aligned[:,None]==CODE_BY_COL).astype(np.float32)
   compressed=raw[np.isin(raw,list(STAGES))][:n]
   comp_oh=np.zeros((n,5),np.float32); comp_oh[:len(compressed)]=(compressed[:,None]==CODE_BY_COL)
   match_raw=bool(np.array_equal(cached,raw_oh)); match_compressed=bool(np.array_equal(cached,comp_oh))
   channel=np.zeros(6,bool); phys_duration=np.nan; phys_start=''
   if pp.is_file():
    pe=edfio.read_edf(pp,lazy_load_data=True); labels=[s.label for s in pe.signals]
    rename,_=extractor._standardize_channels(labels); std={rename.get(x,x.lower().strip()) for x in labels}
    specs=[('f3-m2','f3','m2'),('f4-m1','f4','m1'),('c3-m2','c3','m2'),('c4-m1','c4','m1'),('o1-m2','o1','m2'),('o2-m1','o2','m1')]
    channel=np.array([target in std or (pos in std and ref in std) for target,pos,ref in specs])
    phys_duration=float(pe.duration); phys_start=safe(pe,'startdate')+'T'+safe(pe,'starttime')
   status='reliable_official_relative_prefix' if pp.is_file() and annotation=='available' and np.isclose(stage_fs,1/30) else ('annotation_missing' if annotation=='missing' else 'unreliable')
   d=demo_map[(str(r['BidsFolder']),str(r['SessionID']))]
   np.savez_compressed(OUT/f'{rid}.npz',record_id=np.asarray(rid),epoch_start_sec=np.arange(n,dtype=np.float32)*30,stage_code=aligned,stage_valid=valid,eeg_channel_available=channel,alignment_status=np.asarray(status))
   rows.append({'split':split,'record_id':rid,'site_id':str(r['SiteID']),'bdsp_patient_id':str(d.BDSPPatientID),'n_cache':n,'n_raw_stage':len(raw),'valid_epochs':int(valid.sum()),'invalid_epochs':int((~valid).sum()),'annotation_status':annotation,'alignment_status':status,'cached_onehot_matches_raw_prefix':match_raw,'cached_onehot_matches_compressed':match_compressed,'eeg_channels_available':int(channel.sum()),'phys_duration_sec':phys_duration,'algo_duration_sec':algo_duration,'phys_start':phys_start,'algo_start':algo_start})
 pd.DataFrame(rows).to_csv(OUT/'manifest.csv',index=False)
 summary={'records':len(rows),'by_split':pd.Series([x['split'] for x in rows]).value_counts().to_dict(),'alignment':pd.Series([x['alignment_status'] for x in rows]).value_counts().to_dict(),'cached_matches_raw_prefix':sum(x['cached_onehot_matches_raw_prefix'] for x in rows),'cached_matches_compressed':sum(x['cached_onehot_matches_compressed'] for x in rows),'records_with_invalid_in_cache_prefix':sum(x['invalid_epochs']>0 for x in rows),'source_data':str(DATA),'source_demographics_sha256':sha(DATA/'demographics.csv')}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__': main()
