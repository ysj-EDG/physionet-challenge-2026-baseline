"""Direct DATA2 temporal inputs, validity semantics, and T0 audits."""
from __future__ import annotations
import hashlib, json, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import feature_scaling as scaling
from per_epoch_features.feature_extractor_algorithmic import AlgorithmicMixin
SITES=('I0002','I0006','S0001'); STAGE_SLICE=slice(470,483); EEG_INDICES=tuple(range(54))+tuple(range(414,432))
STAGE_FEATURE_NAMES=('N1','N2','N3','REM','Wake','arousal_fraction','OA','CA','MA','HY','RERA','limb_isolated','PLM')
STAGE_STATIC_INDICES=(11,12,13,14,16,17,18,19,20,32,34); ALGO_SLEEP_INDICES=(1,2,3,4,6,7,8,9,10,22,24)
EVENT_KEYS=(('arousal_caisr',5,6),('resp_caisr',6,11),('limb_caisr',11,13))
@dataclass
class TemporalRecord:
 record_id:str; patient_id:str; site:str; label:int; age:float; values:np.ndarray; valid:np.ndarray; n_epochs:int; n_blocks:int; dropped_tail_epochs:int

def _json_scalar(z,key): return json.loads(str(z[key].item()))
def event_epoch_validity(starts,durations,lengths,freqs,key):
 length=int(lengths.get(key,0) or 0)
 if length<=0: return np.zeros(len(starts),bool),{'missing':True,'partial':False,'trailing_invalid':len(starts),'coverage_sec':0.}
 try: frequency=float(freqs.get(key,0.) or 0.)
 except (TypeError,ValueError): frequency=0.
 if not np.isfinite(frequency) or frequency<=0: raise ValueError(f'{key}: positive signal length {length} but invalid sampling frequency {frequency!r}')
 coverage=length/frequency; valid=(starts>=0)&((starts+durations)<=coverage+1e-9); return valid,{'missing':False,'partial':bool(valid.any() and not valid.all()),'trailing_invalid':int((~valid).sum()),'coverage_sec':float(coverage)}
def blockify(values):
 n=len(values); blocks=n//10
 if blocks<1: raise ValueError(f'At least 10 physical epochs required, got {n}')
 return np.asarray(values[:blocks*10]).reshape(blocks,10,values.shape[1]),n-blocks*10

def deterministic_permutation(record_id,block_index):
 payload=f'{record_id}{block_index}20260918_local_shuffle'.encode(); seed=int.from_bytes(hashlib.sha256(payload).digest()[:8],'little'); return np.random.default_rng(seed).permutation(10)
def apply_block_order(blocks,record_id,shuffle):
 if not shuffle:return blocks
 return np.stack([block[deterministic_permutation(record_id,i)] for i,block in enumerate(blocks)])

def _selected(path,modality,run_audit=False):
 with np.load(path,allow_pickle=False) as z:
  rid=str(z['record_id'].item()); site=str(z['site_id'].item()); pid=str(z['bdsp_patient_id'].item()); version=str(z['extraction_version'].item()); label=int(z['y'].item()); static=np.asarray(z['x_static'],np.float64); starts=np.asarray(z['epoch_start_sec'],np.float64); durations=np.asarray(z['epoch_duration_sec'],np.float64); n=len(starts)
  if version!='timegrid_v2.0.0' or path.stem!=rid or path.parent.name!=site or site not in SITES or static.shape!=(196,) or label not in (0,1): raise RuntimeError(f'Identity/version/dimension failure: {path}')
  lengths=_json_scalar(z,'annotation_signal_lengths_json'); freqs=_json_scalar(z,'annotation_sampling_frequencies_json')
  event={}; event_meta={}
  for key,a,b in EVENT_KEYS: event[key],event_meta[key]=event_epoch_validity(starts,durations,lengths,freqs,key)
  if modality=='stage_event13':
   values=np.array(z['X_seq'][:,STAGE_SLICE],dtype=np.float32,copy=True); stage=np.asarray(z['stage_valid'],bool); valid=np.zeros((n,13),bool); valid[:,:5]=stage[:,None]
   for key,a,b in EVENT_KEYS: valid[:,a:b]=event[key][:,None]
  elif modality=='eeg72':
   x=np.asarray(z['X_seq'],np.float32); values=np.concatenate([x[:,:54],x[:,414:432]],axis=1); avail=np.asarray(z['eeg_channel_available'],bool); clean=np.asarray(z['eeg_clean_subsegment_count']); total=np.asarray(z['eeg_total_subsegment_count']); spec_valid=(avail[None,:]&(total>0)&(clean>0)); valid=np.concatenate([np.repeat(spec_valid,9,axis=1),np.tile(avail,3)[None,:].repeat(n,axis=0)],axis=1)
  else: raise ValueError(modality)
  audit=None
  if run_audit:
   stage_code=np.asarray(z['stage_code_aligned'],np.float64); stage_valid=np.asarray(z['stage_valid'],bool); reconstructed=np.asarray(AlgorithmicMixin().extract_algorithmic_annotations_features({'__extraction_mode__':'timegrid_v2','__aligned_stage_code__':stage_code}),np.float64)[list(ALGO_SLEEP_INDICES)]; existing=static[list(STAGE_STATIC_INDICES)]; audit={'stage_valid':stage_valid,'events':event,'event_meta':event_meta,'reconstructed':reconstructed,'existing':existing}
 return TemporalRecord(rid,pid,site,label,float(static[0]),values,valid,n,n//10,n%10),audit

def load_dataset(data_root,modality,run_audit=False):
 paths=sorted(Path(data_root).glob('*/*.npz'),key=lambda p:(p.parent.name,p.stem)); start=time.time()
 if len(paths)!=6600: raise RuntimeError(f'Expected 6600 NPZ, got {len(paths)}')
 records=[]; audits=[]
 for i,path in enumerate(paths,1):
  rec,audit=_selected(path,modality,run_audit); records.append(rec); audits.append(audit)
  if i%1000==0: print(f'LOAD {modality} {i}/6600',flush=True)
 ids=[r.record_id for r in records]; patients=[r.patient_id for r in records]
 if len(set(ids))!=6600 or len(set(patients))!=6600: raise RuntimeError('Record/patient identity is not unique')
 expected={'I0002':(319,52),'I0006':(1142,112),'S0001':(5139,334)}
 for site,(n,p) in expected.items():
  subset=[r for r in records if r.site==site]
  if len(subset)!=n or sum(r.label for r in subset)!=p: raise RuntimeError(f'Site counts failed: {site}')
 unusable=[r.record_id for r in records if r.n_blocks<1]
 if unusable: print(f'UNUSABLE_NO_COMPLETE_BLOCK {len(unusable)} first={unusable[:10]}',flush=True)
 return records,audits,time.time()-start

def run_stage_event_audit(data_root,out_path):
 records,audits,elapsed=load_dataset(data_root,'stage_event13',True); bysite={}; errors={name:{'max_abs_error':0.,'sum_abs_error':0.,'values':0,'mismatch_count':0} for name in ('TST','sleep_efficiency','SOL','REM_latency','WASO','N1_pct','N2_pct','N3_pct','REM_pct','transition_rate','short_bout_ratio')}
 for site in SITES:
  indices=[i for i,r in enumerate(records) if r.site==site]; total=sum(records[i].n_epochs for i in indices); item={'n_records':len(indices),'total_physical_epochs':total,'stage_valid_epochs':0,'stage_valid_rate':0.}
  for key,_,_ in EVENT_KEYS: item[key]={'valid_epochs':0,'valid_rate':0.,'missing_record_count':0,'partial_coverage_record_count':0,'trailing_invalid_epoch_count':0}
  for i in indices:
   a=audits[i]; item['stage_valid_epochs']+=int(a['stage_valid'].sum())
   for key,_,_ in EVENT_KEYS:
    item[key]['valid_epochs']+=int(a['events'][key].sum()); item[key]['missing_record_count']+=int(a['event_meta'][key]['missing']); item[key]['partial_coverage_record_count']+=int(a['event_meta'][key]['partial']); item[key]['trailing_invalid_epoch_count']+=a['event_meta'][key]['trailing_invalid']
  item['stage_valid_rate']=item['stage_valid_epochs']/total
  for key,_,_ in EVENT_KEYS:item[key]['valid_rate']=item[key]['valid_epochs']/total
  bysite[site]=item
 for audit in audits:
  diff=np.abs(audit['reconstructed']-audit['existing'])
  for j,name in enumerate(errors):
   errors[name]['max_abs_error']=max(errors[name]['max_abs_error'],float(diff[j])); errors[name]['sum_abs_error']+=float(diff[j]); errors[name]['values']+=1; errors[name]['mismatch_count']+=int(not np.isclose(audit['reconstructed'][j],audit['existing'][j],atol=1e-5,rtol=1e-6))
 for value in errors.values(): value['mean_abs_error']=value.pop('sum_abs_error')/value.pop('values')
 passed=all(v['mismatch_count']==0 for v in errors.values()); payload={'dataset':'DATA2','records':6600,'extraction_version':'timegrid_v2.0.0','wall_time_sec':elapsed,'unusable_no_complete_block':[r.record_id for r in records if r.n_blocks<1],'validity_by_site':bysite,'reconstruction':{'atol':1e-5,'rtol':1e-6,'features':errors,'pass':passed},'sampling_frequency_errors':0}
 Path(out_path).parent.mkdir(parents=True,exist_ok=True); Path(out_path).write_text(json.dumps(payload,indent=2)+'\n')
 if not passed: raise RuntimeError('Stage reconstruction audit failed')
 return records,payload

def fit_neutral(records,indices):
 dim=records[indices[0]].values.shape[1]; sums=np.zeros(dim,np.float64); counts=np.zeros(dim,np.int64)
 for i in indices:
  r=records[i]; sums+=(np.where(r.valid,r.values,0.)).sum(0); counts+=r.valid.sum(0)
 if np.any(counts==0): raise RuntimeError(f'No valid training values for columns {np.flatnonzero(counts==0)}')
 return sums/counts
def impute_values(record,neutral): return np.where(record.valid,record.values,neutral).astype(np.float32)
