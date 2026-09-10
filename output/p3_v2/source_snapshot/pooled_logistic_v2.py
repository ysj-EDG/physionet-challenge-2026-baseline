"""Shared P3 v2 stage-aware EEG pooling and linear preprocessing."""
from __future__ import annotations
import numpy as np

MODEL_TYPE='pooled_logistic_v2'
COMPACT_INDICES=[11,12,13,14,16,17,18,19,20,32,34,87,96,97,98,126,127,141,168,173]
METRICS=[('log_theta_beta',3),('log_delta_beta',4),('log_sigma_beta',1),('sef50',5)]
REGIONS=[('F',(0,1)),('C',(2,3)),('O',(4,5))]
STAGES=[('N1',3),('N2',2),('N3',1),('REM',4),('Wake',5)]

def pooled_names(kind):
 targets=[('global',None)] if kind=='global' else STAGES
 return [f"eeg_{target}_{region}_{metric}_{stat}" for target,_ in targets
         for metric,_ in METRICS for region,_ in REGIONS for stat in ('median','iqr')]

def pool_eeg(x_seq,stage_code,stage_valid,channel_available,kind):
 x=np.asarray(x_seq,float); stage=np.asarray(stage_code,float)[:len(x)]
 valid=np.asarray(stage_valid,bool)[:len(x)]; available=np.asarray(channel_available,bool)
 targets=[('global',None)] if kind=='global' else STAGES; out=[]
 for _,code in targets:
  use=valid if code is None else (valid & (stage==code))
  for _,offset in METRICS:
   for _,channels in REGIONS:
    medians=[]; iqrs=[]
    for ch in channels:
     if ch>=len(available) or not available[ch] or use.sum()<3: continue
     values=x[use,ch*9+offset]
     values=values[np.isfinite(values)]
     if len(values)<3: continue
     medians.append(np.median(values)); iqrs.append(np.percentile(values,75)-np.percentile(values,25))
    out.extend([np.median(medians) if medians else np.nan,
                np.median(iqrs) if iqrs else np.nan])
 return np.asarray(out,dtype=np.float64)

def assemble_features(transformed_static,stage_code,stage_valid,channel_available,x_seq,arm):
 static=np.asarray(transformed_static,float)
 demo=static[:10]; compact=static[COMPACT_INDICES]
 present=np.asarray([np.any(np.asarray(stage_valid,bool)&(np.asarray(stage_code)==code)) for _,code in STAGES],float)
 base=np.r_[demo,compact]
 if arm=='P3_demo10': return demo
 if arm=='P3_compact30': return base
 if arm=='P3_compact35': return np.r_[base,present]
 if arm=='P3_global59': return np.r_[base,present,pool_eeg(x_seq,stage_code,stage_valid,channel_available,'global')]
 if arm=='P3_stage155': return np.r_[base,present,pool_eeg(x_seq,stage_code,stage_valid,channel_available,'stage')]
 raise ValueError(arm)

def fit_imputer_scaler(x):
 x=np.asarray(x,np.float64); med=np.zeros(x.shape[1]); all_missing=np.zeros(x.shape[1],bool)
 for j in range(x.shape[1]):
  finite=np.isfinite(x[:,j]); all_missing[j]=not finite.any(); med[j]=np.median(x[finite,j]) if finite.any() else 0
 filled=np.where(np.isfinite(x),x,med); mean=filled.mean(0); scale=filled.std(0); scale=np.where(scale>0,scale,1)
 return med,mean,scale,all_missing

def apply_imputer_scaler(x,med,mean,scale):
 return (np.where(np.isfinite(x),x,med)-mean)/scale
