#!/usr/bin/env python
from __future__ import annotations
import json, os, subprocess, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from evaluate_model import compute_auroc_age
from feature_scaling import FeatureScaler
from pooled_logistic_v2 import *

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'output'/'p3_v2'; CACHE=ROOT/'npz_new'; SIDE=OUT/'stage_sidecar'
ARMS=['P3_demo10','P3_compact30','P3_compact35','P3_global59','P3_stage155']
DIMS=[10,30,35,59,155]; RULES=ROOT/'feat_input'/'feature_rules_v1.json'
LR=dict(penalty='elasticnet',solver='saga',l1_ratio=.4,C=.03,class_weight='balanced',fit_intercept=True,max_iter=100000,tol=1e-4,random_state=7)

def rid(r): return f"{r['BidsFolder']}_ses-{r['SessionID']}"
def records(split): return json.loads((ROOT/'split'/f'{split}_records.json').read_text())
def cpath(split,r): return CACHE/split/f'{rid(r)}.npz'
def load_one(split,r,pre):
 with np.load(cpath(split,r),allow_pickle=False) as z:
  seq=np.asarray(z['X_seq']); ecg=np.asarray(z['X_ecg']); static=np.asarray(z['x_static']); y=int(z['y'])
 age=float(static[0]); seq,_,static=pre.transform_arrays(seq,ecg,static,record_id=rid(r))
 with np.load(SIDE/f'{rid(r)}.npz',allow_pickle=False) as s:
  stage=s['stage_code']; valid=s['stage_valid']; channels=s['eeg_channel_available']
 return seq,static,y,age,stage,valid,channels

def matrices(split,recs,pre):
 data={arm:[] for arm in ARMS}; labels=[]; ages=[]
 for r in recs:
  seq,static,y,age,stage,valid,ch=load_one(split,r,pre)
  for arm in ARMS: data[arm].append(assemble_features(static,stage,valid,ch,seq,arm))
  labels.append(y); ages.append(age)
 labels=np.asarray(labels); ages=np.asarray(ages)
 return {arm:(np.asarray(data[arm],float),labels,ages) for arm in ARMS}
def metrics(y,logit,age):
 return {'ac_auroc':float(compute_auroc_age(y,logit,age,gap=2)),'auroc':float(roc_auc_score(y,logit)),'auprc':float(average_precision_score(y,logit))}
def fit_lr(x,y):
 model=LogisticRegression(**LR)
 with warnings.catch_warnings(record=True) as w:
  warnings.simplefilter('always',ConvergenceWarning); model.fit(x,y)
 return model,[str(q.message) for q in w if issubclass(q.category,ConvergenceWarning)]
def calibrate(logit,y):
 m=LogisticRegression(solver='lbfgs',max_iter=1000).fit(np.asarray(logit)[:,None],y)
 return {'coef':float(m.coef_[0,0]),'intercept':float(m.intercept_[0])}
def probs(logit,c): return 1/(1+np.exp(-np.clip(c['coef']*logit+c['intercept'],-50,50)))
def threshold(y,p):
 f,t,z=roc_curve(y,p); ok=np.isfinite(z); idx=np.flatnonzero(ok)[np.argmax((t-f)[ok])]; return float(z[idx])

def feature_names(pre,arm):
 names={int(r['index']):r['name'] for r in pre.rules if r['branch']=='x_static'}
 demo=[names[i] for i in range(10)]; compact=[names[i] for i in COMPACT_INDICES]; present=[f'stage_present_{s}' for s,_ in STAGES]
 if arm=='P3_demo10': return demo
 if arm=='P3_compact30': return demo+compact
 if arm=='P3_compact35': return demo+compact+present
 if arm=='P3_global59': return demo+compact+present+pooled_names('global')
 return demo+compact+present+pooled_names('stage')

def save_decisions(path,recs,y,age,logit,role):
 pd.DataFrame({'record_id':[rid(r) for r in recs],'role':role,'label':y,'raw_age':age,'decision_logit':logit}).to_csv(path,index=False)

def run_cv(train):
 manifest=pd.read_csv(SIDE/'manifest.csv').set_index('record_id'); all_rows=[]
 for seed in (7,17,29):
  y=np.array([int(np.load(cpath('train',r),allow_pickle=False)['y']) for r in train]); groups=np.array([str(manifest.loc[rid(r),'bdsp_patient_id']) for r in train]); strata=np.array([f"{r['SiteID']}_{v}" for r,v in zip(train,y)])
  cv=StratifiedGroupKFold(3,shuffle=True,random_state=seed)
  for fold,(ti,vi) in enumerate(cv.split(np.zeros(len(y)),strata,groups)):
   tr=[train[i] for i in ti]; va=[train[i] for i in vi]
   fold_dir=OUT/'cv'/f'seed{seed}'/f'fold{fold}'; fold_dir.mkdir(parents=True)
   pre=FeatureScaler.fit_from_cache(tr,CACHE/'train',RULES,seed=20260907)
   train_m=matrices('train',tr,pre); hold_m=matrices('train',va,pre)
   for arm in ARMS:
    a,b,agea=train_m[arm]; c,d,agec=hold_m[arm]
    med,mean,scale,missing=fit_imputer_scaler(a); aa=apply_imputer_scaler(a,med,mean,scale); cc=apply_imputer_scaler(c,med,mean,scale)
    model,w=fit_lr(aa,b); la=model.decision_function(aa); lc=model.decision_function(cc)
    save_decisions(fold_dir/f'{arm}_train.csv',tr,b,agea,la,'train'); save_decisions(fold_dir/f'{arm}_holdout.csv',va,d,agec,lc,'holdout')
    row={'seed':seed,'fold':fold,'arm':arm,'train_n':len(tr),'holdout_n':len(va),'train_pos':int(b.sum()),'holdout_pos':int(d.sum()),'n_iter':int(model.n_iter_[0]),'converged':not w,'warnings':w,'all_missing_columns':np.flatnonzero(missing).tolist(),'train':metrics(b,la,agea),'holdout':metrics(d,lc,agec)}; all_rows.append(row); print(json.dumps(row),flush=True)
    torch.save({'model_type':MODEL_TYPE,'scope':'cv','seed':seed,'fold':fold,'arm':arm,'feature_names':feature_names(pre,arm),'input_preprocessing':pre.state_dict(),'imputer_median':med,'linear_mean':mean,'linear_scale':scale,'coefficients':model.coef_[0],'intercept':float(model.intercept_[0]),'linear_model':{'config':LR,'n_iter':int(model.n_iter_[0]),'converged':not w,'warnings':w},'train_ids':[rid(r) for r in tr],'holdout_ids':[rid(r) for r in va]},fold_dir/f'{arm}.pt')
 (OUT/'cv_results.json').write_text(json.dumps(all_rows,indent=2))
 return all_rows

def final_fit(train):
 val=records('val'); pre=FeatureScaler.fit_from_cache(train,CACHE/'train',RULES,seed=20260907)
 split_records={'train':train,'val':val,'test':records('test'),'external':records('external')}
 split_matrices={name:matrices(name,recs,pre) for name,recs in split_records.items()}
 for arm in ARMS:
  ad=OUT/'final'/arm; ad.mkdir(parents=True)
  x,y,age=split_matrices['train'][arm]; xv,yv,agev=split_matrices['val'][arm]
  med,mean,scale,missing=fit_imputer_scaler(x); model,w=fit_lr(apply_imputer_scaler(x,med,mean,scale),y)
  lv=model.decision_function(apply_imputer_scaler(xv,med,mean,scale)); cal=calibrate(lv,yv); thr=threshold(yv,probs(lv,cal))
  ck={'model_type':MODEL_TYPE,'scope':'final','arm':arm,'feature_names':feature_names(pre,arm),'input_preprocessing':pre.state_dict(),'imputer_median':med,'linear_mean':mean,'linear_scale':scale,'coefficients':model.coef_[0],'intercept':float(model.intercept_[0]),'calibrator':cal,'threshold':thr,'sidecar_profile':'legacy_npz_new+official_caisr_relative_prefix_v1','train_ids':[rid(r) for r in train],'linear_model':{'config':LR,'n_iter':int(model.n_iter_[0]),'converged':not w,'warnings':w},'all_missing_columns':np.flatnonzero(missing).tolist()}; torch.save(ck,ad/'lstm_model.pt')
  for split,recs in split_records.items():
   xx,yy,ages=split_matrices[split][arm]; logits=model.decision_function(apply_imputer_scaler(xx,med,mean,scale)); pp=probs(logits,cal); od=ad/split; od.mkdir(); save_decisions(od/'decision_outputs.csv',recs,yy,ages,logits,split)
   pd.DataFrame({'record_id':[rid(r) for r in recs],'decision_logit':logits,'calibrated_probability':pp,'binary_prediction':pp>=thr,'raw_age':ages}).to_csv(od/'predictions.csv',index=False)

def main():
 if (OUT/'cv').exists() or (OUT/'final').exists(): raise FileExistsError('P3 CV/final output exists')
 train=records('train'); rows=run_cv(train); final_fit(train)
if __name__=='__main__': main()
