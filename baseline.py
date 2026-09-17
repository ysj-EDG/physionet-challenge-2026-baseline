#!/usr/bin/env python3
"""Standalone CAISR20-only DATA2 three-site LOSO baseline.

Reads frozen timegrid_v2 NPZ files directly.  It never reads temporal arrays,
persistent feature caches, or historical manifests for model construction.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, platform, shutil, subprocess, sys, warnings
from pathlib import Path
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ.setdefault(_k,'1')
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import feature_scaling as scaling
from evaluate_model import compute_auroc_age, compute_auroc_weighted

SITES=('I0002','I0006','S0001')
CAISR20_INDICES=(11,12,13,14,16,17,18,19,20,32,34,87,96,97,98,126,127,141,168,173)
FEATURE_GROUPS={
 'sleep_architecture':(11,12,13,14,16,17,18,19,20),
 'fragmentation_uncertainty':(32,34,87), 'arousal':(96,97,98),
 'respiratory':(126,127,141), 'limb':(168,173)}
EXPECTED_NAMES=(
 'caisr_sleep_tst_sec','caisr_sleep_se','caisr_sleep_sol_sec','caisr_sleep_rem_latency_sec','caisr_sleep_waso_sec',
 'caisr_sleep_n1_pct','caisr_sleep_n2_pct','caisr_sleep_n3_pct','caisr_sleep_rem_pct','caisr_sleep_transition_rate',
 'caisr_sleep_short_bout_ratio','caisr_sleep_mean_stage_entropy','caisr_arousal_arousal_index',
 'caisr_arousal_arousal_burden_ratio','caisr_arousal_mean_arousal_duration','caisr_respiratory_respiratory_event_index',
 'caisr_respiratory_respiratory_burden_ratio','caisr_respiratory_mean_resp_duration','caisr_limb_limb_movement_index','caisr_limb_plmi')
LR_CONFIG={'penalty':'elasticnet','solver':'saga','l1_ratio':.4,'C':.03,'class_weight':'balanced','fit_intercept':True,'max_iter':100000,'tol':1e-4}
ORACLE={'I0002':.7460233297985154,'I0006':.6777064789898571,'S0001':.6166620961991992,'macro':.6801306349958572,'worst':.6166620961991992}
HISTORICAL_COMMIT='79adc97e180e9d38f3743fbf8d10f18e6b5b11cc'
STABILITY_SEEDS=(7,17,27,37,47)

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def git_head(): return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def scalar(z,key):
 if key not in z: raise RuntimeError(f'NPZ missing required key {key}')
 return z[key].item()
def feature_group(index):
 matches=[name for name,indices in FEATURE_GROUPS.items() if index in indices]
 if len(matches)!=1: raise RuntimeError(f'Feature index {index} group mapping is not unique')
 return matches[0]
def load_rules(path):
 rules=scaling.load_feature_rules(path); lookup={(r['branch'],int(r['index'])):r for r in rules}
 names=tuple(lookup['x_static',i]['name'] for i in CAISR20_INDICES)
 if names!=EXPECTED_NAMES: raise RuntimeError(f'Frozen CAISR20 name mismatch: {names}')
 if len(set(CAISR20_INDICES))!=20 or set().union(*map(set,FEATURE_GROUPS.values()))!=set(CAISR20_INDICES): raise RuntimeError('CAISR20 group/index invariant failed')
 if any(n in {'age','BMI'} or n.startswith(('sex_','race_')) for n in names): raise RuntimeError('Demographic feature entered CAISR20')
 return rules,lookup,names

def load_dataset(data_root):
 paths=sorted(Path(data_root).glob('*/*.npz'),key=lambda p:(p.parent.name,p.stem))
 if len(paths)!=6600: raise RuntimeError(f'Expected 6600 NPZ, found {len(paths)}')
 rows=[]; x=[]
 for number,path in enumerate(paths,1):
  with np.load(path,allow_pickle=False) as z:
   static=np.asarray(z['x_static'],dtype=np.float64); y=int(scalar(z,'y')); rid=str(scalar(z,'record_id')); site=str(scalar(z,'site_id')); pid=str(scalar(z,'bdsp_patient_id')); version=str(scalar(z,'extraction_version'))
  if path.stem!=rid or path.parent.name!=site: raise RuntimeError(f'Path/identity mismatch: {path}')
  if static.shape!=(196,) or not np.isfinite(static).all(): raise RuntimeError(f'Invalid x_static: {rid}')
  if y not in (0,1) or site not in SITES or version!='timegrid_v2.0.0': raise RuntimeError(f'Invalid label/site/version: {rid}')
  sentinel=bool(np.isfinite(static[10:]).all() and np.all(static[10:]==0))
  rows.append({'record_id':rid,'patient_id':pid,'site':site,'label':y,'age':float(static[0]),'npz_path':str(path),'extraction_version':version,'caisr_block_missing':sentinel}); x.append(static)
  if number%1000==0: print(f'LOAD {number}/6600',flush=True)
 m=pd.DataFrame(rows); x=np.asarray(x,np.float64)
 if m.record_id.nunique()!=6600 or m.patient_id.nunique()!=6600: raise RuntimeError('Record/patient identity is not unique')
 expected={'I0002':(319,52,267),'I0006':(1142,112,1030),'S0001':(5139,334,4805)}
 for site,want in expected.items():
  g=m[m.site==site]; got=(len(g),int(g.label.sum()),int((g.label==0).sum()))
  if got!=want: raise RuntimeError(f'{site} count mismatch: {got} != {want}')
 if int(m.label.sum())!=498 or int((m.label==0).sum())!=6102: raise RuntimeError('Total label count mismatch')
 return m,x

def fit_typed_static_transform(raw_train,sentinel_train,indices,rules):
 params=[]
 for index in indices:
  rule=rules['x_static',index]; prepared,valid,_=scaling._prepare_values(raw_train[:,index],rule); valid=np.asarray(valid)&~sentinel_train
  vals=prepared[valid]; weights=np.ones(len(vals),np.float64)
  if rule['scaling']=='identity': center,scale,method=0.,1.,'identity'
  elif not len(vals): center,scale,method=0.,1.,'all_missing_fallback'
  else:
   q05,q25,q50,q75,q95=[scaling._weighted_quantile(vals,weights,q) for q in (.05,.25,.5,.75,.95)]; center=q50; tolerance=1e-6*max(1.,abs(center)); mean=np.average(vals,weights=weights)
   candidates=(('iqr',q75-q25),('p95_p05',q95-q05),('std',float(np.sqrt(np.average((vals-mean)**2,weights=weights)))))
   method,scale='unit_fallback',1.
   for candidate,value in candidates:
    if np.isfinite(value) and value>tolerance: method,scale=candidate,float(value); break
  params.append({'index':index,'center':float(center),'scale':float(scale),'scale_method':method,'fit_valid':int(valid.sum())})
 return params

def apply_typed_static_transform(raw,sentinel,indices,rules,params):
 out=np.empty((len(raw),len(indices)),np.float64)
 for col,(index,param) in enumerate(zip(indices,params)):
  rule=rules['x_static',index]; prepared,valid,_=scaling._prepare_values(raw[:,index],rule); valid=np.asarray(valid)&~sentinel
  if rule['scaling']=='robust': out[:,col]=(np.where(valid,prepared,param['center'])-param['center'])/param['scale']
  else: out[:,col]=np.where(valid,prepared,float(rule.get('identity_fill',0.)))
 return out

def fit_imputer_standardizer(x):
 x=np.asarray(x,np.float64); med=np.zeros(x.shape[1]); missing=np.zeros(x.shape[1],bool)
 for j in range(x.shape[1]):
  finite=np.isfinite(x[:,j]); missing[j]=not finite.any(); med[j]=np.median(x[finite,j]) if finite.any() else 0.
 filled=np.where(np.isfinite(x),x,med); mean=filled.mean(0); scale=filled.std(0); scale=np.where(scale>0,scale,1.)
 return med,mean,scale,missing
def apply_imputer_standardizer(x,med,mean,scale): return (np.where(np.isfinite(x),x,med)-mean)/scale

def eligible_pairs(y,age):
 p=age[y==1]; n=age[y==0]; return int((np.abs(p[:,None]-n[None,:])<=2).sum())
def evaluate_predictions(y,score,age):
 return {'n':int(len(y)),'positive':int(np.sum(y==1)),'negative':int(np.sum(y==0)),'eligible_ac_pairs':eligible_pairs(y,age),'ac_auroc':float(compute_auroc_age(y,score,age,gap=2)),'age_weighted_auroc':float(compute_auroc_weighted(y,score,age,gap=2)),'auroc':float(roc_auc_score(y,score)),'auprc':float(average_precision_score(y,score))}

def prepare_fold(m,raw,rules,holdout,indices):
 train=(m.site.to_numpy()!=holdout); test=~train
 if set(m.loc[train,'site'])!=set(SITES)-{holdout} or set(m.loc[test,'site'])!={holdout}: raise RuntimeError(f'LOSO site isolation failed: {holdout}')
 if set(m.loc[train,'patient_id'])&set(m.loc[test,'patient_id']): raise RuntimeError(f'Patient leakage: {holdout}')
 sentinel=m.caisr_block_missing.to_numpy(bool); params=fit_typed_static_transform(raw[train],sentinel[train],indices,rules); xt=apply_typed_static_transform(raw[train],sentinel[train],indices,rules,params); xv=apply_typed_static_transform(raw[test],sentinel[test],indices,rules,params); med,mean,scale,missing=fit_imputer_standardizer(xt)
 return train,test,apply_imputer_standardizer(xt,med,mean,scale),apply_imputer_standardizer(xv,med,mean,scale),params,missing

def fit_fold(m,raw,rules,names,groups,holdout,seed,indices=CAISR20_INDICES):
 train,test,xtr,xte,params,missing=prepare_fold(m,raw,rules,holdout,indices); y=m.label.to_numpy(int); age=m.age.to_numpy(float); config={**LR_CONFIG,'random_state':int(seed)}
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always',ConvergenceWarning); model=LogisticRegression(**config).fit(xtr,y[train])
 warns=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]; trscore=model.decision_function(xtr); score=model.decision_function(xte)
 subset_names=[names[CAISR20_INDICES.index(i)] for i in indices]; subset_groups=[groups[CAISR20_INDICES.index(i)] for i in indices]
 return {'holdout_site':holdout,'seed':seed,'train_mask':train,'test_mask':test,'train_score':trscore,'score':score,'train_metrics':evaluate_predictions(y[train],trscore,age[train]),'metrics':evaluate_predictions(y[test],score,age[test]),'coef':model.coef_[0].copy(),'feature_names':subset_names,'feature_groups':subset_groups,'n_iter':int(model.n_iter_[0]),'converged':not warns,'warnings':warns,'typed_parameters':params,'all_missing_columns':np.flatnonzero(missing).tolist()}

def run_loso(m,raw,rules,names,groups,seed,indices=CAISR20_INDICES): return [fit_fold(m,raw,rules,names,groups,s,seed,indices) for s in SITES]
def aggregate(results):
 vals=[r['metrics']['ac_auroc'] for r in results]
 return {'sites':{r['holdout_site']:r['metrics'] for r in results},'macro_ac_auroc':float(np.mean(vals)),'worst_site_ac_auroc':float(np.min(vals)),'macro_auroc':float(np.mean([r['metrics']['auroc'] for r in results])),'macro_auprc':float(np.mean([r['metrics']['auprc'] for r in results]))}

def save_primary(out,m,results,names,groups):
 pdir=out/'primary'; (pdir/'predictions').mkdir(parents=True); rows=[]; coefs=[]
 for r in results:
  test=r['test_mask']; frame=m.loc[test,['record_id','patient_id','site','label','age']].copy(); frame['decision_score']=r['score']; frame.rename(columns={'age':'raw_age'}).to_csv(pdir/'predictions'/f"holdout_{r['holdout_site']}.csv",index=False)
  rows.append({'seed':r['seed'],'holdout_site':r['holdout_site'],**r['metrics'],'train_ac_auroc':r['train_metrics']['ac_auroc'],'n_iter':r['n_iter'],'converged':r['converged'],'warnings':' | '.join(r['warnings'])})
  for name,group,value in zip(r['feature_names'],r['feature_groups'],r['coef']): coefs.append({'seed':r['seed'],'holdout_site':r['holdout_site'],'feature':name,'group':group,'coefficient':value})
 pd.DataFrame(rows).to_csv(pdir/'fold_metrics.csv',index=False); pd.DataFrame(coefs).to_csv(pdir/'coefficients.csv',index=False); agg=aggregate(results); (pdir/'aggregate_metrics.json').write_text(json.dumps(agg,indent=2)+'\n'); return agg

def historical_regression(out,m,results,agg):
 details={}; aligned=True; maxdiff=0.; corrs=[]
 for r in results:
  site=r['holdout_site']; metric_diff=abs(agg['sites'][site]['ac_auroc']-ORACLE[site]); hist=pd.read_csv(ROOT/'data2_caisr20_results/predictions'/f'holdout_{site}.csv'); new=pd.DataFrame({'record_id':m.loc[r['test_mask'],'record_id'].to_numpy(),'decision_score':r['score']}); merged=new.merge(hist[['record_id','decision_score']],on='record_id',suffixes=('_new','_historical'),validate='one_to_one'); ok=len(merged)==len(new)==len(hist); aligned&=ok; diff=float(np.max(np.abs(merged.decision_score_new-merged.decision_score_historical))); corr=float(np.corrcoef(merged.decision_score_new,merged.decision_score_historical)[0,1]); maxdiff=max(maxdiff,diff); corrs.append(corr); details[site]={'ac':agg['sites'][site]['ac_auroc'],'oracle_ac':ORACLE[site],'absolute_metric_difference':metric_diff,'records_aligned':ok,'n_records':len(merged),'max_abs_score_difference':diff,'score_correlation':corr}
 checks={'historical_commit':HISTORICAL_COMMIT,'oracle':ORACLE,'sites':details,'macro_absolute_difference':abs(agg['macro_ac_auroc']-ORACLE['macro']),'worst_absolute_difference':abs(agg['worst_site_ac_auroc']-ORACLE['worst']),'record_alignment':bool(aligned),'max_abs_score_difference':maxdiff,'minimum_score_correlation':min(corrs),'metric_tolerance':1e-12,'score_atol':1e-6}
 checks['pass']=bool(aligned and maxdiff<=1e-6 and checks['macro_absolute_difference']<=1e-12 and checks['worst_absolute_difference']<=1e-12 and all(v['absolute_metric_difference']<=1e-12 for v in details.values()))
 (out/'regression_check.json').write_text(json.dumps(checks,indent=2)+'\n'); return checks

def run_seed_stability(out,m,raw,rules,names,groups,canonical):
 all_results=[]
 for seed in STABILITY_SEEDS:
  current=canonical if seed==7 else run_loso(m,raw,rules,names,groups,seed)
  all_results.extend(current); print(f'STABILITY seed={seed} complete',flush=True)
 rows=[]; coefs=[]
 for r in all_results:
  rows.append({'seed':r['seed'],'holdout_site':r['holdout_site'],'AC':r['metrics']['ac_auroc'],'AUROC':r['metrics']['auroc'],'AUPRC':r['metrics']['auprc'],'n_iter':r['n_iter'],'converged':r['converged'],'warnings':' | '.join(r['warnings'])})
  for name,group,value in zip(r['feature_names'],r['feature_groups'],r['coef']): coefs.append({'seed':r['seed'],'holdout_site':r['holdout_site'],'feature':name,'group':group,'coefficient':value})
 sf=pd.DataFrame(rows); sf.to_csv(out/'stability/seed_metrics.csv',index=False)
 summary=[]
 for site in SITES:
  v=sf.loc[sf.holdout_site==site,'AC']; summary.append({'scope':site,'mean_AC':v.mean(),'std_AC':v.std(ddof=1),'min_AC':v.min(),'max_AC':v.max()})
 byseed=sf.pivot(index='seed',columns='holdout_site',values='AC').loc[list(STABILITY_SEEDS),list(SITES)]; macro=byseed.mean(1); worst=byseed.min(1)
 for label,v in [('Macro',macro),('Worst',worst)]: summary.append({'scope':label,'mean_AC':v.mean(),'std_AC':v.std(ddof=1),'min_AC':v.min(),'max_AC':v.max()})
 pd.DataFrame(summary).to_csv(out/'stability/seed_summary.csv',index=False)
 cf=pd.DataFrame(coefs); cf.to_csv(out/'interpretability/coefficients.csv',index=False)
 cs=[]
 for (name,group),g in cf.groupby(['feature','group'],sort=False):
  v=g.coefficient.to_numpy(); pos=np.mean(v>0); neg=np.mean(v<0); zero=np.mean(v==0); cs.append({'feature':name,'group':group,'median_coefficient':np.median(v),'mean_coefficient':np.mean(v),'std_coefficient':np.std(v,ddof=1),'median_abs_coefficient':np.median(np.abs(v)),'sign_consistency':max(pos,neg),'positive_fraction':pos,'negative_fraction':neg,'zero_fraction':zero,'n_coefficients':len(v)})
 csum=pd.DataFrame(cs).sort_values(['sign_consistency','median_abs_coefficient'],ascending=[False,False]); csum.to_csv(out/'interpretability/coefficient_summary.csv',index=False)
 return sf,pd.DataFrame(summary),csum

def vectorized_ac(y,score,age):
 y=np.asarray(y); score=np.asarray(score); age=np.asarray(age); ps=score[y==1]; ns=score[y==0]; pa=age[y==1]; na=age[y==0]; eligible=np.abs(pa[:,None]-na[None,:])<=2; wins=(ps[:,None]>ns[None,:]).astype(np.float64)+.5*(ps[:,None]==ns[None,:]); denom=eligible.sum()
 if not denom: return math.nan
 return float((wins*eligible).sum()/denom)

def bootstrap_confidence_intervals(out,m,results,reps,seed):
 rng=np.random.default_rng(seed); boot={s:np.full(reps,np.nan) for s in SITES}
 for r in results:
  site=r['holdout_site']; y=m.loc[r['test_mask'],'label'].to_numpy(int); age=m.loc[r['test_mask'],'age'].to_numpy(float); score=r['score']; official=float(compute_auroc_age(y,score,age,gap=2)); fast=vectorized_ac(y,score,age)
  if abs(official-fast)>1e-12: raise RuntimeError(f'Vectorized AC mismatch {site}: {official} {fast}')
  ps=score[y==1]; ns=score[y==0]; pa=age[y==1]; na=age[y==0]; valid=(np.abs(pa[:,None]-na[None,:])<=2).astype(np.float64); wins=((ps[:,None]>ns[None,:])+.5*(ps[:,None]==ns[None,:]))*valid; pc=rng.multinomial(len(ps),np.full(len(ps),1/len(ps)),size=reps); nc=rng.multinomial(len(ns),np.full(len(ns),1/len(ns)),size=reps)
  for start in range(0,reps,50):
   stop=min(start+50,reps); denom=np.einsum('bi,ij,bj->b',pc[start:stop],valid,nc[start:stop],optimize=True); numer=np.einsum('bi,ij,bj->b',pc[start:stop],wins,nc[start:stop],optimize=True); boot[site][start:stop]=np.divide(numer,denom,out=np.full(stop-start,np.nan),where=denom>0)
 matrix=np.column_stack([boot[s] for s in SITES]); boot['Macro']=np.nanmean(matrix,axis=1); boot['Worst']=np.nanmin(matrix,axis=1); point={s:next(r['metrics']['ac_auroc'] for r in results if r['holdout_site']==s) for s in SITES}; point['Macro']=np.mean(list(point.values())); point['Worst']=np.min([point[s] for s in SITES]); rows=[]
 for scope in (*SITES,'Macro','Worst'):
  v=boot[scope]; valid=v[np.isfinite(v)]; rows.append({'scope':scope,'point_estimate':point[scope],'bootstrap_mean':np.mean(valid),'bootstrap_std':np.std(valid,ddof=1),'ci_2.5':np.percentile(valid,2.5),'ci_97.5':np.percentile(valid,97.5),'valid_replicates':len(valid),'requested_replicates':reps,'bootstrap_seed':seed})
 frame=pd.DataFrame(rows); frame.to_csv(out/'uncertainty/bootstrap_ci.csv',index=False); return frame

def run_group_ablation(out,m,raw,rules,names,groups,full_agg):
 rows=[]
 for removed in FEATURE_GROUPS:
  keep=tuple(i for i in CAISR20_INDICES if i not in FEATURE_GROUPS[removed]); results=run_loso(m,raw,rules,names,groups,7,keep); agg=aggregate(results); row={'model':f'CAISR20_minus_{removed}','removed_group':removed,'remaining_dim':len(keep),**{f'{s}_AC':agg['sites'][s]['ac_auroc'] for s in SITES},'Macro_AC':agg['macro_ac_auroc'],'Worst_AC':agg['worst_site_ac_auroc']}
  for s in SITES: row[f'delta_{s}_vs_full']=row[f'{s}_AC']-full_agg['sites'][s]['ac_auroc']
  row['delta_Macro_vs_full']=row['Macro_AC']-full_agg['macro_ac_auroc']; row['delta_Worst_vs_full']=row['Worst_AC']-full_agg['worst_site_ac_auroc']; rows.append(row); print(f'ABLATION {removed} complete',flush=True)
 frame=pd.DataFrame(rows); frame.to_csv(out/'interpretability/group_ablation.csv',index=False); return frame

def summarize_feature_distributions(out,m,raw,rules,names,groups):
 rows=[]; sentinel=m.caisr_block_missing.to_numpy(bool)
 for site in SITES:
  take=m.site.to_numpy()==site
  for pos,(index,name,group) in enumerate(zip(CAISR20_INDICES,names,groups)):
   values=raw[take,index]; prepared,valid,_=scaling._prepare_values(values,rules['x_static',index]); valid=np.asarray(valid)&~sentinel[take]; observed=values[valid]; q=np.percentile(observed,[25,50,75]) if len(observed) else [np.nan]*3
   rows.append({'site':site,'position':pos,'x_static_index':index,'feature_name':name,'feature_group':group,'valid_n':int(valid.sum()),'missing_n':int((~valid).sum()),'missing_rate':float((~valid).mean()),'median':q[1],'q25':q[0],'q75':q[2],'IQR':q[2]-q[0]})
 frame=pd.DataFrame(rows); frame.to_csv(out/'interpretability/feature_distribution_by_site.csv',index=False); return frame

def write_feature_manifest(out,names,groups):
 pd.DataFrame({'position':np.arange(1,21),'x_static_index':CAISR20_INDICES,'feature_name':names,'feature_group':groups,'source':'CAISR_algorithmic_annotation'}).to_csv(out/'feature_manifest.csv',index=False)
def write_config(out,args,m,names,groups):
 payload={'dataset':'DATA2','data_root':str(Path(args.data_root).resolve()),'n_records':len(m),'extraction_version':'timegrid_v2.0.0','feature_set':'CAISR20','feature_indices':list(CAISR20_INDICES),'feature_names':list(names),'feature_groups':{k:list(v) for k,v in FEATURE_GROUPS.items()},'preprocessing':'outer-training-fold typed_v1 low-level rules; CAISR all-zero sentinel invalid; typed center/identity fill; training median imputation; training population mean/std standardization','LR_config':LR_CONFIG,'canonical_seed':7,'stability_seeds':list(STABILITY_SEEDS),'bootstrap_seed':args.bootstrap_seed,'bootstrap_replicates':args.bootstrap_replicates,'primary_metric':'Age-conditioned AUROC','gap':2,'python_version':sys.version,'numpy_version':np.__version__,'sklearn_version':sklearn.__version__,'platform':platform.platform(),'git_base_commit':git_head(),'baseline_py_sha256':sha256(ROOT/'baseline.py'),'feature_rules_v1_sha256':sha256(args.feature_rules),'uses_human_annotations':False,'uses_demographics_as_predictors':False,'age_used_for_metric_only':True,'creates_feature_cache':False,'historical_regression_reference_commit':HISTORICAL_COMMIT,'primary_only':args.primary_only}
 (out/'config.json').write_text(json.dumps(payload,indent=2)+'\n')
def table(headers,rows): return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def write_report(out,m,names,agg,regression,stability=None,bootstrap=None,coef=None,ablation=None,distribution=None):
 rows=[]
 for site in SITES:
  x=agg['sites'][site]; rows.append([site,x['n'],x['positive'],x['negative'],f"{x['ac_auroc']:.6f}",f"{x['auroc']:.6f}",f"{x['auprc']:.6f}"])
 missing=m.groupby('site').caisr_block_missing.agg(['sum','count']); missrows=[[s,int(missing.loc[s,'sum']),int(missing.loc[s,'count']),f"{missing.loc[s,'sum']/missing.loc[s,'count']:.3%}"] for s in SITES]
 lines=['# Standalone DATA2 CAISR20 baseline','','## 1. Objective','','Freeze a reproducible low-capacity CAISR20-only elastic-net logistic-regression baseline for three-site DATA2 LOSO. The primary metric is official age-conditioned AUROC at gap 2.','','## 2. Dataset and LOSO protocol','',f'Directly scanned 6600 `timegrid_v2.0.0` NPZ files from `{Path(m.iloc[0].npz_path).parents[1]}`. Each held-out site was invisible to transform, imputation, standardization, and model fitting. Age was used only for evaluation.','','CAISR block missingness (finite all-zero algorithmic186 sentinel):','',table(['Site','Missing','N','Rate'],missrows),'','## 3. CAISR20 feature definition','']
 for i,(name,index,group) in enumerate(zip(names,CAISR20_INDICES,[feature_group(i) for i in CAISR20_INDICES]),1): lines.append(f'{i}. `{name}` — x_static[{index}], {group}')
 lines += ['','All inputs are automated CAISR algorithmic annotation summaries; demographics are excluded.','', '## 4. Preprocessing','','For each outer fold and column, typed_v1 rules were applied and robust center/scale parameters were fitted only on the two training sites. The all-zero algorithmic186 sentinel was invalid. P3-equivalent median imputation and population mean/std standardization were then fitted only on training data. No cache or transformed NPZ was created.','','## 5. Classifier','',f'Frozen elastic-net logistic regression: `{json.dumps({**LR_CONFIG,"random_state":7})}`. No tuning, calibration, ranking loss, or threshold selection.','','## 6. Primary results','',table(['Site','N','Positive','Negative','AC-AUROC','AUROC','AUPRC'],rows),'',f"- Macro AC: **{agg['macro_ac_auroc']:.6f}**",f"- Worst-site AC: **{agg['worst_site_ac_auroc']:.6f}**",'','## 7. Historical regression validation','',f"Regression against commit `{HISTORICAL_COMMIT}`: **{'PASS' if regression['pass'] else 'FAIL'}**. Maximum aligned score difference: {regression['max_abs_score_difference']:.3g}; all metric differences were required to be <=1e-12."]
 if stability is not None:
  lines += ['','## 8. Solver seed stability','',table(['Scope','Mean AC','SD','Min','Max'],[[r.scope,f'{r.mean_AC:.6f}',f'{r.std_AC:.6g}',f'{r.min_AC:.6f}',f'{r.max_AC:.6f}'] for r in stability.itertuples(index=False)]),'','The canonical paper point estimate remains seed 7; the five-seed mean is not substituted for it.']
 if bootstrap is not None:
  brows=[[r['scope'],f"{r['point_estimate']:.6f}",f"{r['bootstrap_mean']:.6f}",f"{r['bootstrap_std']:.6f}",f"[{r['ci_2.5']:.6f}, {r['ci_97.5']:.6f}]",r['valid_replicates']] for r in bootstrap.to_dict('records')]
  lines += ['','## 9. Bootstrap uncertainty','',table(['Scope','Point','Bootstrap mean','SD','95% CI','Valid'],brows),'','Patient-level bootstrap sampled positives and negatives separately within site (1000 replicates; seed 20260917).']
 if coef is not None and ablation is not None:
  top=coef.sort_values(['sign_consistency','median_abs_coefficient'],ascending=[False,False]).head(8)
  lines += ['','## 10. Interpretability','','Most sign-stable standardized coefficients:','',table(['Feature','Group','Median coef','Median abs(coef)','Sign consistency','Zero fraction'],[[r.feature,r.group,f'{r.median_coefficient:.4f}',f'{r.median_abs_coefficient:.4f}',f'{r.sign_consistency:.3f}',f'{r.zero_fraction:.3f}'] for r in top.itertuples(index=False)]),'','Leave-one-group-out results (delta is ablation minus full CAISR20):','',table(['Removed group','Remaining dim','Macro AC','Worst AC','Delta Macro','Delta Worst'],[[r.removed_group,r.remaining_dim,f'{r.Macro_AC:.6f}',f'{r.Worst_AC:.6f}',f'{r.delta_Macro_vs_full:+.6f}',f'{r.delta_Worst_vs_full:+.6f}'] for r in ablation.itertuples(index=False)]),'','Coefficients describe associations in fold-standardized linear models, not causal importance. Correlated features can share or substitute coefficients. Site distributions and missingness are reported descriptively and were not used for feature selection.']
 lines += ['','## 11. Limitations','','CAISR20 is derived from automated CAISR annotations, not human expert annotations. `mean_stage_entropy` uses CAISR stage-posterior uncertainty. Human annotations exist in the broader training data, but this baseline does not use them. A matched human-vs-CAISR annotation-source comparison remains a separate future experiment and no result is implied here.','','No LSTM, temporal model, demographics fusion, ranking objective, human-annotation baseline, SHAP, feature search, or extractor change was performed.']
 (out/'BASELINE_RESULTS.md').write_text('\n'.join(lines)+'\n')

def parse_args():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--data-root',type=Path,default=ROOT/'npz_data2'); p.add_argument('--output-dir',type=Path,default=ROOT/'baseline_results'); p.add_argument('--feature-rules',type=Path,default=ROOT/'feat_input/feature_rules_v1.json'); p.add_argument('--primary-only',action='store_true'); p.add_argument('--bootstrap-replicates',type=int,default=1000); p.add_argument('--bootstrap-seed',type=int,default=20260917); p.add_argument('--overwrite',action='store_true'); return p.parse_args()
def main():
 args=parse_args(); out=args.output_dir.resolve()
 if out.exists():
  if not args.overwrite: raise FileExistsError(f'Refusing to overwrite completed/partial output: {out}')
  shutil.rmtree(out)
 for path in (out,out/'primary',out/'stability',out/'uncertainty',out/'interpretability'): path.mkdir(parents=True,exist_ok=True)
 _,rules,names=load_rules(args.feature_rules); groups=tuple(feature_group(i) for i in CAISR20_INDICES); m,raw=load_dataset(args.data_root); write_feature_manifest(out,names,groups); write_config(out,args,m,names,groups)
 canonical=run_loso(m,raw,rules,names,groups,7)
 if not all(r['converged'] for r in canonical): raise RuntimeError(f'Canonical convergence failure: {[r["warnings"] for r in canonical]}')
 agg=save_primary(out,m,canonical,names,groups); regression=historical_regression(out,m,canonical,agg); write_report(out,m,names,agg,regression)
 if not regression['pass']: raise RuntimeError(f'Historical regression failed: {json.dumps(regression)}')
 if args.primary_only: print('PRIMARY_ONLY_COMPLETE',json.dumps(agg),flush=True); return
 seed_metrics,seed_summary,coef=run_seed_stability(out,m,raw,rules,names,groups,canonical)
 if not bool(seed_metrics.converged.all()): raise RuntimeError('At least one stability fit did not converge')
 bootstrap=bootstrap_confidence_intervals(out,m,canonical,args.bootstrap_replicates,args.bootstrap_seed); ablation=run_group_ablation(out,m,raw,rules,names,groups,agg); distribution=summarize_feature_distributions(out,m,raw,rules,names,groups); write_report(out,m,names,agg,regression,seed_summary,bootstrap,coef,ablation,distribution)
 required=['BASELINE_RESULTS.md','config.json','feature_manifest.csv','regression_check.json','primary/fold_metrics.csv','primary/aggregate_metrics.json','primary/coefficients.csv','stability/seed_metrics.csv','stability/seed_summary.csv','uncertainty/bootstrap_ci.csv','interpretability/coefficients.csv','interpretability/coefficient_summary.csv','interpretability/group_ablation.csv','interpretability/feature_distribution_by_site.csv']
 missing=[p for p in required if not (out/p).is_file()]
 if missing: raise RuntimeError(f'Missing outputs: {missing}')
 print('BASELINE_COMPLETE',json.dumps(agg),flush=True)
if __name__=='__main__': main()
