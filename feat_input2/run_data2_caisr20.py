#!/usr/bin/env python3
"""Strict DATA2 CAISR20-only LOSO using the frozen compact30 pipeline."""
from __future__ import annotations
import json, os, subprocess, sys, warnings
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ.setdefault(k,'1')
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import feature_scaling as fs
import pooled_logistic_v2 as pooled
from feat_input.feat_mody import run_p3_v2 as p3
from feat_input.feat_mody import run_loso_1103 as p5
MANIFEST=ROOT/'feat_input2/output/data2_lowcost/data2_manifest.csv'
CACHE=ROOT/'feat_input2/output/data2_lowcost/cache/data2_patient_summaries.npz'
RULES=ROOT/'feat_input/feature_rules_v1.json'; OUT=ROOT/'data2_caisr20_results'; SITES=('I0002','I0006','S0001')
BASE={'demo10':{'I0002':.6052492046659597,'I0006':.4899606706685986,'S0001':.5769456480008776,'macro':.5573851744451453,'worst':.4899606706685986},'compact30':{'I0002':.7476139978791092,'I0006':.6805354308976748,'S0001':.6538199967092634,'macro':.6939898084953491,'worst':.6538199967092634}}

def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def prepare():
 if OUT.exists(): raise FileExistsError(f'Refusing to overwrite: {OUT}')
 m=pd.read_csv(MANIFEST,dtype={'record_id':str,'patient_id':str,'site':str})
 exp={'I0002':(319,52,267),'I0006':(1142,112,1030),'S0001':(5139,334,4805)}
 if len(m)!=6600 or m.record_id.nunique()!=6600 or m.patient_id.nunique()!=6600 or int(m.label.sum())!=498: raise RuntimeError('DATA2 identity/count mismatch')
 for s,(n,p,neg) in exp.items():
  g=m[m.site==s]
  if (len(g),int(g.label.sum()),int((g.label==0).sum()))!=(n,p,neg): raise RuntimeError(f'count mismatch {s}')
 with np.load(CACHE,allow_pickle=False) as z: ids=z['record_id'].astype(str)
 if not np.array_equal(ids,m.record_id.to_numpy(str)): raise RuntimeError('cache/manifest order mismatch')
 rules=fs.load_feature_rules(RULES); holder=type('RuleHolder',(),{'rules':rules})(); demo=p3.feature_names(holder,'P3_demo10'); compact=p3.feature_names(holder,'P3_compact30'); caisr=compact[len(demo):]; indices=list(map(int,pooled.COMPACT_INDICES))
 forbidden=('age','sex','race','bmi')
 assert len(demo)==10 and len(caisr)==20 and len(indices)==20
 assert set(compact)==set(demo)|set(caisr) and set(demo).isdisjoint(caisr)
 assert not any(name.lower() in ('age','bmi') or name.lower().startswith('sex_') or name.lower().startswith('race_') for name in caisr)
 OUT.mkdir(); (OUT/'predictions').mkdir(); (OUT/'folds').mkdir()
 pd.DataFrame({'caisr_position':range(20),'compact30_position':range(10,30),'x_static_index':indices,'feature_name':caisr}).to_csv(OUT/'feature_manifest.csv',index=False)
 (OUT/'feature_manifest.json').write_text(json.dumps({'dimension':20,'x_static_indices':indices,'feature_names':caisr,'demo10_names':demo,'compact30_names':compact,'assertions':{'dimension_20':True,'no_demographics':True,'compact_equals_disjoint_union':True}},indent=2)+'\n')
 return m,caisr,indices

def main():
 m,names,indices=prepare(); y=m.label.to_numpy(int); ages=m.age.to_numpy(float); sites=m.site.to_numpy(str); rows=[]
 with np.load(CACHE,allow_pickle=False) as z:
  matrices={s:np.asarray(z[f'p3_{s}_compact30'][:,10:30],float) for s in SITES}
 for hold in SITES:
  tr=sites!=hold; te=~tr
  if set(sites[tr])!={s for s in SITES if s!=hold} or set(sites[te])!={hold}: raise RuntimeError(f'LOSO leakage {hold}')
  xtr,xte=matrices[hold][tr],matrices[hold][te]
  if xtr.shape[1]!=20 or xte.shape[1]!=20: raise RuntimeError('dimension mismatch')
  med,center,scale,missing=p3.fit_imputer_scaler(xtr); xtr=p3.apply_imputer_scaler(xtr,med,center,scale); xte=p3.apply_imputer_scaler(xte,med,center,scale)
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always',ConvergenceWarning); model=LogisticRegression(**p3.LR).fit(xtr,y[tr])
  warn=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]; train_score=model.decision_function(xtr); test_score=model.decision_function(xte); tm=p5.metric_bundle(y[tr],train_score,ages[tr]); hm=p5.metric_bundle(y[te],test_score,ages[te])
  pred=m.loc[te,['record_id','patient_id','site','label','age']].copy(); pred['decision_score']=test_score; pred.rename(columns={'age':'raw_age'}).to_csv(OUT/'predictions'/f'holdout_{hold}.csv',index=False)
  config={'holdout_site':hold,'feature_set':'CAISR20 = frozen compact30 columns 10:30','feature_names':names,'x_static_indices':indices,'model_config':p3.LR,'preprocessing':'same fold-specific typed selected-column cache and P3 fold-local imputer/scaler as DATA2 compact30','preprocessing_fit_scope':'outer training sites only','seed':7,'train_sites':sorted(set(sites[tr])),'holdout_used_for_fit':False,'all_missing_columns':np.flatnonzero(missing).tolist()}
  (OUT/'folds'/f'{hold}_config.json').write_text(json.dumps(config,indent=2)+'\n')
  row={'holdout_site':hold,'train_n':int(tr.sum()),'train_positive':int(y[tr].sum()),'holdout_n':int(te.sum()),'holdout_positive':int(y[te].sum()),'holdout_negative':int((y[te]==0).sum()),**{f'train_{k}':v for k,v in tm.items()},**{f'holdout_{k}':v for k,v in hm.items()},'n_iter':int(model.n_iter_[0]),'converged':not warn,'warnings':' | '.join(warn)}; rows.append(row); (OUT/'folds'/f'{hold}_metrics.json').write_text(json.dumps(row,indent=2)+'\n'); print(json.dumps(row),flush=True)
 fold=pd.DataFrame(rows); fold.to_csv(OUT/'fold_metrics.csv',index=False); g=fold.set_index('holdout_site').loc[list(SITES)]
 vals={s:float(g.loc[s,'holdout_ac_auroc']) for s in SITES}; agg={'model':'DATA2-CAISR20','sites':{s:{k:(int(g.loc[s,f'holdout_{k}']) if k in ('n','positives','negatives','eligible_ac_pairs') else float(g.loc[s,f'holdout_{k}'])) for k in ('ac_auroc','age_weighted_auroc','auroc','auprc','n','positives','negatives','eligible_ac_pairs')} for s in SITES},'macro_ac_auroc':float(np.mean(list(vals.values()))),'worst_site_ac_auroc':float(min(vals.values())),'macro_auroc':float(g.holdout_auroc.mean()),'macro_auprc':float(g.holdout_auprc.mean()),'feature_dimension':20,'seed':7,'source_commit':git('rev-parse','HEAD')}
 (OUT/'aggregate_summary.json').write_text(json.dumps(agg,indent=2)+'\n')
 d_demo={s:vals[s]-BASE['demo10'][s] for s in SITES}; d_comp={s:BASE['compact30'][s]-vals[s] for s in SITES}; dm=agg['macro_ac_auroc']-BASE['demo10']['macro']; dw=agg['worst_site_ac_auroc']-BASE['demo10']['worst']; cm=BASE['compact30']['macro']-agg['macro_ac_auroc']; cw=BASE['compact30']['worst']-agg['worst_site_ac_auroc']
 near=cm<=.01 and cw<=.01; case='Case A: CAISR20 is within 0.01 of compact30 on Macro and Worst.' if near else ('Case B: demographics10 and CAISR20 show meaningful complementarity across sites.' if sum(d_comp[s]>0 for s in SITES)>=2 else 'Case C/mixed: the compact30 benefit is site-specific.')
 lines=['# DATA2 CAISR20-only LOSO baseline','',f'- Git base: `{git("rev-parse","HEAD")}`',f'- DATA2: `{ROOT/"npz_data2"}` (6600 unique subjects; 498 positive / 6102 negative)',f'- Model config: `{json.dumps(p3.LR,default=str)}`','- Preprocessing: identical frozen DATA2 compact30 fold-specific typed selected-column transform, then P3 median imputation/linear scaling fitted on outer training sites only.','- Official primary metric: Age-conditioned AUROC, gap=2, via the existing P5 wrapper.','', '## Exact CAISR20 manifest','']+[f'{i+1}. `{n}` (x_static index {j})' for i,(n,j) in enumerate(zip(names,indices))]+['','## Results','','| Model | I0002 | I0006 | S0001 | Macro | Worst |','|---|---:|---:|---:|---:|---:|',f"| demo10 | {BASE['demo10']['I0002']:.3f} | {BASE['demo10']['I0006']:.3f} | {BASE['demo10']['S0001']:.3f} | {BASE['demo10']['macro']:.3f} | {BASE['demo10']['worst']:.3f} |",f"| CAISR20 | {vals['I0002']:.3f} | {vals['I0006']:.3f} | {vals['S0001']:.3f} | {agg['macro_ac_auroc']:.3f} | {agg['worst_site_ac_auroc']:.3f} |",f"| compact30 | {BASE['compact30']['I0002']:.3f} | {BASE['compact30']['I0006']:.3f} | {BASE['compact30']['S0001']:.3f} | {BASE['compact30']['macro']:.3f} | {BASE['compact30']['worst']:.3f} |",'','## Deltas','',f"- CAISR20 - demo10: I0002 {d_demo['I0002']:+.3f}, I0006 {d_demo['I0006']:+.3f}, S0001 {d_demo['S0001']:+.3f}, Macro {dm:+.3f}, Worst {dw:+.3f}.",f"- compact30 - CAISR20: I0002 {d_comp['I0002']:+.3f}, I0006 {d_comp['I0006']:+.3f}, S0001 {d_comp['S0001']:+.3f}, Macro {cm:+.3f}, Worst {cw:+.3f}.",'',f'## Pre-specified interpretation','',case,'','No feature-definition, patient/site leakage, or preprocessing-fit anomaly was detected. The concurrent R1 process and its files were not stopped, changed, staged, or committed.']
 (OUT/'DATA2_CAISR20_RESULTS.md').write_text('\n'.join(lines)+'\n'); print('CAISR20_COMPLETE',json.dumps(agg),flush=True)
if __name__=='__main__': main()
