#!/usr/bin/env python
"""Three-site LOSO temporal-order study using direct DATA2 timegrid_v2 NPZ inputs."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, subprocess, time, warnings
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from evaluate_model import compute_auroc_age, compute_auroc_weighted
import feature_scaling as scaling
from temporal.data import (SITES, EEG_INDICES, STAGE_FEATURE_NAMES, TemporalRecord,
    apply_block_order, blockify, fit_neutral, impute_values, load_dataset, _selected)
from temporal.models import LocalTCNClassifier

ROOT=Path(__file__).resolve().parents[1]
DATA_ROOT=ROOT/'npz_data2'; RULES=ROOT/'feat_input'/'feature_rules_v1.json'
LR_CONFIG=dict(penalty='elasticnet',solver='saga',l1_ratio=.4,C=.03,class_weight='balanced',fit_intercept=True,max_iter=100000,tol=1e-4,random_state=7)
MODALITY_DIM={'stage_event13':13,'eeg72':72}

def git_head(): return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def json_dump(path,obj): Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def runtime_snapshot(device):
    """Process-local device/CPU/RAM snapshot from the actual remote worker."""
    memory={}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            key,value=line.split(':',1); memory[key]=int(value.strip().split()[0])
    except (OSError,ValueError): pass
    try:
        status={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in Path('/proc/self/status').read_text().splitlines() if ':' in line}
        rss_kib=int(status.get('VmRSS','0 kB').split()[0])
    except (OSError,ValueError): rss_kib=None
    out={'device':str(device),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'cpu_count':os.cpu_count(),
         'process_rss_kib':rss_kib,'system_mem_total_kib':memory.get('MemTotal'),'system_mem_available_kib':memory.get('MemAvailable')}
    if device.type=='cuda':
        out.update({'cuda_device_name':torch.cuda.get_device_name(device),'cuda_allocated_bytes':torch.cuda.memory_allocated(device),
                    'cuda_reserved_bytes':torch.cuda.memory_reserved(device),'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(device)})
    return out
def seed_all(seed=7):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False

def vectorized_ac(y,score,age,gap=2):
    y=np.asarray(y); score=np.asarray(score); age=np.asarray(age); p=np.flatnonzero(y==1); n=np.flatnonzero(y==0)
    eligible=np.abs(age[p,None]-age[n][None,:])<=gap
    if not eligible.any(): return math.nan
    diff=score[p,None]-score[n][None,:]
    return float(((diff>0)+.5*(diff==0))[eligible].mean())

def metric_bundle(y,score,age,include_weighted=True):
    y=np.asarray(y,int); score=np.asarray(score,float); age=np.asarray(age,float)
    p=np.flatnonzero(y==1); n=np.flatnonzero(y==0); pairs=int((np.abs(age[p,None]-age[n][None,:])<=2).sum())
    ac=float(compute_auroc_age(y,score,age,gap=2))
    if abs(ac-vectorized_ac(y,score,age,2))>1e-12: raise RuntimeError('Vectorized AC disagrees with official metric')
    return {'ac_auroc':ac,'age_weighted_auroc':float(compute_auroc_weighted(y,score,age,gap=2)) if include_weighted else None,
            'auroc':float(roc_auc_score(y,score)),'auprc':float(average_precision_score(y,score)),
            'n':len(y),'positives':int(y.sum()),'negatives':int((y==0).sum()),'eligible_ac_pairs':pairs}

def inner_split(records, outer_train):
    train=[]; val=[]
    for site in sorted({records[i].site for i in outer_train}):
        for label in (0,1):
            cell=sorted([i for i in outer_train if records[i].site==site and records[i].label==label],key=lambda i:records[i].record_id)
            if len(cell)<2: raise RuntimeError(f'Cannot split {site}/{label}: n={len(cell)}')
            tr,va=train_test_split(cell,test_size=.2,random_state=7,shuffle=True)
            if not tr or not va: raise RuntimeError(f'Empty inner split {site}/{label}')
            train.extend(tr); val.extend(va)
    train=sorted(train,key=lambda i:(records[i].site,records[i].record_id)); val=sorted(val,key=lambda i:(records[i].site,records[i].record_id))
    if set(train)&set(val): raise RuntimeError('Inner leakage')
    return train,val

def fit_eeg_transform(records, indices):
    rules_all=scaling.load_feature_rules(RULES); rules={int(r['index']):r for r in rules_all if r['branch']=='X_seq'}
    params=[]
    for local,actual in enumerate(EEG_INDICES):
        xs=[]; ws=[]
        for i in indices:
            r=records[i]; column=r.values[:,local]; valid=r.valid[:,local]
            selected=scaling._stable_indices(r.record_id,f'X_seq:{actual}',len(column),128,20260907)
            prepared,rule_valid,_=scaling._prepare_values(column[selected],rules[actual]); ok=valid[selected]&rule_valid
            if ok.any():
                use=prepared[ok]; xs.append(use); ws.append(np.full(len(use),1/len(use)))
        if not xs: raise RuntimeError(f'No valid EEG training values: {actual}')
        x=np.concatenate(xs); w=np.concatenate(ws); rule=rules[actual]
        if rule['scaling']=='identity': center,scale,method=0.,1.,'identity'
        else:
            q=[scaling._weighted_quantile(x,w,v) for v in (.05,.25,.5,.75,.95)]; center=q[2]
            candidates=[('iqr',q[3]-q[1]),('p95_p05',q[4]-q[0]),('std',float(np.sqrt(np.average((x-np.average(x,weights=w))**2,weights=w))))]
            tol=1e-6*max(1.,abs(center)); method,scale='unit_fallback',1.
            for name,value in candidates:
                if np.isfinite(value) and value>tol: method,scale=name,float(value); break
        params.append({'actual_index':actual,'rule':rule,'center':float(center),'scale':float(scale),'method':method})
    # Training-only neutral is the mean of valid transformed observations (all epochs, no cap).
    sums=np.zeros(72); counts=np.zeros(72,dtype=np.int64)
    for i in indices:
        r=records[i]
        for j,p in enumerate(params):
            prepared,rv,_=scaling._prepare_values(r.values[:,j],p['rule']); ok=r.valid[:,j]&rv
            if ok.any(): sums[j]+=((prepared[ok]-p['center'])/p['scale']).sum(); counts[j]+=int(ok.sum())
    if np.any(counts==0): raise RuntimeError('No valid EEG values for neutral imputation')
    return params,sums/counts

def apply_eeg_transform(record, state):
    params,neutral=state; out=np.empty_like(record.values,dtype=np.float32)
    for j,p in enumerate(params):
        prepared,rv,_=scaling._prepare_values(record.values[:,j],p['rule']); ok=record.valid[:,j]&rv
        transformed=(prepared-p['center'])/p['scale']; out[:,j]=np.where(ok,transformed,neutral[j]).astype(np.float32)
    if not np.isfinite(out).all(): raise RuntimeError(f'Nonfinite transformed EEG: {record.record_id}')
    return out

def fit_preprocessing(records, indices, modality):
    if modality=='stage_event13': return {'kind':'neutral','neutral':fit_neutral(records,indices)}
    params,neutral=fit_eeg_transform(records,indices)
    return {'kind':'typed_eeg72','state':(params,neutral)}

def apply_preprocessing(record,state):
    return impute_values(record,state['neutral']) if state['kind']=='neutral' else apply_eeg_transform(record,state['state'])

def preprocessing_json(state):
    if state['kind']=='neutral': return {'kind':'training_valid_neutral_mean','neutral':state['neutral'].tolist()}
    params,neutral=state['state']; return {'kind':'selected_column_typed_v1_training_only','sample_cap_per_record':128,'sampling_seed':20260907,
        'columns':[{'actual_index':p['actual_index'],'name':p['rule']['name'],'center':p['center'],'scale':p['scale'],'method':p['method']} for p in params], 'neutral':neutral.tolist()}

class BlockDataset(Dataset):
    def __init__(self,records,indices,state,shuffle):
        started=time.perf_counter(); self.total_blocks=0
        self.items=[]
        for i in indices:
            r=records[i]; blocks,_=blockify(apply_preprocessing(r,state)); blocks=apply_block_order(blocks,r.record_id,shuffle)
            self.total_blocks+=len(blocks)
            self.items.append((torch.from_numpy(np.ascontiguousarray(blocks)),float(r.label),i))
        self.build_wall_time_sec=time.perf_counter()-started
    def __len__(self): return len(self.items)
    def __getitem__(self,k): return self.items[k]

def collate(batch):
    maximum=max(x[0].shape[0] for x in batch); d=batch[0][0].shape[-1]; x=torch.zeros(len(batch),maximum,10,d); padding=torch.ones(len(batch),maximum,dtype=torch.bool)
    y=torch.tensor([v[1] for v in batch],dtype=torch.float32); idx=torch.tensor([v[2] for v in batch])
    for j,(blocks,_,_) in enumerate(batch): x[j,:len(blocks)]=blocks; padding[j,:len(blocks)]=False
    return x,padding,y,idx

def sampler_loader(records,indices,state,shuffle,batch_size=64):
    counts={};
    for i in indices: counts[(records[i].site,records[i].label)]=counts.get((records[i].site,records[i].label),0)+1
    cells=[]
    for site in sorted({records[i].site for i in indices}):
        for label in (0,1):
            count=counts.get((site,label),0)
            if count==0: raise RuntimeError(f'Empty sampler cell {site}/{label}')
            cells.append({'site':site,'label':label,'raw_count':count,'sample_weight':1/count,'expected_sampling_proportion':.25})
    weights=[1/counts[(records[i].site,records[i].label)] for i in indices]; generator=torch.Generator().manual_seed(7)
    sampler=WeightedRandomSampler(torch.tensor(weights,dtype=torch.double),len(indices),replacement=True,generator=generator)
    dataset=BlockDataset(records,indices,state,shuffle)
    loader=DataLoader(dataset,batch_size=batch_size,sampler=sampler,collate_fn=collate,num_workers=0,generator=generator)
    loader.block_build_wall_time_sec=dataset.build_wall_time_sec; loader.total_blocks=dataset.total_blocks
    return loader,cells

def natural_loader(records,indices,state,shuffle,batch_size=64):
    dataset=BlockDataset(records,indices,state,shuffle)
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,collate_fn=collate,num_workers=0)
    loader.block_build_wall_time_sec=dataset.build_wall_time_sec; loader.total_blocks=dataset.total_blocks
    return loader

def collect(model,loader,records,device):
    model.eval(); scores=[]; indices=[]
    with torch.no_grad():
        for x,pad,_,idx in loader:
            logits=model(x.to(device),pad.to(device));
            if not torch.isfinite(logits).all(): raise RuntimeError('Nonfinite logits')
            scores.extend(logits.cpu().numpy()); indices.extend(idx.numpy())
    y=np.array([records[i].label for i in indices]); age=np.array([records[i].age for i in indices])
    return np.asarray(indices),y,np.asarray(scores),age

def state_hash(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items(): h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def run_orderless_fold(records,outer_train,holdout,fold_dir,modality):
    state=fit_preprocessing(records,outer_train,modality)
    def summary(indices):
        rows=[]
        for i in indices:
            x=apply_preprocessing(records[i],state); rows.append(np.r_[x.mean(0),x.std(0)])
        return np.asarray(rows)
    x=summary(outer_train); xt=summary(holdout); y=np.array([records[i].label for i in outer_train]); yt=np.array([records[i].label for i in holdout]); ages=np.array([records[i].age for i in holdout])
    scaler=StandardScaler().fit(x); model=LogisticRegression(**LR_CONFIG)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning); model.fit(scaler.transform(x),y)
    convergence=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
    score=model.decision_function(scaler.transform(xt)); metrics=metric_bundle(yt,score,ages); metrics.update({'arm':'O','best_epoch':None,'convergence_warnings':convergence})
    save_predictions(fold_dir/'outer_predictions.csv',records,holdout,score)
    json_dump(fold_dir/'metrics.json',metrics); json_dump(fold_dir/'config.json',{'modality':modality,'arm':'O','indices':list(range(470,483)) if modality=='stage_event13' else list(EEG_INDICES),'preprocessing':preprocessing_json(state),'classifier':LR_CONFIG,'outer_holdout_never_used_for_selection':True})
    return metrics

def save_predictions(path,records,indices,scores):
    pd.DataFrame({'record_id':[records[i].record_id for i in indices],'patient_id':[records[i].patient_id for i in indices],'site':[records[i].site for i in indices],
        'label':[records[i].label for i in indices],'age':[records[i].age for i in indices],'raw_score':scores}).to_csv(path,index=False)

def run_tcn_fold(records,inner_train,inner_val,holdout,fold_dir,modality,shuffle,device,max_epochs=30,min_epochs=8,patience=5,batch_size=64,
                 holdout_site='unknown',save_checkpoint=True,profile_output=None):
    state=fit_preprocessing(records,inner_train,modality); seed_all(7); model=LocalTCNClassifier(MODALITY_DIM[modality]).to(device); initial_hash=state_hash(model)
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    train_loader,cells=sampler_loader(records,inner_train,state,shuffle,batch_size); pd.DataFrame(cells).to_csv(fold_dir/'sampler_cells.csv',index=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.,device=device))
    val_sites=sorted({records[i].site for i in inner_val})
    val_loaders={site:natural_loader(records,[i for i in inner_val if records[i].site==site],state,shuffle,batch_size) for site in val_sites}
    block_build={'inner_train_sec':train_loader.block_build_wall_time_sec,
                 **{f'inner_val_{site}_sec':loader.block_build_wall_time_sec for site,loader in val_loaders.items()}}
    history=[]; best=-math.inf; best_epoch=0; best_state=None; stale=0; started=time.perf_counter(); arm='S' if shuffle else 'R'
    profile={'holdout':holdout_site,'arm':arm,'device_start':runtime_snapshot(device),'block_construction':block_build,'epochs':[]}
    print(json.dumps({'event':'fold_start','holdout':holdout_site,'arm':arm,'device':runtime_snapshot(device),'block_construction':block_build}),flush=True)
    for epoch in range(1,max_epochs+1):
        epoch_start=time.perf_counter(); train_start=epoch_start; model.train(); total=0.; seen=0; batch_times=[]; total_batches=len(train_loader)
        for batch_no,(x,pad,y,_) in enumerate(train_loader,1):
            batch_start=time.perf_counter()
            x=x.to(device); pad=pad.to(device); y=y.to(device); optimizer.zero_grad(set_to_none=True); logits=model(x,pad); loss=criterion(logits,y)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); total+=float(loss)*len(y); seen+=len(y)
            if device.type=='cuda': torch.cuda.synchronize(device)
            batch_times.append(time.perf_counter()-batch_start)
            if batch_no==1 or batch_no==total_batches or batch_no%max(1,total_batches//10)==0:
                print(json.dumps({'event':'train_progress','holdout':holdout_site,'arm':arm,'epoch':epoch,
                    'train_batches':f'{batch_no}/{total_batches}','elapsed_total':time.perf_counter()-started}),flush=True)
        train_time=time.perf_counter()-train_start
        row={'epoch':epoch,'train_loss':total/seen,'lr':optimizer.param_groups[0]['lr'],'train_time_sec':train_time,
             'train_batches':total_batches,'mean_batch_time_sec':float(np.mean(batch_times))}; site_scores=[]; val_times={}; ac_times={}
        for site in val_sites:
            val_start=time.perf_counter(); _,yy,ss,aa=collect(model,val_loaders[site],records,device)
            if device.type=='cuda': torch.cuda.synchronize(device)
            val_times[site]=time.perf_counter()-val_start; ac_start=time.perf_counter(); mm=metric_bundle(yy,ss,aa,include_weighted=False); ac_times[site]=time.perf_counter()-ac_start; site_scores.append(mm['ac_auroc'])
            for key in ('ac_auroc','auroc','auprc','eligible_ac_pairs'): row[f'val_{site}_{key}']=mm[key]
        selection=float(np.mean(site_scores)); row['site_macro_ac']=selection; improved=selection>best+1e-4; row['is_best']=improved
        if improved: best=selection; best_epoch=epoch; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        row['val_time_sec']=float(sum(val_times.values())); row['ac_time_sec']=float(sum(ac_times.values())); row['epoch_total_time_sec']=time.perf_counter()-epoch_start
        row['best_epoch']=best_epoch; row['best_selection']=best; history.append(row)
        profile['epochs'].append({'epoch':epoch,'train_batches_total':total_batches,'mean_batch_time_sec':float(np.mean(batch_times)),
            'train_time_sec':train_time,'validation_time_sec_by_site':val_times,'ac_time_sec_by_site':ac_times,
            'epoch_total_time_sec':row['epoch_total_time_sec'],'selection':selection,'loss':row['train_loss']})
        print(json.dumps({'event':'epoch_end','holdout':holdout_site,'arm':arm,'epoch':epoch,'train_batches':f'{total_batches}/{total_batches}',
            'train_time':train_time,'val_time':val_times,'ac_time':ac_times,'loss':row['train_loss'],
            **{f'val_{site}_AC':row[f'val_{site}_ac_auroc'] for site in val_sites},'selection':selection,
            'best_epoch':best_epoch,'best_selection':best,'epoch_total_time':row['epoch_total_time_sec'],
            'elapsed_total':time.perf_counter()-started,'runtime':runtime_snapshot(device)}),flush=True)
        if epoch>=min_epochs and stale>=patience: break
    if best_state is None: raise RuntimeError('No checkpoint selected')
    model.load_state_dict(best_state)
    if save_checkpoint: torch.save({'state_dict':best_state,'best_epoch':best_epoch,'selection_score':best},fold_dir/'best_checkpoint.pt')
    holdout_loader=natural_loader(records,holdout,state,shuffle,batch_size); holdout_start=time.perf_counter(); _,yt,score,ages=collect(model,holdout_loader,records,device)
    if device.type=='cuda': torch.cuda.synchronize(device)
    holdout_inference=time.perf_counter()-holdout_start; holdout_ac_start=time.perf_counter(); metrics=metric_bundle(yt,score,ages); holdout_metric_time=time.perf_counter()-holdout_ac_start
    metrics.update({'arm':arm,'best_epoch':best_epoch,'validation_site_macro_ac':best,'training_wall_time_sec':time.perf_counter()-started,'initial_weight_hash':initial_hash})
    profile.update({'holdout_block_construction_sec':holdout_loader.block_build_wall_time_sec,'holdout_inference_sec':holdout_inference,
                    'holdout_metric_sec':holdout_metric_time,'device_end':runtime_snapshot(device),'total_fold_sec':time.perf_counter()-started})
    if profile_output is not None: json_dump(profile_output,profile)
    pd.DataFrame(history).to_csv(fold_dir/'training_history.csv',index=False); save_predictions(fold_dir/'outer_predictions.csv',records,holdout,score); json_dump(fold_dir/'metrics.json',metrics)
    json_dump(fold_dir/'config.json',{'source_git_sha':git_head(),'dataset':'DATA2','modality':modality,'arm':metrics['arm'],'x_seq_indices':list(range(470,483)) if modality=='stage_event13' else list(EEG_INDICES),'validity_definition':'metadata-only; no validity mask is supplied to model','imputation':preprocessing_json(state),'architecture':'LocalTCNClassifier','parameter_count':sum(p.numel() for p in model.parameters()),'receptive_field_epochs':13,'optimizer':{'name':'AdamW','lr':1e-3,'weight_decay':1e-4},'sampler':'site+class balanced 1/count(site,label)','pos_weight':1.0,'seed':7,'inner_split_seed':7,'shuffle_seed_phrase':'20260918_local_shuffle' if shuffle else None,'best_epoch':best_epoch,'checkpoint_metric':'mean AC across two seen-site validation sets','outer_holdout_never_used_for_selection':True,'batch_size':batch_size})
    return metrics

def write_inner_split(path,records,train,val,holdout):
    role={i:'inner_train' for i in train}; role.update({i:'inner_val' for i in val}); role.update({i:'outer_holdout' for i in holdout})
    pd.DataFrame([{'record_id':records[i].record_id,'patient_id':records[i].patient_id,'site':records[i].site,'label':records[i].label,'role':role[i]} for i in sorted(role)]).to_csv(path,index=False)

def aggregate(rows):
    out={}
    for arm in ('O','S','R'):
        selected=[r for r in rows if r['arm']==arm]; by={r['holdout_site']:r for r in selected}
        out[arm]={'sites':by}
        for metric in ('ac_auroc','age_weighted_auroc','auroc','auprc'):
            out[arm]['macro_'+metric]=float(np.mean([r[metric] for r in selected]))
        out[arm]['worst_ac_auroc']=float(min(r['ac_auroc'] for r in selected))
    return out

def bootstrap_delta(output,records,root,replicates=1000):
    rng=np.random.default_rng(20260918); samples={site:{} for site in SITES}
    for site in SITES:
        for arm in ('S','R'):
            frame=pd.read_csv(root/arm/f'holdout_{site}'/'outer_predictions.csv'); samples[site][arm]=frame
        if list(samples[site]['S'].record_id)!=list(samples[site]['R'].record_id): raise RuntimeError('Paired predictions misaligned')
    rows=[]
    site_deltas={s:[] for s in SITES}; macro=[]
    for rep in range(replicates):
        current={}
        for site in SITES:
            f=samples[site]['S']; pos=np.flatnonzero(f.label.to_numpy()==1); neg=np.flatnonzero(f.label.to_numpy()==0); choose=np.r_[rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)]
            s=vectorized_ac(f.label.to_numpy()[choose],f.raw_score.to_numpy()[choose],f.age.to_numpy()[choose],2); rf=samples[site]['R']; rr=vectorized_ac(rf.label.to_numpy()[choose],rf.raw_score.to_numpy()[choose],rf.age.to_numpy()[choose],2); current[site]=rr-s; site_deltas[site].append(current[site])
        macro.append(float(np.mean(list(current.values()))))
    for name,values in [*(site_deltas.items()),('Macro',macro)]:
        a=np.asarray(values); rows.append({'scope':name,'mean_delta':a.mean(),'bootstrap_sd':a.std(ddof=1),'ci_low':np.quantile(a,.025),'ci_high':np.quantile(a,.975),'replicates':len(a)})
    pd.DataFrame(rows).to_csv(output,index=False)

def write_report(path,modality,agg,rows,total_wall):
    label='T1' if modality=='stage_event13' else 'T2'; title='Stage/Event13' if modality=='stage_event13' else 'EEG72'
    lines=[f'# {label} {title} temporal LOSO', '', 'Outer training sites were internally split 80/20 for checkpoint selection; no outer holdout sample was used for training or model selection.','',
        '| Model | I0002 AC | I0006 AC | S0001 AC | Macro | Worst |','|---|---:|---:|---:|---:|---:|']
    for arm,desc in [('O',f'{label}-O {title} orderless LR'),('S',f'{label}-S shuffled local TCN'),('R',f'{label}-R real-order local TCN')]:
        a=agg[arm]; lines.append(f"| {desc} | {a['sites']['I0002']['ac_auroc']:.6f} | {a['sites']['I0006']['ac_auroc']:.6f} | {a['sites']['S0001']['ac_auroc']:.6f} | {a['macro_ac_auroc']:.6f} | {a['worst_ac_auroc']:.6f} |")
    if modality=='stage_event13': lines.append('| CAISR20 reference | 0.746023 | 0.677706 | 0.616662 | 0.680131 | 0.616662 |')
    lines += ['',f"Real - Shuffle macro AC: {agg['R']['macro_ac_auroc']-agg['S']['macro_ac_auroc']:+.6f}.",f"Real - Orderless macro AC: {agg['R']['macro_ac_auroc']-agg['O']['macro_ac_auroc']:+.6f}.",f'Total wall time: {total_wall:.1f} seconds.','',
        'The principal temporal-order comparison is real versus shuffled order, not comparison with CAISR20. Validity metadata was used only for training-fold neutral imputation and was not provided to the network.']
    Path(path).write_text('\n'.join(lines)+'\n')

def run(modality,output_root,data_root=DATA_ROOT,smoke=False,only_holdout=None,arms=('O','S','R'),profile_benchmark=False):
    started=time.time(); records,_,load_sec=load_dataset(data_root,modality,False); unusable=[r.record_id for r in records if r.n_blocks<1]; records=[r for r in records if r.n_blocks>=1]; output_root=Path(output_root)
    if output_root.exists() and any(output_root.iterdir()): raise FileExistsError(output_root)
    output_root.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); rows=[]
    holds=[only_holdout] if only_holdout else list(SITES)
    for hold in holds:
        outer_train=[i for i,r in enumerate(records) if r.site!=hold]; holdout=[i for i,r in enumerate(records) if r.site==hold]; inner_train,inner_val=inner_split(records,outer_train)
        if set(records[i].patient_id for i in outer_train)&set(records[i].patient_id for i in holdout): raise RuntimeError('Outer leakage')
        for arm in arms:
            fold=output_root/arm/f'holdout_{hold}'; fold.mkdir(parents=True); write_inner_split(fold/'inner_split.csv',records,inner_train,inner_val,holdout)
            arm_start=time.time()
            if arm=='O': metric=run_orderless_fold(records,outer_train,holdout,fold,modality)
            else: metric=run_tcn_fold(records,inner_train,inner_val,holdout,fold,modality,arm=='S',device,
                max_epochs=2 if (smoke or profile_benchmark) else 30,min_epochs=2 if (smoke or profile_benchmark) else 8,
                patience=1 if (smoke or profile_benchmark) else 5,holdout_site=hold,save_checkpoint=not profile_benchmark,
                profile_output=(fold/'profile.json') if profile_benchmark else None)
            metric.update({'holdout_site':hold,'wall_time_sec':time.time()-arm_start,'inner_train_n':len(inner_train),'inner_val_n':len(inner_val),'outer_train_n':len(outer_train),'holdout_n':len(holdout)}); rows.append(metric)
    pd.DataFrame(rows).to_csv(output_root/'fold_metrics.csv',index=False)
    if len(holds)==3:
        agg=aggregate(rows); json_dump(output_root/'aggregate_metrics.json',agg); bootstrap_delta(output_root/'paired_bootstrap.csv',records,output_root,1000)
        comparisons=[]
        for comp,a,b in [('R-S','R','S'),('R-O','R','O')]:
            row={'comparison':comp}
            for site in SITES: row[site]=agg[a]['sites'][site]['ac_auroc']-agg[b]['sites'][site]['ac_auroc']
            row['Macro']=agg[a]['macro_ac_auroc']-agg[b]['macro_ac_auroc']; row['Worst']=agg[a]['worst_ac_auroc']-agg[b]['worst_ac_auroc']; comparisons.append(row)
        pd.DataFrame(comparisons).to_csv(output_root/'comparisons.csv',index=False); write_report(output_root/'REPORT.md',modality,agg,rows,time.time()-started)
    json_dump(output_root/'config.json',{'source_git_sha':git_head(),'dataset':'DATA2','data_root':str(Path(data_root).resolve()),'modality':modality,'records_total':6600,'records_usable':len(records),'unusable_no_complete_block':unusable,'load_wall_time_sec':load_sec,'device':runtime_snapshot(device),'smoke':smoke,'profile_benchmark':profile_benchmark,'arms':list(arms),'seed':7})
    return rows,time.time()-started

def smoke(data_root,out):
    paths=sorted(Path(data_root).glob("*/*.npz"),key=lambda p:(p.parent.name,p.stem)); meta=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as z: meta.append((path,str(z["site_id"].item()),int(z["y"].item()),float(z["x_static"][0])))
    chosen=[]; rng=np.random.default_rng(7)
    for site in SITES:
        pos=[i for i,(_,s,y,_) in enumerate(meta) if s==site and y==1]; neg=[i for i,(_,s,y,_) in enumerate(meta) if s==site and y==0]
        chosen_pos=sorted(rng.choice(pos,4,replace=False).tolist()); eligible=[n for n in neg if any(abs(meta[p][3]-meta[n][3])<=2 for p in chosen_pos)]
        if not eligible: raise RuntimeError(f"No smoke-eligible negative for {site}")
        forced=int(rng.choice(eligible)); chosen_neg=[forced]+rng.choice([n for n in neg if n!=forced],3,replace=False).tolist(); chosen += chosen_pos+sorted(chosen_neg)
    records=[_selected(meta[i][0],"stage_event13",False)[0] for i in chosen]; selected=list(range(len(records))); out=Path(out)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subset=records
    for hold in SITES:
        outer=[i for i,r in enumerate(subset) if r.site!=hold]; test=[i for i,r in enumerate(subset) if r.site==hold]
        # Choose one deterministic age-matched positive/negative validation pair per seen site.
        train=[]; val=[]
        for site in sorted({subset[i].site for i in outer}):
            pos=[i for i in outer if subset[i].site==site and subset[i].label==1]; neg=[i for i in outer if subset[i].site==site and subset[i].label==0]
            pairs=[(abs(subset[p].age-subset[n].age),subset[p].record_id,subset[n].record_id,p,n) for p in pos for n in neg if abs(subset[p].age-subset[n].age)<=2]
            if not pairs: raise RuntimeError(f"Smoke selection lacks eligible AC pair for {site}")
            _,_,_,vp,vn=min(pairs); val += [vp,vn]; train += [i for i in pos+neg if i not in (vp,vn)]
        state=fit_preprocessing(subset,train,'stage_event13'); seed_all(7); m1=LocalTCNClassifier(13); h1=state_hash(m1); seed_all(7); m2=LocalTCNClassifier(13); h2=state_hash(m2)
        if h1!=h2: raise RuntimeError('Real/shuffle initial weights differ')
        for arm,shuffle in [('S',True),('R',False)]:
            fold=out/arm/f'holdout_{hold}'; fold.mkdir(parents=True); run_tcn_fold(subset,train,val,test,fold,'stage_event13',shuffle,device,max_epochs=2,min_epochs=2,patience=1,batch_size=64)
        ofold=out/'O'/f'holdout_{hold}'; ofold.mkdir(parents=True); run_orderless_fold(subset,outer,test,ofold,'stage_event13')
    json_dump(out/'smoke_pass.json',{'pass':True,'selected_per_site_label':4,'epochs':2,'initial_weights_identical':True,'sampler_definition_identical':True,'only_difference_real_vs_shuffle':'within-block deterministic permutation'})

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--modality',choices=MODALITY_DIM); p.add_argument('--data-root',default=str(DATA_ROOT)); p.add_argument('--output-root'); p.add_argument('--smoke',action='store_true'); p.add_argument('--only-holdout',choices=SITES); p.add_argument('--arms',default='O,S,R'); p.add_argument('--profile-benchmark',action='store_true')
    a=p.parse_args()
    if a.smoke: smoke(a.data_root,a.output_root or 'temporal_results/smoke')
    else:
        if not a.modality or not a.output_root: p.error('--modality and --output-root required')
        arms=tuple(v.strip() for v in a.arms.split(',') if v.strip())
        if any(v not in ('O','S','R') for v in arms): p.error('--arms must contain only O,S,R')
        if a.profile_benchmark and (a.modality!='stage_event13' or a.only_holdout!='S0001' or arms!=('R',)):
            p.error('profiling benchmark is fixed to stage_event13, holdout S0001, arm R')
        run(a.modality,a.output_root,a.data_root,False,a.only_holdout,arms,a.profile_benchmark)
