"""Deterministic per-record pooling for the frozen SIREN Branch A contract."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

STAGES = {"Wake": {5}, "REM": {4}, "N2": {2}, "N3": {1},
          "NREM": {1, 2, 3}, "All-stage": {1, 2, 3, 4, 5}}
LEADS = {"F3": 0, "F4": 1, "C3": 2, "C4": 3, "O1": 4, "O2": 5}


@dataclass
class NightFeatureResult:
    unique_ids: list[str]
    unique_feature_values: np.ndarray
    unique_feature_valid_mask: np.ndarray
    unique_feature_support_count: np.ndarray
    view_values: dict[str, np.ndarray]
    view_valid_masks: dict[str, np.ndarray]
    view_support_counts: dict[str, np.ndarray]


@dataclass
class PreparedRecord:
    raw: dict
    xseq: np.ndarray
    xecg: np.ndarray
    static: np.ndarray
    stage_codes: np.ndarray
    stage_valid: np.ndarray
    stage_masks: dict[str, np.ndarray]
    lead_valid: dict[str, np.ndarray]
    hrv_success: np.ndarray
    hrv_feature_valid: np.ndarray
    emg_channel_available: np.ndarray
    emg_epoch_success: np.ndarray
    resp_feature_valid: np.ndarray


def _prepare_record(record):
    xseq = np.asarray(record["X_seq"], dtype=np.float64)
    xecg = np.asarray(record["X_ecg"], dtype=np.float64)
    static = np.asarray(record["x_static"], dtype=np.float64)
    codes = np.asarray(record["stage_code_aligned"], dtype=np.float64)
    stage_valid = np.asarray(record["stage_valid"], dtype=bool)
    stage_masks = {stage: stage_valid & np.isfinite(codes) & np.isin(codes, list(stage_codes))
                   for stage, stage_codes in STAGES.items()}
    available = np.asarray(record["eeg_channel_available"], dtype=bool)
    clean = np.asarray(record["eeg_clean_subsegment_count"])
    lead_valid = {lead: bool(available[index]) & (clean[:, index] > 0)
                  for lead, index in LEADS.items()}
    return PreparedRecord(
        raw=record, xseq=xseq, xecg=xecg, static=static,
        stage_codes=codes, stage_valid=stage_valid,
        stage_masks=stage_masks, lead_valid=lead_valid,
        hrv_success=np.asarray(record["hrv_success"], dtype=bool),
        hrv_feature_valid=np.asarray(record["hrv_feature_valid"], dtype=bool),
        emg_channel_available=np.asarray(record["emg_channel_available"], dtype=bool),
        emg_epoch_success=np.asarray(record["emg_epoch_success"], dtype=bool),
        resp_feature_valid=np.asarray(record["resp_feature_valid"], dtype=bool),
    )


def pool_values(values, valid, aggregation):
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(values)
    support = int(mask.sum())
    if support < 5:
        return np.nan, False, support
    selected = values[mask]
    if aggregation == "Mean":
        value = float(selected.mean())
    elif aggregation == "IQR":
        q25, q75 = np.quantile(selected, [0.25, 0.75], method="linear")
        value = float(q75 - q25)
    else:
        raise ValueError(f"Unsupported aggregation: {aggregation}")
    return value, True, support


def _stage_mask(prepared, stage): return prepared.stage_masks[stage]


def _lead_valid(prepared, lead): return prepared.lead_valid[lead]


def entropy_source_valid(record, value):
    if record["annotation_status"] != "available" or not np.isfinite(value): return False
    codes=np.asarray(record["stage_code_aligned"],float); sv=np.asarray(record["stage_valid"],bool); t=len(sv)
    if codes.shape != (t,) or not np.any(sv & np.isin(codes,[1,2,3,4])): return False
    p=np.asarray(record["stage_probability"],float); pv=np.asarray(record["stage_probability_valid"],bool)
    present=np.asarray(record["stage_probability_channels_present"],bool)
    if p.shape!=(t,5) or pv.shape!=(t,) or present.shape!=(5,) or not present.all() or not pv.all(): return False
    aliases=(("caisr_prob_n1",),("caisr_prob_n2",),("caisr_prob_n3",),("caisr_prob_r","caisr_prob_rem"),("caisr_prob_w","caisr_prob_wake"))
    lengths=record["annotation_signal_lengths"]
    if any(not any(lengths.get(k)==t for k in keys) for keys in aliases): return False
    return bool(np.isfinite(p).all() and np.all((p>=0)&(p<=1)) and np.all(p.sum(axis=1)>0))


def compute_demo5(static):
    x=np.asarray(static,float)
    if x.shape!=(196,): raise ValueError("x_static shape")
    sex=x[1:4]; race=x[4:9]
    if not np.isin(sex,[0,1]).all() or sex.sum()!=1: raise ValueError("Illegal sex one-hot")
    if not np.isin(race,[0,1]).all() or race.sum()!=1: raise ValueError("Illegal race one-hot")
    values=np.full(5,np.nan); valid=np.zeros(5,bool); support=np.zeros(5,np.int64)
    if sex[2]==0: values[0]=sex[1]; valid[0]=True; support[0]=1
    if np.isfinite(x[9]) and x[9]!=0: values[1]=x[9]; valid[1]=True; support[1]=1
    if race[3]==0: values[2:]=[race[4],race[1],race[0]]; valid[2:]=True; support[2:]=1
    return values,valid,support


def _spatial_prepared(row, prepared):
    idx=[int(x) for x in row["source_indices"].split(";")]
    x=prepared.xseq; stage=_stage_mask(prepared,row["stage_group"])
    if row["family"]=="spatial_asymmetry_candidate":
        if len(idx)!=6: raise ValueError("Asymmetry requires six indices")
        left,right=row["channel"].split("-")
        raw=x[:,idx]; value=(raw[:,0]+raw[:,2]-raw[:,4])-(raw[:,1]+raw[:,3]-raw[:,5])
        valid=stage & _lead_valid(prepared,left) & _lead_valid(prepared,right) & np.isfinite(raw).all(axis=1)
    elif row["family"]=="spatial_ap_gradient_candidate":
        if len(idx)!=12 or any(c in row["channel"] for c in ("C3","C4")): raise ValueError("Gradient indices")
        raw=x[:,idx]; rel=raw[:,:4]+raw[:,4:8]-raw[:,8:12]
        value=rel[:,:2].mean(axis=1)-rel[:,2:].mean(axis=1)
        valid=stage & np.isfinite(raw).all(axis=1)
        for lead in ("F3","F4","O1","O2"): valid &= _lead_valid(prepared,lead)
    else: raise ValueError("Not a spatial row")
    return pool_values(value,valid,"Mean")


def spatial_value_and_mask(row, record):
    return _spatial_prepared(row, _prepare_record(record))


def _compute_row(row, prepared, demo):
    fam=row["family"]; xseq=prepared.xseq; record=prepared.raw
    if fam=="EEG54_stage_spectral":
        idx=int(row["source_index"]); valid=_stage_mask(prepared,row["stage_group"]) & _lead_valid(prepared,row["channel"])
        return pool_values(xseq[:,idx],valid,row["aggregation"])
    if fam.startswith("spatial_"): return _spatial_prepared(row,prepared)
    if fam=="HRV11":
        idx=int(row["source_index"]); valid=prepared.hrv_success&prepared.hrv_feature_valid[:,idx]
        return pool_values(prepared.xecg[:,idx],valid,row["aggregation"])
    if fam=="EMG24":
        idx=int(row["source_index"]); lead={"chin":0,"lleg":1,"rleg":2}[row["channel"]]
        valid=bool(prepared.emg_channel_available[lead]) & prepared.emg_epoch_success[:,lead]
        return pool_values(xseq[:,idx],valid,row["aggregation"])
    if fam=="Resp14":
        idx=int(row["source_index"]); valid=prepared.resp_feature_valid[:,idx-456]
        return pool_values(xseq[:,idx],valid,row["aggregation"])
    if fam=="Demo5":
        order={"demo_male":0,"demo_bmi":1,"demo_race_white":2,"demo_race_black":3,"demo_race_asian":4}; i=order[row["unique_feature_id"]]
        return float(demo[0][i]),bool(demo[1][i]),int(demo[2][i])
    if fam=="CAISR20":
        idx=int(row["source_index"]); value=float(prepared.static[idx])
        valid=record["annotation_status"]=="available" and bool(np.any(prepared.stage_valid&np.isin(prepared.stage_codes,[1,2,3,4,5]))) and np.isfinite(value)
        uid=row["unique_feature_id"]
        if uid=="caisr_sleep_mean_stage_entropy": valid=entropy_source_valid(record,value)
        elif uid.startswith("caisr_arousal_"): valid &= bool(np.any(record["arousal_valid"]))
        elif uid.startswith("caisr_respiratory_"): valid &= bool(np.any(record["resp_event_valid"]))
        elif uid.startswith("caisr_limb_"): valid &= bool(np.any(record["limb_event_valid"]))
        return (value if valid else np.nan),bool(valid),int(valid)
    raise ValueError(f"Unknown family {fam}")


def compute_night_features(record, manifest):
    prepared=_prepare_record(record)
    demo=compute_demo5(prepared.static); by_id={}
    for row in manifest.rows:
        uid=row["unique_feature_id"]
        if uid not in by_id: by_id[uid]=_compute_row(row,prepared,demo)
    values=np.array([by_id[u][0] for u in manifest.unique_ids],dtype=np.float64)
    valid=np.array([by_id[u][1] for u in manifest.unique_ids],dtype=bool)
    support=np.array([by_id[u][2] for u in manifest.unique_ids],dtype=np.int64)
    if not np.isfinite(values[valid]).all() or not np.isnan(values[~valid]).all() or np.any(support<0): raise ValueError("Output invariant violation")
    vv={}; vm={}; vs={}
    for view,size in manifest.view_sizes.items():
        rows=[r for r in manifest.rows if r["view_id"]==view]
        vv[view]=np.array([by_id[r["unique_feature_id"]][0] for r in rows],float)
        vm[view]=np.array([by_id[r["unique_feature_id"]][1] for r in rows],bool)
        vs[view]=np.array([by_id[r["unique_feature_id"]][2] for r in rows],np.int64)
        if len(vv[view])!=size: raise ValueError("View size")
    return NightFeatureResult(manifest.unique_ids,values,valid,support,vv,vm,vs)
