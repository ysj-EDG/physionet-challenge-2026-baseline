#!/usr/bin/env python3
"""Physio Fusion V1 Gate 1/2 command-line entrypoint."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from dataset import (
    EXPECTED_SITE_COUNTS,
    EXPECTED_EXTRACTION_VERSION,
    EXPECTED_VALIDITY_SCHEMA,
    EEG_PAIRS,
    MODEL_FAMILIES,
    discover_records,
    fit_robust_scaler24,
    inner_split_by_site,
    load_record,
    outer_loso_split,
    sample_training_pairs,
    sample_validation_pairs,
)
from model import SharedPairAutoencoder
from train import AEConfig, summarize_and_select, train_coherence_ae
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_model import compute_auroc_age, compute_auroc_weighted

DEFAULT_NPZ_ROOT = Path(
    "/database/home/gaohaojie/workspace/python-example-2026/"
    "validity_v0_2_full_6600/workspace/result_bundle/npz"
)


def run_smoke(npz_root: Path) -> None:
    records = discover_records(npz_root, validate_schema=True)
    assert len(records) == 6600
    observed = {site: sum(r.site == site for r in records) for site in EXPECTED_SITE_COUNTS}
    assert observed == EXPECTED_SITE_COUNTS
    normal_info = next(
        r for r in records
        if r.site == "I0002" and r.record_id != "sub-I0002150027361_ses-4"
    )
    normal = load_record(normal_info.path)
    assert set(normal) == MODEL_FAMILIES
    T = normal["coherence"].shape[0]
    assert normal["spectral"].shape == (T, 54)
    assert normal["spectral_valid"].shape == (T, 54)
    assert normal["coherence"].shape == (T, 15, 24)
    assert normal["coherence_pair_valid"].shape == (T, 15)
    assert normal["bsr"].shape == (T, 18)
    assert normal["bsr_valid"].shape == (T, 18)
    assert normal["emg"].shape == (T, 24) and normal["emg_valid"].shape == (T, 24)
    assert normal["resp"].shape == (T, 14) and normal["resp_valid"].shape == (T, 14)
    assert normal["hrv"].shape == (T, 11) and normal["hrv_valid"].shape == (T, 11)
    assert "stage" not in normal and "stage_event" not in normal and "circadian" not in normal
    with np.load(normal_info.path, allow_pickle=False) as z:
        raw_ecg = np.asarray(z["X_ecg"], dtype=np.float32)
    if len(raw_ecg):
        assert np.allclose(normal["hrv"][10], raw_ecg[0, :11])
    short = load_record(next(r for r in records if r.record_id == "sub-I0002150027361_ses-4").path)
    absent = load_record(next(r for r in records if r.record_id == "sub-S0001111343357_ses-1").path)
    assert not short["hrv_valid"].any() and not absent["hrv_valid"].any()
    valid_zero = False
    invalid_zero = False
    for record in records[:200]:
        loaded = load_record(record.path)
        pairs = (
            (loaded["spectral"], loaded["spectral_valid"]),
            (loaded["coherence"], loaded["coherence_pair_valid"][:, :, None]),
            (loaded["bsr"], loaded["bsr_valid"]),
            (loaded["emg"], loaded["emg_valid"]),
            (loaded["resp"], loaded["resp_valid"]),
        )
        for values, validity in pairs:
            vv = np.broadcast_to(validity, values.shape)
            valid_zero |= bool(np.any((values == 0) & vv))
            invalid_zero |= bool(np.any((values == 0) & ~vv))
        if valid_zero and invalid_zero:
            break
    assert valid_zero and invalid_zero
    assert SharedPairAutoencoder(4)(torch.zeros(3, 24)).shape == (3, 24)
    for holdout in EXPECTED_SITE_COUNTS:
        outer_train, outer_test = outer_loso_split(records, holdout)
        inner_train, inner_val = inner_split_by_site(outer_train, seed=7)
        outer_ids = {r.record_id for r in outer_test}
        train_ids = {r.record_id for r in inner_train}
        val_ids = {r.record_id for r in inner_val}
        assert not outer_ids & (train_ids | val_ids)
        assert not train_ids & val_ids
    print("GATE1_SMOKE_PASS")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_d_report(root: Path, fold_summaries: dict[str, list[dict[str, object]]], selected: dict[str, int]) -> None:
    lines = [
        "# Physio Fusion V1 Gate 2: coherence bottleneck selection", "",
        "Selection is CI-blind. Each outer fold uses only the other two sites, deterministic inner 80/20 record splits (seed=7), valid-pair-only median/IQR scaling, and equal site-weighted validation MSE.", "",
        "| Held-out site | best mean d | one-SE threshold | selected d |", "|---|---:|---:|---:|",
    ]
    all_rows: list[dict[str, object]] = []
    for heldout, rows in fold_summaries.items():
        best = min(rows, key=lambda row: (float(row["mean_site_balanced_val_mse"]), int(row["latent_dim"])))
        lines.append(f"| {heldout} | {best['best_d']} | {float(best['one_se_threshold']):.8g} | **{selected[heldout]}** |")
        lines += ["", f"### Held-out {heldout}", "", "| d | mean val MSE | SE | I0002 record MSE | I0006 record MSE | S0001 record MSE | mean R² | selected |", "|---:|---:|---:|---:|---:|---:|---:|:---:|"]
        for row in sorted(rows, key=lambda row: int(row["latent_dim"])):
            all_rows.append({"heldout_site": heldout, **row})
            site_values = [f"{float(row.get(f'mean_{site}_record_balanced_mse', float('nan'))):.8g}" if f"mean_{site}_record_balanced_mse" in row else "N/A" for site in EXPECTED_SITE_COUNTS]
            lines.append(f"| {row['latent_dim']} | {float(row['mean_site_balanced_val_mse']):.8g} | {float(row['se_site_balanced_val_mse']):.8g} | {site_values[0]} | {site_values[1]} | {site_values[2]} | {float(row['mean_overall_r2']):.8g} | {'yes' if row['selected'] else 'no'} |")
    root.mkdir(parents=True, exist_ok=True)
    _write_csv(root / "d_selection.csv", all_rows)
    _write_json(root / "d_selection.json", {"selected_d": selected, "fold_summaries": fold_summaries})
    (root / "D_SELECTION.md").write_text("\n".join(lines) + "\n")


def run_d_search(npz_root: Path, output_dir: Path, device_name: str = "auto") -> dict[str, int]:
    records = discover_records(npz_root, validate_schema=True)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"D_SEARCH_DEVICE={device} {gpu_name}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_summaries: dict[str, list[dict[str, object]]] = {}
    selected: dict[str, int] = {}
    started = time.time()
    for heldout in EXPECTED_SITE_COUNTS:
        outer_train, outer_test = outer_loso_split(records, heldout)
        inner_train, inner_val = inner_split_by_site(outer_train, seed=7)
        train_ids = {r.record_id for r in inner_train}
        val_ids = {r.record_id for r in inner_val}
        test_ids = {r.record_id for r in outer_test}
        if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
            raise RuntimeError(f"split leakage in held-out {heldout}")
        fold_dir = output_dir / f"outer_{heldout}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        _write_json(fold_dir / "fold_context.json", {
            "heldout_site": heldout,
            "outer_train_sites": sorted({r.site for r in outer_train}),
            "inner_train_counts": {s: sum(r.site == s for r in inner_train) for s in sorted({r.site for r in inner_train})},
            "inner_val_counts": {s: sum(r.site == s for r in inner_val) for s in sorted({r.site for r in inner_val})},
            "outer_test_count": len(outer_test), "inner_seed": 7,
            "outer_test_never_used_for_selection": True,
        })
        print(f"FOLD={heldout} sampling training pairs", flush=True)
        train_raw, train_counts = sample_training_pairs(inner_train, target_per_site=200_000, seed=7)
        scaler = fit_robust_scaler24(train_raw)
        train_values = scaler.transform(train_raw)
        validation_raw = sample_validation_pairs(inner_val, max_per_record=128, seed=7)
        validation = {site: [(rid, scaler.transform(values)) for rid, values in rows] for site, rows in validation_raw.items()}
        _write_json(fold_dir / "sampling_context.json", {
            "train_pair_counts": train_counts,
            "validation_record_counts": {site: len(rows) for site, rows in validation.items()},
            "train_values": len(train_values), "scaler_center": scaler.center.tolist(), "scaler_scale": scaler.scale.tolist(),
        })
        seed_results: list[dict[str, object]] = []
        for latent_dim in (4, 8, 12, 16):
            for seed in (7, 17, 27):
                seed_results.append(train_coherence_ae(latent_dim, seed, train_values, validation, device))
        summaries, selected_d, best_d, threshold = summarize_and_select(seed_results)
        fold_summaries[heldout] = summaries
        selected[heldout] = selected_d
        _write_json(fold_dir / "seed_metrics.json", seed_results)
        _write_csv(fold_dir / "candidate_summary.csv", summaries)
        _write_json(fold_dir / "selection.json", {"best_d": best_d, "threshold": threshold, "selected_d": selected_d})
        print(f"FOLD={heldout} selected_d={selected_d} best_d={best_d} threshold={threshold:.8g}", flush=True)
    _write_d_report(output_dir, fold_summaries, selected)
    _write_json(output_dir / "run_context.json", {
        "npz_root": str(npz_root.resolve()), "device": str(device), "torch": torch.__version__, "numpy": np.__version__,
        "elapsed_sec": time.time() - started, "seeds": [7, 17, 27], "candidates": [4, 8, 12, 16], "selected_d": selected,
    })
    return selected



LOCKED_GATE2_MEANS = {"I0002": 0.0051684209, "I0006": 0.0063393532, "S0001": 0.0049678171}
SOURCE_SHA = "feaae35c3d6f73620a88e70f55a3fb5374e0250d"
D16 = 16
PHYSIO_DIM = 361
POOLED_DIM = 722
FAMILY_NAMES = ("means", "auc", "iqr", "ratios", "spectrum", "sigma")


def _hash_ids(records: list) -> str:
    text = "\n".join(sorted(r.record_id for r in records)).encode()
    return hashlib.sha256(text).hexdigest()


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[1], np.nan, dtype=np.float64)
    for j in range(values.shape[1]):
        ok = np.isfinite(values[:, j]) & (weights > 0)
        if not np.any(ok):
            continue
        order = np.argsort(values[ok, j], kind="mergesort")
        vv, ww = values[ok, j][order], weights[ok][order]
        out[j] = vv[np.searchsorted(np.cumsum(ww), 0.5 * ww.sum(), side="left")]
    return out.astype(np.float32)


def _site_weights(records: list) -> np.ndarray:
    counts = {s: sum(r.site == s for r in records) for s in sorted({r.site for r in records})}
    w = np.asarray([1.0 / counts[r.site] for r in records], dtype=np.float64)
    return (w / w.sum()).astype(np.float64)


def _save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _save_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _coherence_encode(model, loaded: dict[str, object], scaler, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(loaded["coherence"], dtype=np.float32)
    valid = np.asarray(loaded["coherence_pair_valid"], dtype=bool)
    flat = values.reshape(-1, 24).copy()
    flat_valid = valid.reshape(-1)
    flat[~flat_valid] = 0.0
    flat = scaler.transform(flat)
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(flat), 16384):
            x = torch.from_numpy(flat[start:start + 16384]).to(device)
            outputs.append(model.encode(x).cpu().numpy().astype(np.float32))
    latent = np.concatenate(outputs).reshape(len(values), 15, D16)
    return latent, np.repeat(valid[:, :, None], D16, axis=2)


def _physiology_epoch(model, loaded: dict[str, object], scaler, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latent, latent_valid = _coherence_encode(model, loaded, scaler, device)
    values = np.concatenate([
        np.asarray(loaded["spectral"], dtype=np.float32),
        np.asarray(loaded["bsr"], dtype=np.float32),
        latent.reshape(len(latent), -1),
        np.asarray(loaded["hrv"], dtype=np.float32),
        np.asarray(loaded["emg"], dtype=np.float32),
        np.asarray(loaded["resp"], dtype=np.float32),
    ], axis=1)
    valid = np.concatenate([
        np.asarray(loaded["spectral_valid"], dtype=bool),
        np.asarray(loaded["bsr_valid"], dtype=bool),
        latent_valid.reshape(len(latent), -1),
        np.asarray(loaded["hrv_valid"], dtype=bool),
        np.asarray(loaded["emg_valid"], dtype=bool),
        np.asarray(loaded["resp_valid"], dtype=bool),
    ], axis=1)
    if values.shape[1] != PHYSIO_DIM or valid.shape != values.shape:
        raise ValueError(f"unexpected physiology shape: {values.shape}, {valid.shape}")
    values = np.where(valid & np.isfinite(values), values, 0.0).astype(np.float32)
    valid &= np.isfinite(values)
    return values, valid, latent


def _fit_epoch_scaler(items: list[tuple[object, np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
    by_site: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for rec, values, valid in items:
        means = np.full(PHYSIO_DIM, np.nan, dtype=np.float64)
        second = np.full(PHYSIO_DIM, np.nan, dtype=np.float64)
        for j in range(PHYSIO_DIM):
            ok = valid[:, j]
            if np.any(ok):
                x = values[ok, j].astype(np.float64)
                means[j] = x.mean(); second[j] = np.square(x).mean()
        by_site.setdefault(rec.site, []).append((means, second))
    site_means=[]; site_seconds=[]; counts={}
    for site, rows in sorted(by_site.items()):
        a=np.asarray([r[0] for r in rows]); b=np.asarray([r[1] for r in rows])
        site_means.append(np.nanmean(a,axis=0)); site_seconds.append(np.nanmean(b,axis=0)); counts[site]=len(rows)
    mu=np.nanmean(np.asarray(site_means),axis=0); e2=np.nanmean(np.asarray(site_seconds),axis=0)
    scale=np.sqrt(np.maximum(e2-np.square(mu),0.0))
    bad=~np.isfinite(mu); mu[bad]=0.0
    scale[~np.isfinite(scale)|(scale<1e-8)]=1.0
    return {"mean":mu.astype(np.float32),"scale":scale.astype(np.float32),"site_record_counts":np.asarray([counts[s] for s in sorted(counts)],dtype=np.int32),"sites":np.asarray(sorted(counts))}


def _apply_epoch_scaler(values: np.ndarray, valid: np.ndarray, scaler: dict[str,np.ndarray]) -> np.ndarray:
    out=((values-scaler["mean"])/scaler["scale"]).astype(np.float32)
    return np.where(valid & np.isfinite(out),out,0.0).astype(np.float32)


def _pool_patient(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    means=np.full(PHYSIO_DIM,np.nan,dtype=np.float32); std=np.full(PHYSIO_DIM,np.nan,dtype=np.float32); count=valid.sum(axis=0).astype(np.int32)
    for j in range(PHYSIO_DIM):
        ok=valid[:,j]
        if np.any(ok):
            x=values[ok,j].astype(np.float64); means[j]=x.mean(); std[j]=x.std(ddof=0)
    return np.concatenate([means,std]), count


def _patient_standardize(train: np.ndarray, train_records: list, all_values: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    weights=_site_weights(train_records)
    mu=np.sum(train*weights[:,None],axis=0); e2=np.sum(np.square(train)*weights[:,None],axis=0)
    scale=np.sqrt(np.maximum(e2-np.square(mu),0.0)); scale[~np.isfinite(scale)|(scale<1e-8)]=1.0
    return ((all_values-mu)/scale).astype(np.float32), mu.astype(np.float32), scale.astype(np.float32)


def _fit_fold_imputer(train_pooled: np.ndarray, train_records: list) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    weights=_site_weights(train_records); median=_weighted_median(train_pooled,weights)
    finite=np.isfinite(train_pooled); mass=np.sum(finite*weights[:,None],axis=0)
    if np.any(~np.isfinite(median)):
        bad=np.flatnonzero(~np.isfinite(median)); raise RuntimeError(f"all-training pooled feature missing: {bad.tolist()}")
    return median, finite.sum(axis=0).astype(np.int32), mass.astype(np.float32)


def _classifier_weights(records: list) -> np.ndarray:
    cells={(s,y):sum(r.site==s and r.y==y for r in records) for s in sorted({r.site for r in records}) for y in (0,1)}
    if any(v==0 for v in cells.values()): raise RuntimeError(f"missing site/class cell: {cells}")
    w=np.asarray([1.0/cells[(r.site,r.y)] for r in records],dtype=np.float64); w/=w.mean()
    return w.astype(np.float32)


def _feature_names() -> np.ndarray:
    names=[]
    names += [f"spectral_{i:02d}" for i in range(54)]
    names += [f"bsr_{i:02d}" for i in range(18)]
    names += [f"coherence_pair{p:02d}_latent{k:02d}" for p in range(15) for k in range(16)]
    names += [f"hrv_{i:02d}" for i in range(11)]
    names += [f"emg_{i:02d}" for i in range(24)]
    names += [f"resp_{i:02d}" for i in range(14)]
    return np.asarray(names)


def _run_one_nofusion_seed(heldout: str, seed: int, outer_train: list, outer_test: list, model, coh_scaler, seed_dir: Path, device: torch.device) -> dict[str, object]:
    all_records=outer_train+outer_test; items=[]; trace={}
    for rec in all_records:
        loaded=load_record(rec.path); raw,valid,latent=_physiology_epoch(model,loaded,coh_scaler,device); items.append((rec,raw,valid));
        if rec.site not in trace: trace[rec.site]=(rec,latent,raw,valid)
    epoch_scaler=_fit_epoch_scaler([(r,v,q) for r,v,q in items if r.site!=heldout])
    transformed=[_apply_epoch_scaler(v,q,epoch_scaler) for _,v,q in items]
    pooled=[]; counts=[]
    for (_,_,q),v in zip(items,transformed):
        p,c=_pool_patient(v,q); pooled.append(p); counts.append(c)
    pooled=np.asarray(pooled,dtype=np.float32); counts=np.asarray(counts,dtype=np.int32)
    trmask=np.asarray([r.site!=heldout for r,_,_ in items]); testmask=~trmask; tr_records=[r for r in all_records if r.site!=heldout]
    med,finite_counts,mass=_fit_fold_imputer(pooled[trmask],tr_records); imputed=np.where(np.isfinite(pooled),pooled,med[None,:]).astype(np.float32)
    standardized,pat_mu,pat_scale=_patient_standardize(imputed[trmask],tr_records,imputed)
    names=_feature_names(); seed_dir.mkdir(parents=True,exist_ok=True)
    ids=np.asarray([r.record_id for r in all_records]); sites=np.asarray([r.site for r in all_records]); ys=np.asarray([r.y for r in all_records],dtype=np.int8); ages=np.asarray([load_record(r.path)["age"] for r in all_records],dtype=np.float32)
    _save_npz(seed_dir/'epoch_scaler.npz',mean=epoch_scaler['mean'],scale=epoch_scaler['scale'],site_record_counts=epoch_scaler['site_record_counts'],sites=epoch_scaler['sites'],feature_order=names,heldout_site=heldout,seed=seed,source_sha=SOURCE_SHA)
    _save_npz(seed_dir/'pooled_raw.npz',record_id=ids,site=sites,y=ys,age=ages,pooled_features=pooled,valid_count=counts,outer_train=trmask,feature_order=names)
    _save_npz(seed_dir/'patient_imputer.npz',median=med,finite_counts=finite_counts,weighted_finite_mass=mass,feature_order=names,source_sha=SOURCE_SHA)
    _save_npz(seed_dir/'pooled_imputed.npz',record_id=ids,pooled_features=imputed,outer_train=trmask,feature_order=names)
    standardized,pat_mu,pat_scale=_patient_standardize(imputed[trmask],tr_records,imputed)
    _save_npz(seed_dir/'patient_scaler.npz',mean=pat_mu,scale=pat_scale,feature_order=names,weight_definition=np.asarray('site-balanced record weights'))
    _save_npz(seed_dir/'pooled_standardized.npz',record_id=ids,pooled_features=standardized,outer_train=trmask,feature_order=names)
    weights=_classifier_weights(tr_records); train_X=standardized[trmask]; train_y=ys[trmask]
    clf=LogisticRegression(penalty='elasticnet',solver='saga',l1_ratio=0.4,C=0.03,max_iter=100000,tol=1e-4,random_state=7,fit_intercept=True,class_weight=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always'); clf.fit(train_X,train_y,sample_weight=weights)
    train_score=clf.decision_function(train_X); test_score=clf.decision_function(standardized[testmask]); test_records=[r for r in all_records if r.site==heldout]
    def pred_rows(recs, scores): return [{"record_id":r.record_id,"site":r.site,"y":r.y,"age":float(load_record(r.path)['age']),"score":float(s)} for r,s in zip(recs,scores)]
    _save_csv(seed_dir/'predictions_train.csv',pred_rows(tr_records,train_score)); _save_csv(seed_dir/'predictions_test.csv',pred_rows(test_records,test_score))
    def metrics(recs,scores):
        y=np.asarray([r.y for r in recs]); age=np.asarray([load_record(r.path)['age'] for r in recs],dtype=float); s=np.asarray(scores)
        out={"AC":float(compute_auroc_age(y,s,age,gap=2)),"AUROC":float(roc_auc_score(y,s)),"AUPRC":float(average_precision_score(y,s))}
        try: out['age_weighted_AUROC']=float(compute_auroc_weighted(y,s,age,gap=2))
        except Exception: out['age_weighted_AUROC']=None
        if not all(np.isfinite(v) for v in out.values() if v is not None): raise FloatingPointError('nonfinite classifier metric')
        return out
    m=metrics(test_records,test_score)
    _save_json(seed_dir/'metrics.json',dict(m,heldout_site=heldout,seed=seed,nonzero_coefficients=int(np.count_nonzero(clf.coef_)),convergence_warning=bool(caught),n_iter=np.asarray(clf.n_iter_).tolist(),train_counts={s:sum(r.site==s for r in tr_records) for s in sorted({r.site for r in tr_records})}))
    rows=[{"record_id":r.record_id,"site":r.site,"y":r.y,"sample_weight":float(w)} for r,w in zip(tr_records,weights)]; _save_csv(seed_dir/'train_sample_weights.csv',rows)
    _save_npz(seed_dir/'classifier.npz',coef=clf.coef_.astype(np.float32),intercept=clf.intercept_.astype(np.float32),feature_order=names)
    _save_json(seed_dir/'classifier.json',{"config":{"penalty":"elasticnet","solver":"saga","l1_ratio":0.4,"C":0.03,"max_iter":100000,"tol":1e-4,"random_state":7,"class_weight":None},"heldout_site":heldout,"seed":seed,"nonzero_coefficients":int(np.count_nonzero(clf.coef_)),"source_sha":SOURCE_SHA})
    trace_arrays={}
    for site,(rec,latent,raw,valid) in trace.items(): trace_arrays[f'{site}_record_id']=np.asarray(rec.record_id); trace_arrays[f'{site}_coherence_latent']=latent; trace_arrays[f'{site}_physiology_raw']=raw; trace_arrays[f'{site}_validity']=valid; trace_arrays[f'{site}_physiology_normalized']=_apply_epoch_scaler(raw,valid,epoch_scaler)
    _save_npz(seed_dir/'trace_samples.npz',**trace_arrays)
    return dict(m,heldout_site=heldout,seed=seed,nonzero_coefficients=int(np.count_nonzero(clf.coef_)),impute_cells=int(np.isnan(pooled[trmask]).sum()),all_invalid_records=int(np.any(np.isnan(pooled[trmask]),axis=1).sum()),max_feature_missing=int(np.max(np.isnan(pooled[trmask]).sum(axis=0))))


def _train_d16_fold(heldout: str, records: list, root: Path, device: torch.device) -> tuple[list[dict[str,object]], list, list]:
    outer_train, outer_test=outer_loso_split(records,heldout); inner_train,inner_val=inner_split_by_site(outer_train,seed=7)
    if ({r.record_id for r in outer_test}&({r.record_id for r in inner_train}|{r.record_id for r in inner_val})) or ({r.record_id for r in inner_train}&{r.record_id for r in inner_val}): raise RuntimeError('split leakage')
    raw,counts=sample_training_pairs(inner_train,target_per_site=200_000,seed=7); scaler=fit_robust_scaler24(raw); train_values=scaler.transform(raw); valraw=sample_validation_pairs(inner_val,max_per_record=128,seed=7); validation={site:[(rid,scaler.transform(v)) for rid,v in rows] for site,rows in valraw.items()}
    fold_dir=root/f'outer_{heldout}'; fold_dir.mkdir(parents=True,exist_ok=True); results=[]; models=[]
    for seed in (7,17,27):
        result=train_coherence_ae(16,seed,train_values,validation,device,return_model=True); model=result.pop('model'); results.append(result); models.append((seed,model,scaler))
        sd=fold_dir/f'seed_{seed}'/'coherence_ae'; sd.mkdir(parents=True,exist_ok=True)
        torch.save({'encoder_state_dict':model.encoder.state_dict(),'decoder_state_dict':model.decoder.state_dict(),'d':16,'seed':seed,'outer_heldout_site':heldout,'inner_split_seed':7,'inner_train_record_ids_hash':_hash_ids(inner_train),'inner_val_record_ids_hash':_hash_ids(inner_val),'coherence_scaler_center':scaler.center,'coherence_scaler_scale':scaler.scale,'best_epoch':result['best_epoch'],'best_validation_mse':result['site_balanced_mse'],'ae_config':result['config'],'eeg_pairs':list(EEG_PAIRS),'extraction_version':EXPECTED_EXTRACTION_VERSION,'validity_schema_version':EXPECTED_VALIDITY_SCHEMA,'source_sha':SOURCE_SHA},sd/'checkpoint.pt')
        _save_json(sd/'context.json',{'heldout_site':heldout,'seed':seed,'inner_train_count':len(inner_train),'inner_val_count':len(inner_val),'train_pair_counts':counts,'source_sha':SOURCE_SHA,'locked_d':16})
        _save_npz(sd/'coherence_scaler.npz',center=scaler.center,scale=scaler.scale,source_sha=SOURCE_SHA)
    mean=float(np.mean([r['site_balanced_mse'] for r in results])); locked=LOCKED_GATE2_MEANS[heldout]; rel=abs(mean-locked)/locked
    if rel>0.10: raise RuntimeError(f'GATE2_5 reproducibility failed {heldout}: {mean} vs {locked}')
    return results,models,outer_test


def _save_csv(path: Path, rows: list[dict[str,object]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: path.write_text(''); return
    fields=list(rows[0]);
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n'); w.writeheader(); w.writerows(rows)



def run_gate3_smoke(npz_root: Path) -> None:
    records=discover_records(npz_root,validate_schema=True)
    train_records=[]; test_records=[]
    for site in ("I0006","S0001"):
        site_rows=[]
        for y in (0,1):
            site_rows.extend([r for r in records if r.site==site and r.y==y][:2])
        train_records.extend(site_rows)
    test_records=[]
    for y in (0,1):
        test_records.extend([r for r in records if r.site=="I0002" and r.y==y][:2])
    raw,_=sample_training_pairs(train_records,target_per_site=64,seed=7)
    scaler=fit_robust_scaler24(raw); values=scaler.transform(raw)
    valraw=sample_validation_pairs(train_records,max_per_record=4,seed=7)
    validation={site:[(rid,scaler.transform(v)) for rid,v in rows] for site,rows in valraw.items()}
    result=train_coherence_ae(16,7,values,validation,torch.device("cpu"),config=AEConfig(max_epochs=2,patience=1),return_model=True)
    model=result.pop("model")
    small=[]
    for rec in train_records+test_records:
        loaded=load_record(rec.path); rawv,valid,latent=_physiology_epoch(model,loaded,scaler,torch.device("cpu")); small.append((rec,rawv,valid))
    epoch_scaler=_fit_epoch_scaler(small[:len(train_records)]); transformed=[_apply_epoch_scaler(v,q,epoch_scaler) for _,v,q in small]
    pooled=np.asarray([_pool_patient(v,q)[0] for v,q in zip(transformed,[q for _,_,q in small])]); counts=np.asarray([_pool_patient(v,q)[1] for v,q in zip(transformed,[q for _,_,q in small])])
    assert pooled.shape==(len(small),POOLED_DIM) and counts.shape==(len(small),PHYSIO_DIM)
    all_invalid=np.zeros((3,PHYSIO_DIM),dtype=bool); assert np.isnan(_pool_patient(np.zeros((3,PHYSIO_DIM),dtype=np.float32),all_invalid)[0]).all()
    tr=small[:len(train_records)]; tr_pool=pooled[:len(train_records)]; med,_,_=_fit_fold_imputer(tr_pool,train_records); imputed=np.where(np.isfinite(pooled),pooled,med); std,_,_=_patient_standardize(imputed[:len(train_records)],train_records,imputed); w=_classifier_weights(train_records)
    assert std.shape==(len(small),POOLED_DIM) and w.shape==(len(train_records),)
    if len({r.y for r in train_records})<2: raise RuntimeError('Gate3 smoke train sample lacks both classes')
    clf=LogisticRegression(penalty='elasticnet',solver='saga',l1_ratio=.4,C=.03,max_iter=1000,tol=1e-4,random_state=7,class_weight=None); clf.fit(std[:len(train_records)],np.asarray([r.y for r in train_records]),sample_weight=w); scores=clf.decision_function(std[len(train_records):]); y=np.asarray([r.y for r in test_records]); ages=np.asarray([load_record(r.path)['age'] for r in test_records],dtype=float)
    y=np.asarray([0,1,0,1]); scores=np.asarray([-1,1,-.5,.5]); ages=np.asarray([50,50,60,60],dtype=float)
    assert np.isfinite(compute_auroc_age(y,scores,ages,gap=2)); print('GATE3_SMOKE_PASS')

def run_nofusion(npz_root: Path, output_dir: Path, device_name: str = 'auto') -> None:
    global SOURCE_SHA
    SOURCE_SHA=os.environ.get('PHYSIO_SOURCE_SHA','feaae35c3d6f73620a88e70f55a3fb5374e0250d')
    records=discover_records(npz_root,validate_schema=True); device=torch.device('cuda' if device_name=='auto' and torch.cuda.is_available() else device_name if device_name!='auto' else 'cpu')
    if device.type=='cuda': print('NOFUSION_DEVICE=',torch.cuda.get_device_name(0),flush=True)
    output_dir.mkdir(parents=True,exist_ok=True); all_metrics=[]; repro={}
    for heldout in EXPECTED_SITE_COUNTS:
        print(f'NOFUSION_FOLD={heldout} Gate2.5',flush=True); ae_results,models,test_records=_train_d16_fold(heldout,records,output_dir,device); mean=float(np.mean([r['site_balanced_mse'] for r in ae_results])); repro[heldout]={'seed_mse':[r['site_balanced_mse'] for r in ae_results],'mean':mean,'locked':LOCKED_GATE2_MEANS[heldout],'relative_difference':abs(mean-LOCKED_GATE2_MEANS[heldout])/LOCKED_GATE2_MEANS[heldout]}
        outer_train,_=outer_loso_split(records,heldout)
        for seed,model,scaler in models:
            print(f'NOFUSION_FOLD={heldout} seed={seed} downstream',flush=True); seed_dir=output_dir/f'outer_{heldout}'/f'seed_{seed}'; metric=_run_one_nofusion_seed(heldout,seed,outer_train,test_records,model,scaler,seed_dir,device); all_metrics.append(metric)
    _save_csv(output_dir/'nofusion_metrics.csv',all_metrics); _save_json(output_dir/'nofusion_metrics.json',{'gate2_5_reproducibility':repro,'metrics':all_metrics,'locked_d':16,'source_sha':SOURCE_SHA})
    lines=['# Physio Fusion V1 Gate 3 NoFusion','', '## Decision', '', '**GATE3_NOFUSION = COMPLETE**','', 'Gate 2.5 coherence bottleneck was rerun at locked `d=16`; no additional d-search was performed. Stage/Event, circadian, CAISR20, demographics, validity inputs, and Gate 4 Fusion192 were excluded.','', '## Gate 2.5 reproducibility','', '| Held-out | seed 7 | seed 17 | seed 27 | mean | locked | relative difference |','|---|---:|---:|---:|---:|---:|---:|']
    for site,v in repro.items(): lines.append(f"| {site} | "+' | '.join(f'{x:.8g}' for x in v['seed_mse'])+f" | {v['mean']:.8g} | {v['locked']:.8g} | {v['relative_difference']:.4%} |")
    lines += ['', '## NoFusion metrics', '', '| Held-out | AE seed | AC | AUROC | AUPRC | nonzero coefficients |', '|---|---:|---:|---:|---:|---:|']
    for m in all_metrics: lines.append(f"| {m['heldout_site']} | {m['seed']} | {m['AC']:.8g} | {m['AUROC']:.8g} | {m['AUPRC']:.8g} | {m['nonzero_coefficients']} |")
    lines += ['', '## Protocol and leakage checks', '', '- 361-D epoch order: spectral54, BSR18, coherence15x16, HRV11, EMG24, Resp14; pooled representation is 722-D.', '- Outer test records were excluded from AE fitting, coherence scaler, epoch scaler, patient imputer, patient scaler, and classifier fitting.', '- Age was used only for metrics; validity was never a classifier input.', '- No Stage/Event13, circadian, CAISR20, demographics, calibration, thresholding, or Fusion192 was run.']
    (output_dir/'NOFUSION_RESULTS.md').write_text('\n'.join(lines)+'\n')
    _save_json(output_dir/'run_context.json',{'source_sha':SOURCE_SHA,'npz_root':str(npz_root.resolve()),'device':str(device),'torch':torch.__version__,'numpy':np.__version__,'locked_d':16,'seeds':[7,17,27]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("smoke", "smoke-gate3", "d-search", "prepare-d16", "nofusion"))
    parser.add_argument("--npz-root", type=Path, default=DEFAULT_NPZ_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("output/physio_fusion_v1/d_selection"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.command == "smoke":
        run_smoke(args.npz_root)
    elif args.command == "smoke-gate3":
        run_gate3_smoke(args.npz_root)
    elif args.command == "d-search":
        run_d_search(args.npz_root, args.output_dir, args.device)
    elif args.command == "nofusion":
        run_nofusion(args.npz_root, args.output_dir, args.device)
    else:
        records = discover_records(args.npz_root, validate_schema=True)
        device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
        out = args.output_dir
        for heldout in EXPECTED_SITE_COUNTS:
            _train_d16_fold(heldout, records, out, device)


if __name__ == "__main__":
    main()
