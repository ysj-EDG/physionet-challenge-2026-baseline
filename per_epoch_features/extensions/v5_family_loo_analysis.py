"""Aggregate-only scoring and stability analysis for private V5 LOO outputs."""
from __future__ import annotations

import csv
import json
import importlib.util
from pathlib import Path
from typing import Mapping

import numpy as np

from reproduction.siren_a.g5.core import HOLDOUTS, PROTOCOLS, MODEL_TYPES, VIEWS, load_eval_targets, mean_ranks, score_site
from .compare_e0_e3_v2 import load_prediction_row
from .e0e3_stability import pair_contributions, weighted_ac_from_pairs, bootstrap_multiplicity
from .v5_family_loo import CONFIGS, load_v5_manifest, select_config, _sha

SEEDS = tuple(range(42, 57))
SCOPES = (*HOLDOUTS, "Macro", "Worst-site", "Small-site mean")


def _official_module():
    path=Path("/database/home/gaohaojie/workspace/python-example-2026-siren-a/evaluate_model.py")
    spec=importlib.util.spec_from_file_location("featurelab_v5_loo_official",path)
    if spec is None or spec.loader is None: raise ValueError("official scoring module unavailable")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    for name in ("compute_auroc_age","compute_auroc","compute_auprc"):
        if not hasattr(module,name): raise ValueError("official scoring function missing")
    return module


OFFICIAL = _official_module()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rows_by_key(reference_manifest: Path, experiment: str) -> dict[tuple[str, str, str, str, int], Mapping[str, object]]:
    payload = _read(reference_manifest)
    rows = [row for row in payload["entries"] if row.get("experiment") == experiment]
    out = {}
    for row in rows:
        key=(str(row["holdout_site"]),str(row["protocol"]),str(row["view"]),str(row["model_type"]),int(row["seed"]))
        if key in out: raise ValueError("duplicate frozen reference key")
        out[key]=row
    if len(out) != 1350: raise ValueError("reference coverage mismatch")
    return out


def _new_prediction(path: Path):
    with np.load(Path(path)/"payload.npz", allow_pickle=False) as archive:
        scores=np.asarray(archive["scores"],dtype=np.float64); indices=np.asarray(archive["test_global_indices"],dtype=np.int64)
    if scores.ndim != 1 or indices.ndim != 1 or len(scores)!=len(indices) or not np.isfinite(scores).all(): raise ValueError("invalid LOO prediction")
    return scores, indices


def _load_columns(old_rows, new_root: Path, config_id: str, holdout: str, protocol: str, *, seed: int | None = None):
    columns=[]; keys=[]; canonical=None
    selected_seeds=(int(seed),) if seed is not None else SEEDS
    for current_seed in selected_seeds:
        for view in VIEWS:
            for model in MODEL_TYPES:
                if view == "V5":
                    path=Path(new_root)/"predictions"/config_id/f"seed{current_seed}"/f"holdout_{holdout}"/protocol/model
                    scores,indices=_new_prediction(path)
                else:
                    row=old_rows[(holdout,protocol,view,model,int(current_seed))]
                    scores,indices=load_prediction_row(row)
                if canonical is None: canonical=indices.copy()
                elif not np.array_equal(canonical,indices): raise ValueError("prediction row order mismatch")
                columns.append(scores); keys.append((current_seed,view,model))
    matrix=np.column_stack(columns).astype(np.float64)
    if matrix.shape[1] not in (15,225): raise ValueError("fusion column count mismatch")
    _,fused=mean_ranks(matrix)
    return canonical, fused, matrix, tuple(keys)


def _metric_rows(metadata, groups, *, config_id):
    site_rows=[]
    for (protocol,holdout),(indices,scores) in groups.items():
        labels,ages=load_eval_targets(metadata,indices)
        metric=score_site(labels,scores,ages,OFFICIAL)
        site_rows.append({"config_id":config_id,"protocol":protocol,"holdout_site":holdout,**metric})
    return site_rows


def _aggregate(rows):
    by={r["holdout_site"]:float(r["ac_auroc"]) for r in rows}
    vals=np.asarray([by[s] for s in HOLDOUTS],dtype=np.float64)
    return {"macro_ac":float(vals.mean()),"worst_site_ac":float(vals.min()),"small_site_mean_ac":float((by["I0002"]+by["I0006"])/2)}


def run_analysis(*, metadata, reference_manifest: Path, private_root: Path, public_root: Path, manifest_path: Path, n_bootstrap: int = 2000, bootstrap_seed: int = 20261023):
    old_e0=_rows_by_key(reference_manifest,"E0"); old_e2=_rows_by_key(reference_manifest,"E2")
    public_root=Path(public_root); public_root.mkdir(parents=True,exist_ok=True)
    site_rows=[]; protocol_rows=[]; all_groups={}; baseline_groups={"E0":{},"E2":{}}
    for baseline_name, rows in (("E0",old_e0),("E2",old_e2)):
        for protocol in PROTOCOLS:
            for holdout in HOLDOUTS:
                idx,fused,_,_= _load_columns(rows,Path("/database/home/gaohaojie/workspace/feature-cache-extension/train_seeds43_56_v2"),"__none__",holdout,protocol) if False else (None,None,None,None)
                # Frozen E0/E2 groups use their reference rows; V5 in E2 is the existing v2 source.
                cols=[]; canonical=None
                for seed in SEEDS:
                    for view in VIEWS:
                        for model in MODEL_TYPES:
                            row=rows[(holdout,protocol,view,model,int(seed))]; scores,indices=load_prediction_row(row)
                            if canonical is None: canonical=indices.copy()
                            elif not np.array_equal(canonical,indices): raise ValueError("baseline index mismatch")
                            cols.append(scores)
                _,fused=mean_ranks(np.column_stack(cols).astype(np.float64)); baseline_groups[baseline_name][(protocol,holdout)]=(canonical,fused)
    for config_id in CONFIGS:
        groups={}
        for protocol in PROTOCOLS:
            for holdout in HOLDOUTS:
                idx,fused,_,_= _load_columns(old_e0,Path(private_root),config_id,holdout,protocol)
                groups[(protocol,holdout)]=(idx,fused)
        all_groups[config_id]=groups
        rows=_metric_rows(metadata,groups,config_id=config_id); site_rows.extend(rows)
        for protocol in PROTOCOLS:
            current=[r for r in rows if r["protocol"]==protocol]
            agg=_aggregate(current)
            base2=_aggregate([{"holdout_site":h,"ac_auroc":score_site(*load_eval_targets(metadata,baseline_groups["E2"][(protocol,h)][0]),baseline_groups["E2"][(protocol,h)][1],OFFICIAL)["ac_auroc"]} for h in HOLDOUTS])
            base0=_aggregate([{"holdout_site":h,"ac_auroc":score_site(*load_eval_targets(metadata,baseline_groups["E0"][(protocol,h)][0]),baseline_groups["E0"][(protocol,h)][1],OFFICIAL)["ac_auroc"]} for h in HOLDOUTS])
            protocol_rows.append({"config_id":config_id,"protocol":protocol,**agg,"delta_vs_E2_macro":agg["macro_ac"]-base2["macro_ac"],"delta_vs_E0_macro":agg["macro_ac"]-base0["macro_ac"],"delta_vs_E2_worst":agg["worst_site_ac"]-base2["worst_site_ac"],"delta_vs_E0_worst":agg["worst_site_ac"]-base0["worst_site_ac"]})
    # Seed stability and bootstrap use the same frozen row order, with aggregate rows only.
    stability=[]; bootstrap=[]
    baseline_seed_metrics={"E0":{},"E2":{}}
    for name,rows in (("E0",old_e0),("E2",old_e2)):
        for seed in SEEDS:
            for protocol in PROTOCOLS:
                for holdout in HOLDOUTS:
                    cols=[]; canonical=None
                    for view in VIEWS:
                        for model in MODEL_TYPES:
                            score,idx=load_prediction_row(rows[(holdout,protocol,view,model,int(seed))]); canonical=idx if canonical is None else canonical; cols.append(score)
                    _,fused=mean_ranks(np.column_stack(cols).astype(np.float64)); labels,ages=load_eval_targets(metadata,canonical)
                    baseline_seed_metrics[(name,seed,protocol,holdout)]=score_site(labels,fused,ages,OFFICIAL)["ac_auroc"]
    for config_id in CONFIGS:
        for seed in SEEDS:
            for protocol in PROTOCOLS:
                for holdout in HOLDOUTS:
                    idx,fused,_,_= _load_columns(old_e0,Path(private_root),config_id,holdout,protocol,seed=seed); labels,ages=load_eval_targets(metadata,idx)
                    all_groups.setdefault((config_id,seed,protocol,holdout),score_site(labels,fused,ages,OFFICIAL)["ac_auroc"])
        for baseline in ("E0","E2"):
            for protocol in PROTOCOLS:
                for scope in SCOPES:
                    vals=[]
                    for seed in SEEDS:
                        left={h:all_groups[(config_id,seed,protocol,h)] for h in HOLDOUTS}; right={h:baseline_seed_metrics[(baseline,seed,protocol,h)] for h in HOLDOUTS}
                        vals.append(_scope(left,scope)-_scope(right,scope))
                    v=np.asarray(vals); stability.append({"config_id":config_id,"comparison":f"{config_id}-{baseline}","protocol":protocol,"scope":scope,"n_seeds":15,"n_positive":int((v>0).sum()),"n_zero":int((v==0).sum()),"n_negative":int((v<0).sum()),"positive_fraction":float((v>0).mean()),"mean_delta":float(v.mean()),"median_delta":float(np.median(v)),"q25":float(np.quantile(v,.25)),"q75":float(np.quantile(v,.75)),"min":float(v.min()),"max":float(v.max())})
    # Bootstrap on site-level pairs, sharing each site's multiplicity between LOO and baseline.
    for config_id in CONFIGS:
        for baseline in ("E0","E2"):
            for protocol in PROTOCOLS:
                draws_by_site={}
                for holdout in HOLDOUTS:
                    idx,s=all_groups[config_id][(protocol,holdout)]; bidx,bs=baseline_groups[baseline][(protocol,holdout)]; labels,ages=load_eval_targets(metadata,idx); pairs_l=pair_contributions(labels,ages,s); pairs_b=pair_contributions(labels,ages,bs); rng=np.random.default_rng(bootstrap_seed); mult=np.vstack([bootstrap_multiplicity(len(labels),rng) for _ in range(n_bootstrap)]); draws_by_site[holdout]=(weighted_ac_from_pairs(pairs_l,np.ones(len(labels),dtype=np.int64)),weighted_ac_from_pairs(pairs_b,np.ones(len(labels),dtype=np.int64)),np.asarray([_weighted_batch(pairs_l,x)-_weighted_batch(pairs_b,x) for x in mult]))
                for scope in SCOPES:
                    if scope in HOLDOUTS: delta=draws_by_site[scope][2]
                    else:
                        site=np.column_stack([draws_by_site[h][2] for h in HOLDOUTS]); delta=site.mean(1) if scope=="Macro" else site.min(1) if scope=="Worst-site" else site[:,:2].mean(1)
                    bootstrap.append({"config_id":config_id,"comparison":f"{config_id}-{baseline}","protocol":protocol,"scope":scope,"n_bootstrap":int(n_bootstrap),"bootstrap_seed":int(bootstrap_seed),"invalid_replicates":int(np.sum(~np.isfinite(delta))),"mean_delta":float(np.nanmean(delta)),"median_delta":float(np.nanmedian(delta)),"q025":float(np.nanquantile(delta,.025)),"q975":float(np.nanquantile(delta,.975))})
    _write_csv(public_root/"featurelab_v2_v5_family_loo_site_scores.csv",site_rows)
    _write_csv(public_root/"featurelab_v2_v5_family_loo_protocol_summary.csv",protocol_rows)
    _write_csv(public_root/"featurelab_v2_v5_family_loo_seed_stability.csv",stability)
    _write_csv(public_root/"featurelab_v2_v5_family_loo_bootstrap.csv",bootstrap)
    return {"site_rows":len(site_rows),"protocol_rows":len(protocol_rows),"seed_rows":len(stability),"bootstrap_rows":len(bootstrap),"reference_entries":5400,"fused_groups":24}


def _weighted_batch(pairs, multiplicity):
    return weighted_ac_from_pairs(pairs,multiplicity)


def _scope(values, scope):
    if scope in HOLDOUTS: return float(values[scope])
    a=np.asarray([values[s] for s in HOLDOUTS],dtype=np.float64)
    return float(a.mean() if scope=="Macro" else a.min() if scope=="Worst-site" else a[:2].mean())


def _write_csv(path, rows):
    if not rows: raise ValueError("empty public rows")
    fields=list(rows[0].keys()); path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
