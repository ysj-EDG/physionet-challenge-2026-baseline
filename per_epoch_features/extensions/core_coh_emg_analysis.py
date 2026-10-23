"""CORE_COH_EMG 2x2 analysis helpers and frozen-output orchestration."""
from __future__ import annotations

import math
from typing import Mapping


def build_core_2x2_row(*, protocol: str, scope: str, core: float, caisr_only: float, resp_only: float, both: float) -> dict[str, object]:
    values = {"core": float(core), "caisr_only": float(caisr_only), "resp_only": float(resp_only), "both": float(both)}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("all 2x2 cells must be finite")
    out = {"protocol": str(protocol), "scope": str(scope), **values}
    out.update({
        "caisr_effect_resp_absent": values["caisr_only"] - values["core"],
        "caisr_effect_resp_present": values["both"] - values["resp_only"],
        "resp_effect_caisr_absent": values["resp_only"] - values["core"],
        "resp_effect_caisr_present": values["both"] - values["caisr_only"],
        "interaction": values["both"] - values["caisr_only"] - values["resp_only"] + values["core"],
    })
    if not math.isclose(out["interaction"], out["caisr_effect_resp_present"] - out["caisr_effect_resp_absent"], rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("CAISR interaction closure failed")
    if not math.isclose(out["interaction"], out["resp_effect_caisr_present"] - out["resp_effect_caisr_absent"], rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Resp interaction closure failed")
    return out


def build_core_2x2_rows(metrics: Mapping[tuple[str, str], Mapping[str, float]]) -> list[dict[str, object]]:
    rows = []
    for (protocol, scope), cells in sorted(metrics.items()):
        required = {"CORE_COH_EMG", "LOO_NO_RESP", "LOO_NO_CAISR", "E2"}
        if set(cells) != required:
            raise ValueError("2x2 cell coverage mismatch")
        rows.append(build_core_2x2_row(protocol=protocol, scope=scope, core=cells["CORE_COH_EMG"], caisr_only=cells["LOO_NO_RESP"], resp_only=cells["LOO_NO_CAISR"], both=cells["E2"]))
    return rows


__all__ = ["build_core_2x2_row", "build_core_2x2_rows"]


# The orchestration below is deliberately post-hoc: it consumes frozen old
# references and the completed CORE prediction units only.
import csv
import json
from pathlib import Path
import numpy as np
from reproduction.siren_a.g5.core import HOLDOUTS, PROTOCOLS, MODEL_TYPES, VIEWS, load_eval_targets, mean_ranks, score_site
from .v5_family_loo_analysis import SEEDS, SCOPES, OFFICIAL, _rows_by_key, _old_prediction, _load_columns, _aggregate, _scope, load_protocol_references
from .e0e3_stability import pair_contributions, weighted_ac_batch, bootstrap_multiplicity
from .v5_family_loo_runner import validate_public_payload

CORE_COMPARISONS = ("CORE-E2", "CORE-E0", "CORE-LOO_NO_CAISR", "CORE-LOO_NO_RESP")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _baseline_fused(rows, holdout, protocol, seed=None):
    seeds = SEEDS if seed is None else (int(seed),)
    columns=[]; canonical=None; keys=[]
    for current_seed in seeds:
        for view in VIEWS:
            for model in MODEL_TYPES:
                scores, indices = _old_prediction(rows[(holdout, protocol, view, model, int(current_seed))])
                if canonical is None: canonical = indices.copy()
                elif not np.array_equal(canonical, indices): raise ValueError("baseline prediction order mismatch")
                columns.append(scores); keys.append((current_seed, view, model))
    matrix=np.column_stack(columns).astype(np.float64)
    if seed is None:
        if matrix.shape[1] != 225: raise ValueError("baseline column mismatch")
        _, fused=mean_ranks(matrix)
    else:
        if matrix.shape[1] != 3*len(VIEWS): raise ValueError("seed baseline column mismatch")
        _, fused=mean_ranks(matrix)
    return canonical, np.asarray(fused,dtype=np.float64), matrix, tuple(keys)


def _fused_groups(reference_manifest: Path, core_root: Path, loo_root: Path):
    old_e0=_rows_by_key(reference_manifest,"E0"); old_e2=_rows_by_key(reference_manifest,"E2")
    groups={}
    for protocol in PROTOCOLS:
        for holdout in HOLDOUTS:
            groups[("E0",protocol,holdout)] = _baseline_fused(old_e0,holdout,protocol)
            groups[("E2",protocol,holdout)] = _baseline_fused(old_e2,holdout,protocol)
            idx, fused, matrix, keys = _load_columns(old_e0, Path(loo_root), "LOO_NO_CAISR", holdout, protocol)
            groups[("LOO_NO_CAISR",protocol,holdout)] = (idx,fused,matrix,keys)
            idx, fused, matrix, keys = _load_columns(old_e0, Path(loo_root), "LOO_NO_RESP", holdout, protocol)
            groups[("LOO_NO_RESP",protocol,holdout)] = (idx,fused,matrix,keys)
            idx, fused, matrix, keys = _load_columns(old_e0, Path(core_root), "CORE_COH_EMG", holdout, protocol)
            groups[("CORE_COH_EMG",protocol,holdout)] = (idx,fused,matrix,keys)
    for key, value in groups.items():
        if value[2].shape[1] != 225: raise ValueError(f"fixed fusion width mismatch: {key}")
    return groups


def _site_rows(metadata, groups):
    rows=[]
    for (experiment, protocol, holdout), (indices, scores, _, _) in groups.items():
        labels, ages = load_eval_targets(metadata, indices)
        rows.append({"config_id":experiment,"protocol":protocol,"holdout_site":holdout,**score_site(labels,scores,ages,OFFICIAL)})
    return rows


def _private_reference_manifest(groups, path: Path):
    entries=[]
    for (experiment, protocol, holdout), (_, _, _, keys) in sorted(groups.items()):
        if experiment != "CORE_COH_EMG":
            continue
        for seed, view, model in keys:
            entries.append({"experiment":experiment,"holdout_site":holdout,"protocol":protocol,"seed":int(seed),"view":view,"model_type":model})
    if len(entries) != 6*225: raise ValueError("reference entry coverage mismatch")
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps({"schema":"featurelab_v2_core_coh_emg_reference_v1","entries":entries},sort_keys=True,separators=(",",":")),encoding="utf-8")
    return len(entries)


def _metrics_by_scope(site_rows):
    by_exp={}
    for row in site_rows: by_exp.setdefault((row["config_id"],row["protocol"]),[]).append(row)
    out={}
    for protocol in PROTOCOLS:
        for scope in SCOPES:
            cells={}
            for exp in ("CORE_COH_EMG","LOO_NO_RESP","LOO_NO_CAISR","E2","E0"):
                rows=by_exp[(exp,protocol)]
                vals={r["holdout_site"]:float(r["ac_auroc"]) for r in rows}
                cells[exp]=_scope(vals,scope)
            out[(protocol,scope)]=cells
    return out


def _seed_metrics(metadata, reference_manifest, core_root, loo_root):
    old_e0=_rows_by_key(reference_manifest,"E0"); old_e2=_rows_by_key(reference_manifest,"E2")
    out={}
    for seed in SEEDS:
        for protocol in PROTOCOLS:
            for holdout in HOLDOUTS:
                cells={}
                for exp, rows, root, config in (("E0",old_e0,None,None),("E2",old_e2,None,None)):
                    idx, fused, _, _ = _baseline_fused(rows,holdout,protocol,seed)
                    labels, ages=load_eval_targets(metadata,idx); cells[exp]=score_site(labels,fused,ages,OFFICIAL)["ac_auroc"]
                for exp, root, config in (("LOO_NO_CAISR",loo_root,"LOO_NO_CAISR"),("LOO_NO_RESP",loo_root,"LOO_NO_RESP"),("CORE_COH_EMG",core_root,"CORE_COH_EMG")):
                    idx, fused, _, _ = _load_columns(old_e0,Path(root),config,holdout,protocol,seed=seed)
                    labels, ages=load_eval_targets(metadata,idx); cells[exp]=score_site(labels,fused,ages,OFFICIAL)["ac_auroc"]
                out[(seed,protocol,holdout)]=cells
    return out


def _summarize_seed(values, *, experiment, comparison, protocol, scope):
    arr=np.asarray(values,dtype=np.float64)
    return {"config_id":experiment,"comparison":comparison,"protocol":protocol,"scope":scope,"n_seeds":len(arr),"n_positive":int((arr>0).sum()),"n_zero":int((arr==0).sum()),"n_negative":int((arr<0).sum()),"positive_fraction":float((arr>0).mean()),"mean_delta":float(arr.mean()),"median_delta":float(np.median(arr)),"q25":float(np.quantile(arr,.25)),"q75":float(np.quantile(arr,.75)),"min":float(arr.min()),"max":float(arr.max())}


def _bootstrap_rows(metadata, groups, n_bootstrap, bootstrap_seed):
    # One multiplicity matrix per site/protocol is shared by all four cells.
    rng=np.random.default_rng(int(bootstrap_seed)); draws={}
    observed={}
    for protocol in PROTOCOLS:
        for holdout in HOLDOUTS:
            vectors={}
            idx=groups[("CORE_COH_EMG",protocol,holdout)][0]
            labels, ages=load_eval_targets(metadata,idx)
            mult=np.vstack([bootstrap_multiplicity(len(labels),rng) for _ in range(int(n_bootstrap))])
            for exp in ("CORE_COH_EMG","LOO_NO_RESP","LOO_NO_CAISR","E2","E0"):
                score=groups[(exp,protocol,holdout)][1]
                vectors[exp]=weighted_ac_batch(pair_contributions(labels,ages,score),mult)
                observed[(exp,protocol,holdout)]=float(score_site(labels,score,ages,OFFICIAL)["ac_auroc"])
            draws[(protocol,holdout)] = vectors
    rows=[]
    comparisons=(("CORE-E2","CORE_COH_EMG","E2"),("CORE-E0","CORE_COH_EMG","E0"),("CORE-LOO_NO_CAISR","CORE_COH_EMG","LOO_NO_CAISR"),("CORE-LOO_NO_RESP","CORE_COH_EMG","LOO_NO_RESP"))
    for protocol in PROTOCOLS:
        for scope in SCOPES:
            for name,left,right in comparisons:
                if scope in HOLDOUTS: delta=draws[(protocol,scope)][left]-draws[(protocol,scope)][right]
                else:
                    arr=np.column_stack([draws[(protocol,h)][left]-draws[(protocol,h)][right] for h in HOLDOUTS])
                    delta=arr.mean(1) if scope=="Macro" else arr.min(1) if scope=="Worst-site" else arr[:,:2].mean(1)
                obs_values={exp:{h:observed[(exp,protocol,h)] for h in HOLDOUTS} for exp in ("CORE_COH_EMG","LOO_NO_RESP","LOO_NO_CAISR","E2","E0")}
                obs_delta={h:obs_values[left][h]-obs_values[right][h] for h in HOLDOUTS}
                observed_delta=_scope(obs_delta,scope)
                valid=np.isfinite(delta); vals=delta[valid]
                rows.append({"comparison":name,"protocol":protocol,"scope":scope,"observed_delta":float(observed_delta),"mean_delta":float(np.nanmean(delta)),"median_delta":float(np.nanmedian(delta)),"sd_delta":float(np.nanstd(delta,ddof=1)),"q025":float(np.nanquantile(delta,.025)),"q975":float(np.nanquantile(delta,.975)),"fraction_gt_0":float(np.mean(vals>0)) if len(vals) else float("nan"),"fraction_lt_0":float(np.mean(vals<0)) if len(vals) else float("nan"),"valid_replicates":int(valid.sum()),"invalid_replicates":int((~valid).sum()),"n_bootstrap":int(n_bootstrap),"bootstrap_seed":int(bootstrap_seed)})
            if scope in HOLDOUTS:
                delta=draws[(protocol,scope)]["E2"]-draws[(protocol,scope)]["LOO_NO_RESP"]-draws[(protocol,scope)]["LOO_NO_CAISR"]+draws[(protocol,scope)]["CORE_COH_EMG"]
            else:
                arr=np.column_stack([draws[(protocol,h)]["E2"]-draws[(protocol,h)]["LOO_NO_RESP"]-draws[(protocol,h)]["LOO_NO_CAISR"]+draws[(protocol,h)]["CORE_COH_EMG"] for h in HOLDOUTS])
                delta=arr.mean(1) if scope=="Macro" else arr.min(1) if scope=="Worst-site" else arr[:,:2].mean(1)
            obs_values={exp:{h:observed[(exp,protocol,h)] for h in HOLDOUTS} for exp in ("CORE_COH_EMG","LOO_NO_RESP","LOO_NO_CAISR","E2","E0")}
            obs_interaction={h:obs_values["E2"][h]-obs_values["LOO_NO_RESP"][h]-obs_values["LOO_NO_CAISR"][h]+obs_values["CORE_COH_EMG"][h] for h in HOLDOUTS}
            observed_delta=_scope(obs_interaction,scope)
            valid=np.isfinite(delta); vals=delta[valid]
            rows.append({"comparison":"CAISR_RESP_INTERACTION","protocol":protocol,"scope":scope,"observed_delta":float(observed_delta),"mean_delta":float(np.nanmean(delta)),"median_delta":float(np.nanmedian(delta)),"sd_delta":float(np.nanstd(delta,ddof=1)),"q025":float(np.nanquantile(delta,.025)),"q975":float(np.nanquantile(delta,.975)),"fraction_gt_0":float(np.mean(vals>0)) if len(vals) else float("nan"),"fraction_lt_0":float(np.mean(vals<0)) if len(vals) else float("nan"),"valid_replicates":int(valid.sum()),"invalid_replicates":int((~valid).sum()),"n_bootstrap":int(n_bootstrap),"bootstrap_seed":int(bootstrap_seed)})
    return rows


def run_core_analysis(*, metadata, reference_manifest: Path, core_root: Path, loo_root: Path, public_root: Path, private_root: Path, protocol_reference_path: Path, n_bootstrap: int=2000, bootstrap_seed: int=20261023):
    groups=_fused_groups(reference_manifest,core_root,loo_root)
    private_root=Path(private_root); private_root.mkdir(parents=True,exist_ok=True)
    _private_reference_manifest(groups,private_root/"reference_manifest.json")
    site_rows=_site_rows(metadata,groups)
    core_site=[row for row in site_rows if row["config_id"]=="CORE_COH_EMG"]
    refs=load_protocol_references(protocol_reference_path)
    summary=[]
    for protocol in PROTOCOLS:
        rows=[r for r in core_site if r["protocol"]==protocol]
        agg=_aggregate(rows); ref2=refs[("E2",protocol)]; ref0=refs[("E0",protocol)]
        summary.append({"config_id":"CORE_COH_EMG","protocol":protocol,**agg,"delta_vs_E2_macro":agg["macro_ac"]-ref2["macro_ac"],"delta_vs_E0_macro":agg["macro_ac"]-ref0["macro_ac"],"delta_vs_E2_worst":agg["worst_site_ac"]-ref2["worst_site_ac"],"delta_vs_E0_worst":agg["worst_site_ac"]-ref0["worst_site_ac"]})
    metrics=_metrics_by_scope(site_rows); two_metrics={(protocol,scope): {k:v for k,v in cells.items() if k in {"CORE_COH_EMG","LOO_NO_RESP","LOO_NO_CAISR","E2"}} for (protocol,scope),cells in metrics.items()}; two_by_two=build_core_2x2_rows(two_metrics)
    seed_metrics=_seed_metrics(metadata,reference_manifest,core_root,loo_root); stability=[]
    for protocol in PROTOCOLS:
        for scope in SCOPES:
            for exp,base,name in (("CORE_COH_EMG","E2","CORE-E2"),("CORE_COH_EMG","E0","CORE-E0"),("CORE_COH_EMG","LOO_NO_CAISR","CORE-LOO_NO_CAISR"),("CORE_COH_EMG","LOO_NO_RESP","CORE-LOO_NO_RESP")):
                vals=[]
                for seed in SEEDS:
                    cell={h:seed_metrics[(seed,protocol,h)] for h in HOLDOUTS}; vals.append(_scope({h:cell[h][exp]-cell[h][base] for h in HOLDOUTS},scope))
                stability.append(_summarize_seed(vals,experiment="CORE_COH_EMG",comparison=name,protocol=protocol,scope=scope))
            vals=[]
            for seed in SEEDS:
                cell={h:seed_metrics[(seed,protocol,h)] for h in HOLDOUTS}; vals.append(_scope({h:cell[h]["E2"]-cell[h]["LOO_NO_RESP"]-cell[h]["LOO_NO_CAISR"]+cell[h]["CORE_COH_EMG"] for h in HOLDOUTS},scope))
            stability.append(_summarize_seed(vals,experiment="CORE_COH_EMG",comparison="CAISR_RESP_INTERACTION",protocol=protocol,scope=scope))
    bootstrap=_bootstrap_rows(metadata,groups,n_bootstrap,bootstrap_seed)
    public_root=Path(public_root); public_root.mkdir(parents=True,exist_ok=True)
    _write_csv(public_root/"featurelab_v2_core_coh_emg_site_scores.csv",core_site)
    _write_csv(public_root/"featurelab_v2_core_coh_emg_protocol_summary.csv",summary)
    _write_csv(public_root/"featurelab_v2_core_coh_emg_seed_stability.csv",stability)
    _write_csv(public_root/"featurelab_v2_core_coh_emg_bootstrap.csv",bootstrap)
    _write_csv(public_root/"featurelab_v2_core_caisr_resp_2x2.csv",two_by_two)
    for rows in (core_site,summary,stability,bootstrap,two_by_two):
        for row in rows: validate_public_payload(row)
    return {"site_rows":len(core_site),"summary_rows":len(summary),"reference_entries":1350,"fused_groups":6,"stability_rows":len(stability),"bootstrap_rows":len(bootstrap),"two_by_two_rows":len(two_by_two)}

__all__ += ["CORE_COMPARISONS", "build_core_2x2_rows", "run_core_analysis"]
