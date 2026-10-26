"""Read-only SIREN-A plus historical Matcha-B prediction-level fusion.

The module deliberately has no model, feature, or embedding code.  It loads
the two frozen prediction sources, proves their identities, applies the
pre-registered branch-rank interface, and writes aggregate public evidence
plus record-level private payloads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from reproduction.siren_a.g5.core import load_eval_targets, mean_ranks, score_site
from .e0e3_stability import bootstrap_multiplicity, pair_contributions, weighted_ac_batch

SITES = ("I0002", "I0006", "S0001")
EXPECTED_COUNTS = {"I0002": 319, "I0006": 1142, "S0001": 5139}
SEEDS = (20260806, 20260807, 20260808)
SCOPES = (*SITES, "Macro", "Worst-site", "Small-site mean")
A_EXPECTED_AC = {
    "I0002": 0.7423117709437964,
    "I0006": 0.7045470227006141,
    "S0001": 0.6618493939560138,
}
B_EXPECTED_AC = {
    "I0002": 0.7481442205726405,
    "I0006": 0.7393224315186642,
    "S0001": 0.6680414632808643,
}
B_EXPECTED_MACRO = 0.7185027051240563
RAW_LOGIT_COLUMNS = tuple(f"B_seed_{seed}_logit" for seed in SEEDS)
SOURCES = {
    "a_route": "DEDUP_FUSION",
    "b_route": "MATCHA_M2_HISTORICAL_INNER1TO4",
    "b_e1_sha256": "e9f135c973c0540970f34acbde89096c028eb3a95a0a601d85f21e8f83d36ac0",
}


class FusionStop(RuntimeError):
    """A frozen-source, identity, metric, or privacy gate failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_tie_percentile_rank(values: np.ndarray) -> np.ndarray:
    """Use SIREN's existing average-tie mean-rank implementation."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise FusionStop("rank input must be a nonempty finite vector")
    return np.asarray(mean_ranks(array[:, None])[1], dtype=np.float64)


def build_b_final(frame: pd.DataFrame) -> np.ndarray:
    missing = set(RAW_LOGIT_COLUMNS) - set(frame.columns)
    if missing:
        raise FusionStop(f"missing native B logits: {sorted(missing)}")
    matrix = frame.loc[:, RAW_LOGIT_COLUMNS].to_numpy(dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise FusionStop("B native logits must be finite")
    return matrix.mean(axis=1, dtype=np.float64)


def build_fusions(r_a: np.ndarray, r_b: np.ndarray) -> dict[str, np.ndarray]:
    a = np.asarray(r_a, dtype=np.float64)
    b = np.asarray(r_b, dtype=np.float64)
    if a.ndim != 1 or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise FusionStop("branch ranks must be aligned finite vectors")
    return {
        "AB60": (0.6 * a + 0.4 * b).astype(np.float64),
        "AB50": (0.5 * a + 0.5 * b).astype(np.float64),
    }


def _official_module():
    path = Path(__file__).resolve().parents[2] / "evaluate_model.py"
    spec = importlib.util.spec_from_file_location("siren_matcha_fusion_official", path)
    if spec is None or spec.loader is None:
        raise FusionStop(f"official scorer unavailable: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = ("compute_auroc_age", "compute_auroc", "compute_auprc")
    if any(not hasattr(module, name) for name in required):
        raise FusionStop("official scorer is incomplete")
    return module


OFFICIAL = _official_module()


def _metric(labels: np.ndarray, ages: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    result = score_site(np.asarray(labels), np.asarray(scores), np.asarray(ages), OFFICIAL)
    return {
        "n_records": int(result["n_records"]),
        "n_positive": int(result["n_positive"]),
        "ac_pair_count": int(result["ac_pair_count"]),
        "ac_auroc": float(result["ac_auroc"]),
        "auroc": float(result["auroc"]),
        "auprc": float(result["auprc"]),
    }


def join_sources(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Join by exact site/record identity, preserving A's frozen order."""
    key = ["site", "record_id"]
    for name, frame in (("A", a), ("B", b)):
        if not set(key).issubset(frame.columns):
            raise FusionStop(f"{name} identity columns missing")
        if frame.duplicated(key).any():
            raise FusionStop(f"{name} has duplicate identity keys")
    a_keys = set(map(tuple, a[key].itertuples(index=False, name=None)))
    b_keys = set(map(tuple, b[key].itertuples(index=False, name=None)))
    if a_keys != b_keys:
        raise FusionStop(f"A/B identity sets differ: A-only={len(a_keys - b_keys)}, B-only={len(b_keys - a_keys)}")
    b_index = b.set_index(key, verify_integrity=True)
    ordered_keys = pd.MultiIndex.from_frame(a[key])
    selected = b_index.loc[ordered_keys].reset_index(drop=True)
    if not np.array_equal(a["label"].to_numpy(), selected["label"].to_numpy()):
        raise FusionStop("A/B labels disagree after exact identity join")
    if not np.array_equal(a["age"].to_numpy(dtype=float), selected["age"].to_numpy(dtype=float)):
        raise FusionStop("A/B ages disagree after exact identity join")
    result = a.reset_index(drop=True).copy()
    for column in RAW_LOGIT_COLUMNS:
        result[column] = selected[column].to_numpy(dtype=np.float64)
    result["B_final"] = build_b_final(result)
    if not np.isfinite(result["B_final"].to_numpy()).all():
        raise FusionStop("B final contains nonfinite values")
    return result


def _load_metadata(demographics: Path, g2_root: Path):
    from reproduction.siren_a.pipeline.label_join import load_metadata_index

    return load_metadata_index(Path(demographics), Path(g2_root), expected_counts=EXPECTED_COUNTS)


def load_a_source(a_root: Path, demographics: Path, g2_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    metadata = _load_metadata(demographics, g2_root)
    rows = []
    source_files = []
    for site in SITES:
        path = Path(a_root) / "final" / f"holdout_{site}" / "payload.npz"
        if not path.is_file():
            raise FusionStop(f"A payload unavailable: {path}")
        source_files.append(path)
        with np.load(path, allow_pickle=False) as archive:
            required = {"test_global_indices", "CORE_NO_CAISR20_FINAL", "CAISR20_FINAL", "DEDUP_FUSION_FINAL"}
            if not required.issubset(archive.files):
                raise FusionStop(f"A payload keys incomplete: {path}")
            indices = np.asarray(archive["test_global_indices"], dtype=np.int64)
            core = np.asarray(archive["CORE_NO_CAISR20_FINAL"], dtype=np.float64)
            caisr = np.asarray(archive["CAISR20_FINAL"], dtype=np.float64)
            direct = np.asarray(archive["DEDUP_FUSION_FINAL"], dtype=np.float64)
        if len(np.unique(indices)) != len(indices) or not (np.isfinite(core).all() and np.isfinite(caisr).all() and np.isfinite(direct).all()):
            raise FusionStop(f"A payload identity/finite gate failed: {path}")
        replay = 0.5 * core + 0.5 * caisr
        if not np.array_equal(replay, direct):
            raise FusionStop(f"A DEDUP replay differs from frozen payload: {site}")
        labels, ages = load_eval_targets(metadata, indices)
        if not np.all(metadata.sites[indices].astype(str) == site):
            raise FusionStop(f"A payload crosses holdout site: {site}")
        for index, label, age, a_score in zip(indices, labels, ages, replay):
            record = metadata.records[int(index)]
            rows.append({
                "global_index": int(index), "site": site,
                "record_id": record.record_stem, "label": int(label), "age": float(age),
                "A_final": float(a_score),
            })
    frame = pd.DataFrame(rows)
    if len(frame) != sum(EXPECTED_COUNTS.values()) or frame.duplicated(["site", "record_id"]).any():
        raise FusionStop("A coverage or identity gate failed")
    metrics = {site: _metric(
        frame.loc[frame.site == site, "label"].to_numpy(),
        frame.loc[frame.site == site, "age"].to_numpy(),
        frame.loc[frame.site == site, "A_final"].to_numpy(),
    ) for site in SITES}
    diffs = [abs(metrics[site]["ac_auroc"] - A_EXPECTED_AC[site]) for site in SITES]
    if max(diffs) > 1e-12:
        raise FusionStop(f"A parent replay metric drift exceeds 1e-12: {max(diffs):.3e}")
    return frame, {
        "gate": "PASS", "route": SOURCES["a_route"], "metrics": metrics,
        "max_metric_diff": float(max(diffs)), "files": [_file_meta(path) for path in source_files],
        "metadata_source": _file_meta(Path(demographics)),
        "metadata_index_source": _file_meta(Path(g2_root) / "inventory.json"),
    }


def load_b_source(parquet_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    path = Path(parquet_path)
    if not path.is_file():
        raise FusionStop(f"B prediction artifact unavailable: {path}")
    raw = pd.read_parquet(path)
    required = {"record_id", "SiteID", "_label", "_age", *[f"m2_logit_{seed}" for seed in SEEDS]}
    if not required.issubset(raw.columns):
        raise FusionStop(f"B prediction columns incomplete: {sorted(required - set(raw.columns))}")
    frame = raw.rename(columns={"SiteID": "site", "_label": "label", "_age": "age"}).copy()
    frame = frame[["site", "record_id", "label", "age", *[f"m2_logit_{seed}" for seed in SEEDS]]]
    frame = frame.rename(columns={f"m2_logit_{seed}": f"B_seed_{seed}_logit" for seed in SEEDS})
    if len(frame) != sum(EXPECTED_COUNTS.values()) or frame.duplicated(["site", "record_id"]).any():
        raise FusionStop("B coverage or identity gate failed")
    if frame.groupby("site").size().to_dict() != EXPECTED_COUNTS:
        raise FusionStop("B site counts do not match frozen A cohorts")
    frame["label"] = frame["label"].astype(np.int8)
    frame["age"] = frame["age"].astype(float)
    for column in RAW_LOGIT_COLUMNS:
        frame[column] = frame[column].astype(float)
    if not np.isfinite(frame[list(RAW_LOGIT_COLUMNS)].to_numpy()).all():
        raise FusionStop("B native logits contain nonfinite values")
    frame["B_final"] = build_b_final(frame)
    metrics = {site: _metric(
        frame.loc[frame.site == site, "label"].to_numpy(),
        frame.loc[frame.site == site, "age"].to_numpy(),
        frame.loc[frame.site == site, "B_final"].to_numpy(),
    ) for site in SITES}
    diffs = [abs(metrics[site]["ac_auroc"] - B_EXPECTED_AC[site]) for site in SITES]
    macro = float(np.mean([metrics[site]["ac_auroc"] for site in SITES]))
    if max(diffs) > 1e-12 or abs(macro - B_EXPECTED_MACRO) > 1e-12:
        raise FusionStop(f"B parent replay drift: max_site={max(diffs):.3e}, macro={macro:.16f}")
    return frame, {
        "gate": "PASS", "route": SOURCES["b_route"], "metrics": metrics,
        "macro_ac": macro, "max_metric_diff": float(max(diffs)), "file": _file_meta(path),
        "raw_logit_source": "native",
    }


def _file_meta(path: Path) -> dict[str, object]:
    path = Path(path)
    stat = path.stat()
    return {"absolute_source_path": str(path.resolve()), "bytes": int(stat.st_size), "sha256": sha256_file(path)}


def _scope(values: Mapping[str, float], name: str) -> float:
    ordered = np.asarray([float(values[site]) for site in SITES], dtype=np.float64)
    if name in SITES:
        return float(values[name])
    if name == "Macro":
        return float(ordered.mean())
    if name == "Worst-site":
        return float(ordered.min())
    if name == "Small-site mean":
        return float(ordered[:2].mean())
    raise FusionStop(f"unknown scope: {name}")


def _correlations(a: np.ndarray, b: np.ndarray, r_a: np.ndarray, r_b: np.ndarray) -> dict[str, float]:
    def corr(x: np.ndarray, y: np.ndarray) -> float:
        if np.std(x) == 0 or np.std(y) == 0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])
    return {"spearman_a_b": corr(r_a, r_b), "spearman_rank_a_b": corr(r_a, r_b), "pearson_rank_a_b": corr(r_a, r_b)}


def _pair_diagnostics(labels: np.ndarray, ages: np.ndarray, a: np.ndarray, b: np.ndarray, ab60: np.ndarray, ab50: np.ndarray) -> dict[str, object]:
    pa, pb = pair_contributions(labels, ages, a), pair_contributions(labels, ages, b)
    pf60, pf50 = pair_contributions(labels, ages, ab60), pair_contributions(labels, ages, ab50)
    if not (np.array_equal(pa["positive"], pb["positive"]) and np.array_equal(pa["negative"], pb["negative"])):
        raise FusionStop("A/B pair definitions differ")
    ca, cb = pa["contribution"], pb["contribution"]
    tie = (ca == 0.5) | (cb == 0.5)
    strict = ~tie
    both_correct = int(np.sum((ca == 1) & (cb == 1) & strict))
    a_only = int(np.sum((ca == 1) & (cb == 0) & strict))
    b_only = int(np.sum((ca == 0) & (cb == 1) & strict))
    both_wrong = int(np.sum((ca == 0) & (cb == 0) & strict))
    total = len(ca)
    def fraction(count: int) -> float:
        return float(count / total) if total else float("nan")
    def flips(parent: np.ndarray, fused: np.ndarray) -> tuple[int, int]:
        return int(np.sum((parent < 0.5) & (fused >= 0.5))), int(np.sum((parent >= 0.5) & (fused < 0.5)))
    a_to_60 = flips(ca, pf60["contribution"])
    b_to_60 = flips(cb, pf60["contribution"])
    a_to_50 = flips(ca, pf50["contribution"])
    b_to_50 = flips(cb, pf50["contribution"])
    return {
        "legal_pair_count": int(total), "both_correct": both_correct, "a_only_correct": a_only,
        "b_only_correct": b_only, "both_wrong": both_wrong, "tie_involved": int(np.sum(tie)),
        "a_wrong_b_correct_fraction": fraction(b_only), "b_wrong_a_correct_fraction": fraction(a_only),
        "net_complementary_pair_advantage": int(b_only - a_only),
        "ab60_a_wrong_to_correct": a_to_60[0], "ab60_a_correct_to_wrong": a_to_60[1],
        "ab60_b_wrong_to_correct": b_to_60[0], "ab60_b_correct_to_wrong": b_to_60[1],
        "ab50_a_wrong_to_correct": a_to_50[0], "ab50_a_correct_to_wrong": a_to_50[1],
        "ab50_b_wrong_to_correct": b_to_50[0], "ab50_b_correct_to_wrong": b_to_50[1],
    }


def paired_bootstrap_delta(labels: np.ndarray, ages: np.ndarray, left: np.ndarray, right: np.ndarray, *, n_bootstrap: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(int(seed))
    multiplicities = np.vstack([bootstrap_multiplicity(len(labels), rng) for _ in range(int(n_bootstrap))])
    left_pairs = pair_contributions(labels, ages, left)
    right_pairs = pair_contributions(labels, ages, right)
    left_draws = weighted_ac_batch(left_pairs, multiplicities)
    right_draws = weighted_ac_batch(right_pairs, multiplicities)
    delta = left_draws - right_draws
    valid = np.isfinite(delta)
    return {"draws": delta[valid], "valid": int(valid.sum()), "invalid": int((~valid).sum())}



def _write_bytes_no_clobber(path: Path, payload: bytes) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload: raise FusionStop(f"existing output differs: {path}")
        return
    path.write_bytes(payload)


def _write_json_no_clobber(path: Path, value: object) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode()
    if Path(path).exists():
        try:
            old = json.loads(Path(path).read_text(encoding="utf-8")); new = json.loads(payload.decode())
            if isinstance(old, dict) and isinstance(new, dict) and old.get("schema") == new.get("schema"):
                old.pop("generated_at_utc", None); new.pop("generated_at_utc", None)
                if old == new: return
        except (OSError, ValueError, TypeError):
            pass
    _write_bytes_no_clobber(path, payload)


def _write_csv_no_clobber(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows: raise FusionStop(f"empty public output: {path}")
    fields=[]
    for row in rows:
        for field in row:
            if field not in fields: fields.append(field)
    import io
    buffer=io.StringIO(newline="")
    writer=csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    _write_bytes_no_clobber(path, buffer.getvalue().encode())


def _private_payload(path: Path, frame: pd.DataFrame) -> str:
    fields=["global_index","record_id","label","age","A_final","rA",*RAW_LOGIT_COLUMNS,"B_final","rB","AB60","AB50"]
    import io
    arrays={}
    for field in fields:
        value=frame[field].to_numpy()
        if value.dtype.kind == "O": value=value.astype("U")
        arrays[field]=value
    buffer=io.BytesIO(); np.savez(buffer, **arrays)
    payload=buffer.getvalue(); _write_bytes_no_clobber(path,payload)
    return hashlib.sha256(payload).hexdigest()


def _private_snapshots(private_root: Path, a: pd.DataFrame, b: pd.DataFrame, source_meta: dict[str, object]) -> dict[str, object]:
    source_a=private_root/"source_a"/"a_prediction_source.csv"; source_b=private_root/"source_b"/"b_prediction_source.csv"
    _write_bytes_no_clobber(source_a,a[["global_index","site","record_id","label","age","A_final"]].to_csv(index=False,lineterminator="\n").encode())
    _write_bytes_no_clobber(source_b,b[["site","record_id","label","age",*RAW_LOGIT_COLUMNS,"B_final"]].to_csv(index=False,lineterminator="\n").encode())
    snapshot={"schema":"siren_matcha_ab_private_source_snapshot_v1","generated_at_utc":pd.Timestamp.utcnow().isoformat(),"source_git_branch":"research/featurelab-extractors-v2","source_git_ref":"626ec1e9f178ae5922db09fab79955a6182e0f67","source_model_id":"A_DEDUP_FUSION_and_B_M2_HISTORICAL_INNER1TO4","source_files":source_meta,"snapshot_files":{"source_a":{"bytes":source_a.stat().st_size,"sha256":sha256_file(source_a)},"source_b":{"bytes":source_b.stat().st_size,"sha256":sha256_file(source_b)}}}
    _write_json_no_clobber(private_root/"manifests"/"private_source_manifest.json",snapshot)
    return snapshot


def _correlations(a,b,r_a,r_b):
    def corr(x,y):
        if np.std(x)==0 or np.std(y)==0: return float("nan")
        return float(np.corrcoef(x,y)[0,1])
    return {"spearman_a_b":corr(r_a,r_b),"spearman_rank_a_b":corr(r_a,r_b),"pearson_rank_a_b":corr(r_a,r_b)}


def _pair_diagnostics(labels,ages,a,b,ab60,ab50):
    pa,pb=pair_contributions(labels,ages,a),pair_contributions(labels,ages,b)
    pf60,pf50=pair_contributions(labels,ages,ab60),pair_contributions(labels,ages,ab50)
    if not (np.array_equal(pa["positive"],pb["positive"]) and np.array_equal(pa["negative"],pb["negative"])): raise FusionStop("A/B pair definitions differ")
    ca,cb=pa["contribution"],pb["contribution"]; tie=(ca==.5)|(cb==.5); strict=~tie; total=len(ca)
    both_correct=int(np.sum((ca==1)&(cb==1)&strict)); a_only=int(np.sum((ca==1)&(cb==0)&strict)); b_only=int(np.sum((ca==0)&(cb==1)&strict)); both_wrong=int(np.sum((ca==0)&(cb==0)&strict))
    def frac(value): return float(value/total) if total else float("nan")
    def flips(parent,fused): return int(np.sum((parent<.5)&(fused>=.5))),int(np.sum((parent>=.5)&(fused<.5)))
    a60,b60=flips(ca,pf60["contribution"]),flips(cb,pf60["contribution"]); a50,b50=flips(ca,pf50["contribution"]),flips(cb,pf50["contribution"])
    return {"legal_pair_count":int(total),"both_correct":both_correct,"a_only_correct":a_only,"b_only_correct":b_only,"both_wrong":both_wrong,"tie_involved":int(np.sum(tie)),"a_wrong_b_correct_fraction":frac(b_only),"b_wrong_a_correct_fraction":frac(a_only),"net_complementary_pair_advantage":int(b_only-a_only),"ab60_a_wrong_to_correct":a60[0],"ab60_a_correct_to_wrong":a60[1],"ab60_b_wrong_to_correct":b60[0],"ab60_b_correct_to_wrong":b60[1],"ab50_a_wrong_to_correct":a50[0],"ab50_a_correct_to_wrong":a50[1],"ab50_b_wrong_to_correct":b50[0],"ab50_b_correct_to_wrong":b50[1]}


def _route_rows(joined):
    site_rows=[]; correlations=[]; pair_rows=[]; private_frames={}
    for site in SITES:
        subset=joined.loc[joined.site.to_numpy()==site].copy().reset_index(drop=True)
        a=subset.A_final.to_numpy(float); b=subset.B_final.to_numpy(float); r_a=average_tie_percentile_rank(a); r_b=average_tie_percentile_rank(b); fusions=build_fusions(r_a,r_b)
        subset["rA"],subset["rB"]=r_a,r_b; subset["AB60"],subset["AB50"]=fusions["AB60"],fusions["AB50"]
        private_frames[site]={c:subset[c].to_numpy() for c in subset.columns}
        routes={"A_FINAL":a,"B_MATCHA_M2":b,"AB_SIREN_60_40":fusions["AB60"],"AB_EQUAL_50_50":fusions["AB50"]}
        labels,ages=subset.label.to_numpy(np.int8),subset.age.to_numpy(float); raw_a,raw_b=_metric(labels,ages,a),_metric(labels,ages,b); rank_a,rank_b=_metric(labels,ages,r_a),_metric(labels,ages,r_b)
        if max(abs(raw_a[f]-rank_a[f]) for f in ("ac_auroc","auroc","auprc"))>1e-12: raise FusionStop(f"A rank metric invariance failed: {site}")
        if max(abs(raw_b[f]-rank_b[f]) for f in ("ac_auroc","auroc","auprc"))>1e-12: raise FusionStop(f"B rank metric invariance failed: {site}")
        for route,scores in routes.items(): site_rows.append({"route":route,"site":site,**_metric(labels,ages,scores)})
        correlations.append({"site":site,**_correlations(a,b,r_a,r_b),"n_records":int(len(subset))}); pair_rows.append({"site":site,**_pair_diagnostics(labels,ages,a,b,fusions["AB60"],fusions["AB50"])})
    return site_rows,correlations,pair_rows,private_frames


def _bootstrap_rows(private_frames, n_bootstrap, seed):
    routes=("A_FINAL","B_MATCHA_M2","AB_SIREN_60_40","AB_EQUAL_50_50"); comparisons=(("AB60_MINUS_A","AB_SIREN_60_40","A_FINAL"),("AB60_MINUS_B","AB_SIREN_60_40","B_MATCHA_M2"),("AB50_MINUS_A","AB_EQUAL_50_50","A_FINAL"),("AB50_MINUS_B","AB_EQUAL_50_50","B_MATCHA_M2")); rng=np.random.default_rng(int(seed)); draws={}; observed={}
    for site in SITES:
        frame=private_frames[site]; labels,ages=frame["label"],frame["age"]; mult=np.vstack([bootstrap_multiplicity(len(labels),rng) for _ in range(int(n_bootstrap))]); score_map={"A_FINAL":frame["A_final"],"B_MATCHA_M2":frame["B_final"],"AB_SIREN_60_40":frame["AB60"],"AB_EQUAL_50_50":frame["AB50"]}; observed[site]={route:float(_metric(labels,ages,score_map[route])["ac_auroc"]) for route in routes}; draws[site]={route:weighted_ac_batch(pair_contributions(labels,ages,score_map[route]),mult) for route in routes}
    rows=[]
    for comparison,left,right in comparisons:
        for scope in SCOPES:
            columns=np.column_stack([draws[site][left]-draws[site][right] for site in SITES]); delta=columns[:,SITES.index(scope)] if scope in SITES else columns.mean(1) if scope=="Macro" else columns.min(1) if scope=="Worst-site" else columns[:,:2].mean(1); valid=np.isfinite(delta); values=delta[valid]
            if len(values)!=int(n_bootstrap): raise FusionStop(f"invalid paired bootstrap replicates: {comparison}/{scope}")
            rows.append({"comparison":comparison,"scope":scope,"n_bootstrap":int(n_bootstrap),"seed":int(seed),"observed_delta":float(_scope({site: observed[site][left]-observed[site][right] for site in SITES},scope)),"bootstrap_mean":float(values.mean()),"median":float(np.median(values)),"sd":float(values.std(ddof=1)),"q025":float(np.quantile(values,.025)),"q975":float(np.quantile(values,.975)),"fraction_gt_0":float(np.mean(values>0)),"fraction_lt_0":float(np.mean(values<0)),"valid":int(len(values)),"invalid":int((~valid).sum())})
    return rows


def _summary_rows(site_rows):
    routes=("A_FINAL","B_MATCHA_M2","AB_SIREN_60_40","AB_EQUAL_50_50"); by={route:{row["site"]:float(row["ac_auroc"]) for row in site_rows if row["route"]==route} for route in routes}; return [{"route":route,"scope":scope,"ac_auroc":_scope(by[route],scope)} for route in routes for scope in SCOPES]


def _public_manifest(site_rows,summary_rows,correlation_rows,pair_rows,bootstrap_rows,private_payload_hashes,source_meta,git_info):
    return {"schema":"siren_matcha_ab_fusion_public_manifest_v1","branch":"research/siren-matcha-ab-fusion-v1","base_ref":"626ec1e9f178ae5922db09fab79955a6182e0f67","source_a":{"branch":"research/featurelab-extractors-v2","git_ref":"626ec1e9f178ae5922db09fab79955a6182e0f67","route":SOURCES["a_route"]},"source_b":{"route":SOURCES["b_route"],"e1_sha256":SOURCES["b_e1_sha256"],"seeds":list(SEEDS),"raw_logit_source":"native"},"weights":{"AB_SIREN_60_40":[.6,.4],"AB_EQUAL_50_50":[.5,.5]},"counts":{"sites":EXPECTED_COUNTS,"total":sum(EXPECTED_COUNTS.values())},"gate_status":{"A_PARENT_REPLAY":"PASS","B_PARENT_REPLAY":"PASS","A_B_RECORD_SET_IDENTITY":"PASS","A_B_RECORD_ORDER_CANONICALIZED":"PASS","A_RANK_METRIC_INVARIANCE":"PASS","B_RANK_METRIC_INVARIANCE":"PASS","NO_PARENT_MUTATION":"PASS"},"public_row_counts":{"site_scores":len(site_rows),"summary":len(summary_rows),"correlation":len(correlation_rows),"pair_complementarity":len(pair_rows),"bootstrap":len(bootstrap_rows)},"private_payload_sha256":private_payload_hashes,"source_input_hashes":source_meta,"git":git_info}


def _git_info(repo: Path):
    def run(*args): return subprocess.check_output(["git","-C",str(repo),*args],text=True).strip()
    return {"branch":run("branch","--show-current"),"head":run("rev-parse","HEAD")}


def _write_report(path,site_rows,summary_rows,correlation_rows,pair_rows,bootstrap_rows,a_info,b_info):
    summary={(row["route"],row["scope"]):row["ac_auroc"] for row in summary_rows}; lines=["# SIREN + Matcha B prediction-level fusion report","","This report uses only two frozen prediction branches. No model, feature, embedding, or raw-signal computation was run.","","## Gates","",f"- A parent replay: PASS; max AC difference: `{a_info['max_metric_diff']:.17g}`",f"- B parent replay: PASS; historical M2 inner-1-to-4 Macro AC: `{b_info['macro_ac']:.17g}`","- Exact identity bridge and A-order canonicalization: PASS; 6,600 rows retained.","- Rank metric invariance for A and B: PASS; no post-fusion rerank.","- Parent mutation check: PASS.","","## Primary route summary","","| Route | I0002 AC | I0006 AC | S0001 AC | Macro AC | Worst-site | Small-site mean |","|---|---:|---:|---:|---:|---:|---:|"]
    for route in ("A_FINAL","B_MATCHA_M2","AB_SIREN_60_40","AB_EQUAL_50_50"):
        vals=[summary[(route,scope)] for scope in ("I0002","I0006","S0001","Macro","Worst-site","Small-site mean")]; lines.append("| %s | %s |"%(route," | ".join(f"{v:.12f}" for v in vals)))
    lines += ["","## A/B correlation","","| Site | Spearman(A,B) | Spearman(rA,rB) | Pearson(rA,rB) |","|---|---:|---:|---:|"]
    for row in correlation_rows: lines.append("| %s | %.12f | %.12f | %.12f |"%(row["site"],row["spearman_a_b"],row["spearman_rank_a_b"],row["pearson_rank_a_b"]))
    lines += ["","## Pair complementarity","","| Site | Legal pairs | A-only correct | B-only correct | Both correct | Both wrong | B-correct/A-wrong fraction | A-correct/B-wrong fraction |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in pair_rows: lines.append("| %s | %d | %d | %d | %d | %d | %.6f | %.6f |"%(row["site"],row["legal_pair_count"],row["a_only_correct"],row["b_only_correct"],row["both_correct"],row["both_wrong"],row["a_wrong_b_correct_fraction"],row["b_wrong_a_correct_fraction"]))
    lines += ["","## Paired bootstrap","","| Comparison | Scope | Observed delta | 95% CI | P(delta>0) |","|---|---|---:|---:|---:|"]
    for row in bootstrap_rows: lines.append("| %s | %s | %.8f | [%.8f, %.8f] | %.4f |"%(row["comparison"],row["scope"],row["observed_delta"],row["q025"],row["q975"],row["fraction_gt_0"]))
    lines += ["","The primary route is `AB_SIREN_60_40`; `AB_EQUAL_50_50` is a fixed control, not a weight search.","","Record-level vectors and identity fields are stored only in the private artifact root and are not included in this branch.",""]
    _write_bytes_no_clobber(path,"\n".join(lines).encode())


def run_experiment(*,a_root:Path,demographics:Path,g2_root:Path,b_parquet:Path,b_report:Path,private_root:Path,repo_root:Path,n_bootstrap:int=2000,bootstrap_seed:int=20261023):
    source_before={"a":[_file_meta(Path(a_root)/"final"/f"holdout_{site}"/"payload.npz") for site in SITES],"b":_file_meta(b_parquet),"b_report":_file_meta(b_report)}
    a,a_info=load_a_source(a_root,demographics,g2_root); b,b_info=load_b_source(b_parquet); joined=join_sources(a,b)
    if joined.groupby("site").size().to_dict()!=EXPECTED_COUNTS or len(joined)!=6600: raise FusionStop("exact bridge site counts failed")
    _private_snapshots(Path(private_root),a,b,{"a":source_before["a"],"b":source_before["b"],"b_report":source_before["b_report"]})
    site_rows,correlations,pair_rows,private_frames=_route_rows(joined); summary_rows=_summary_rows(site_rows); bootstrap_rows=_bootstrap_rows(private_frames,n_bootstrap,bootstrap_seed)
    private_payload_hashes={}
    for site in SITES: private_payload_hashes[site]=_private_payload(Path(private_root)/"fused"/f"holdout_{site}"/"payload.npz",pd.DataFrame(private_frames[site]))
    source_after={"a":[_file_meta(Path(a_root)/"final"/f"holdout_{site}"/"payload.npz") for site in SITES],"b":_file_meta(b_parquet),"b_report":_file_meta(b_report)}
    if source_before!=source_after: raise FusionStop("NO_PARENT_MUTATION failed")
    git_info=_git_info(Path(repo_root)); public_root=Path(repo_root)/"audit"; public_meta={"a":[{"bytes":x["bytes"],"sha256":x["sha256"]} for x in source_before["a"]],"b":{"bytes":source_before["b"]["bytes"],"sha256":source_before["b"]["sha256"]},"b_report":{"bytes":source_before["b_report"]["bytes"],"sha256":source_before["b_report"]["sha256"]}}
    manifest=_public_manifest(site_rows,summary_rows,correlations,pair_rows,bootstrap_rows,private_payload_hashes,public_meta,git_info)
    _write_csv_no_clobber(public_root/"siren_matcha_ab_site_scores.csv",site_rows); _write_csv_no_clobber(public_root/"siren_matcha_ab_summary.csv",summary_rows); _write_csv_no_clobber(public_root/"siren_matcha_ab_correlation.csv",correlations); _write_csv_no_clobber(public_root/"siren_matcha_ab_pair_complementarity.csv",pair_rows); _write_csv_no_clobber(public_root/"siren_matcha_ab_bootstrap.csv",bootstrap_rows); _write_json_no_clobber(public_root/"siren_matcha_ab_manifest.json",manifest); _write_report(Path(repo_root)/"docs/fusion/SIREN_MATCHA_AB_FUSION_REPORT.md",site_rows,summary_rows,correlations,pair_rows,bootstrap_rows,a_info,b_info)
    _write_json_no_clobber(Path(private_root)/"manifests/private_run_manifest.json",{"schema":"siren_matcha_ab_private_run_manifest_v1","generated_at_utc":pd.Timestamp.utcnow().isoformat(),"source_a":a_info,"source_b":b_info,"source_before":source_before,"source_after":source_after,"private_payload_sha256":private_payload_hashes,"bootstrap":{"n":n_bootstrap,"seed":bootstrap_seed},"git":git_info})
    return {"a_info":a_info,"b_info":b_info,"site_rows":site_rows,"summary_rows":summary_rows,"correlation_rows":correlations,"pair_rows":pair_rows,"bootstrap_rows":bootstrap_rows,"manifest":manifest,"private_root":str(private_root)}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--a-root",type=Path,required=True); parser.add_argument("--demographics",type=Path,required=True); parser.add_argument("--g2-root",type=Path,required=True); parser.add_argument("--b-parquet",type=Path,required=True); parser.add_argument("--b-report",type=Path,required=True); parser.add_argument("--private-root",type=Path,required=True); parser.add_argument("--repo-root",type=Path,default=Path.cwd()); parser.add_argument("--bootstrap",type=int,default=2000); parser.add_argument("--seed",type=int,default=20261023); args=parser.parse_args(); result=run_experiment(a_root=args.a_root,demographics=args.demographics,g2_root=args.g2_root,b_parquet=args.b_parquet,b_report=args.b_report,private_root=args.private_root,repo_root=args.repo_root,n_bootstrap=args.bootstrap,bootstrap_seed=args.seed); print(json.dumps({"branch":result["manifest"]["branch"],"b_macro_ac":result["b_info"]["macro_ac"],"private_root":result["private_root"]},sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
