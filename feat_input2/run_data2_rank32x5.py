#!/usr/bin/env python3
"""Frozen DATA2 compact30 age-windowed pairwise Rank32x5 experiment."""
from __future__ import annotations
import hashlib, json, os, subprocess, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from feat_input.feat_mody import run_p3_v2 as p3
from feat_input.feat_mody import run_loso_1103 as p5
from feature_scaling import load_feature_rules

SITES = ("I0002", "I0006", "S0001")
SEEDS = (7, 17, 27, 37, 47)
BASELINE = {"I0002": 0.7476139978791092, "I0006": 0.6805354308976748, "S0001": 0.6538199967092634}
MANIFEST = ROOT / "feat_input2/output/data2_lowcost/data2_manifest.csv"
CACHE = ROOT / "feat_input2/output/data2_lowcost/cache/data2_patient_summaries.npz"
OUT = ROOT / "output/data2_rank32x5_v1"
EXPECTED_NAMES = [
    "age", "sex_Female", "sex_Male", "sex_OtherUnknown", "race_Asian", "race_Black",
    "race_Others", "race_Unavailable", "race_White", "BMI", "caisr_sleep_tst_sec",
    "caisr_sleep_se", "caisr_sleep_sol_sec", "caisr_sleep_rem_latency_sec", "caisr_sleep_waso_sec",
    "caisr_sleep_n1_pct", "caisr_sleep_n2_pct", "caisr_sleep_n3_pct", "caisr_sleep_rem_pct",
    "caisr_sleep_transition_rate", "caisr_sleep_short_bout_ratio", "caisr_sleep_mean_stage_entropy",
    "caisr_arousal_arousal_index", "caisr_arousal_arousal_burden_ratio", "caisr_arousal_mean_arousal_duration",
    "caisr_respiratory_respiratory_event_index", "caisr_respiratory_respiratory_burden_ratio",
    "caisr_respiratory_mean_resp_duration", "caisr_limb_limb_movement_index", "caisr_limb_plmi",
]

def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

def metrics(y, score, age):
    return p5.metric_bundle(np.asarray(y, int), np.asarray(score, float), np.asarray(age, float))

def load_inputs():
    m = pd.read_csv(MANIFEST, dtype={"record_id": str, "site": str})
    if len(m) != 6600 or m.record_id.nunique() != 6600:
        raise RuntimeError("compact30 manifest is not 6600 unique records")
    if m.groupby("site").size().to_dict() != {"I0002": 319, "I0006": 1142, "S0001": 5139}:
        raise RuntimeError("site counts do not match frozen DATA2 contract")
    rules = load_feature_rules(ROOT / "feat_input/feature_rules_v1.json")
    holder = type("Rules", (), {"rules": rules})()
    names = p3.feature_names(holder, "P3_compact30")
    if names != EXPECTED_NAMES:
        raise RuntimeError(f"compact30 feature order mismatch: {names}")
    z = np.load(CACHE, allow_pickle=False)
    mats = {s: np.asarray(z[f"p3_{s}_compact30"], float) for s in SITES}
    if any(x.shape != (6600, 30) or not np.isfinite(x).all() for x in mats.values()):
        raise RuntimeError("bad compact30 cache matrices")
    return m, mats, names

def baseline_check(m, mats):
    rows = []
    for hold in SITES:
        tr = m.site.to_numpy() != hold; te = ~tr; x = mats[hold]
        med, mu, scale, missing = p3.fit_imputer_scaler(x[tr])
        model = LogisticRegression(**p3.LR).fit(p3.apply_imputer_scaler(x[tr], med, mu, scale), m.label.to_numpy()[tr])
        got = metrics(m.label.to_numpy()[te], model.decision_function(p3.apply_imputer_scaler(x[te], med, mu, scale)), m.age.to_numpy()[te])
        rows.append({"holdout_site": hold, "frozen_ac": BASELINE[hold], "reproduced_ac": got["ac_auroc"], "abs_diff": abs(got["ac_auroc"]-BASELINE[hold]), "status": "PASS" if abs(got["ac_auroc"]-BASELINE[hold]) <= 1e-6 else "FAIL"})
    df = pd.DataFrame(rows); df.to_csv(OUT / "baseline_repro_check.csv", index=False)
    (OUT / "BASELINE_REPRO_CHECK.md").write_text("# Baseline reproducibility check\n\n" + df.to_markdown(index=False) + "\n\nAll absolute differences must be <= 1e-6.\n")
    if not (df.status == "PASS").all(): raise RuntimeError("baseline reproduction failed")
    return df

def make_pairs(x, y, age, sites, seed):
    rng = np.random.default_rng(seed); rows=[]; pair_keys=[]; counts=[]; zero=[]; total=0
    for site in sorted(set(sites)):
        ix = np.flatnonzero(sites == site); pos = ix[y[ix] == 1]; neg = ix[y[ix] == 0]; nsite = 0
        for p in pos:
            eligible = neg[np.abs(age[neg] - age[p]) <= 2]
            if len(eligible) == 0: zero.append({"training_site": site, "positive_index": int(p), "seed": seed}); continue
            chosen = eligible if len(eligible) <= 32 else rng.choice(eligible, 32, replace=False)
            for n in chosen:
                rows.append((x[p]-x[n], 1)); rows.append((x[n]-x[p], 0)); pair_keys.append((site, int(p), int(n))); nsite += 1
        eligible_counts = [int(np.sum(np.abs(age[neg] - age[p]) <= 2)) for p in pos]
        counts.append({"training_site": site, "seed": seed, "original_pairs": nsite, "positives_zero_eligible": sum(z["training_site"] == site for z in zero), "positives_le_32": int(sum(0 < q <= 32 for q in eligible_counts)), "positives_gt_32": int(sum(q > 32 for q in eligible_counts))})
    if not rows or any(c["original_pairs"] == 0 for c in counts): raise RuntimeError("empty site pair set")
    total_pairs = sum(c["original_pairs"] for c in counts); weights=[]
    for c in counts:
        w = total_pairs / (len(counts) * c["original_pairs"]); c.update({"symmetric_examples": 2*c["original_pairs"], "per_example_weight": w, "total_site_weight": 2*c["original_pairs"]*w}); weights.extend([w] * (2*c["original_pairs"]))
    w = np.asarray(weights, float); assert abs(w.mean()-1) < 1e-12
    return np.asarray([r[0] for r in rows]), np.asarray([r[1] for r in rows]), w, counts, pair_keys, zero

def run_rankers(m, mats, names):
    pred_dir = OUT / "predictions"; pred_dir.mkdir(parents=True, exist_ok=True)
    pair_rows=[]; sample_rows=[]; coef_rows=[]; sim_rows=[]; fold_rows=[]; all_scores={}
    yall=m.label.to_numpy(int); ageall=m.age.to_numpy(float); sites=m.site.to_numpy(str)
    for hold in SITES:
        tr = sites != hold; te = ~tr; x = mats[hold]; med, mu, scale, missing = p3.fit_imputer_scaler(x[tr]); xt=p3.apply_imputer_scaler(x[tr],med,mu,scale); xv=p3.apply_imputer_scaler(x[te],med,mu,scale)
        scores=[]; coeff=[]; seed_pairs=[]; seed_metrics=[]
        for seed in SEEDS:
            d, labels, weights, counts, keys, zero = make_pairs(xt, yall[tr], ageall[tr], sites[tr], seed)
            cfg=dict(penalty="l2", C=.03, fit_intercept=False, class_weight=None, solver="lbfgs", max_iter=100000, tol=1e-6)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning); model=LogisticRegression(**cfg).fit(d, labels, sample_weight=weights)
            warns=[str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
            scores.append(model.decision_function(xv)); coeff.append(model.coef_[0]); seed_pairs.append(set(keys)); seed_metrics.append({"seed":seed,"holdout_site":hold,"n_iter":int(model.n_iter_[0]),"converged":not warns,"warnings":" | ".join(warns),"pair_count":sum(c["original_pairs"] for c in counts)})
            for c in counts: pair_rows.append({"outer_holdout":hold,**c})
            sample_rows.extend({"outer_holdout":hold,**q,"zero_eligible_positives":len(zero)} for q in seed_metrics[-1:] for _ in [0])
            if zero: sample_rows.extend({"outer_holdout":hold,"seed":seed,"training_site":q["training_site"],"positive_index":q["positive_index"],"zero_eligible":True} for q in zero)
        ens=np.mean(scores,axis=0); all_scores[hold]=ens; hm=metrics(yall[te],ens,ageall[te])
        frame=m.loc[te,["record_id","patient_id","site","label","age"]].copy(); frame["rank_score"]=ens; frame.to_csv(pred_dir/f"holdout_{hold}.csv",index=False)
        c=np.asarray(coeff); norms=np.linalg.norm(c,axis=1); cos=(c@c.T)/(norms[:,None]*norms[None,:]); sign=(np.sign(c)[:,None,:]==np.sign(c)[None,:,:]).mean(2)
        for i,seed in enumerate(SEEDS):
            fold_rows.append({"holdout_site":hold,"seed":seed,**metrics(yall[te],scores[i],ageall[te]),"pair_count":seed_metrics[i]["pair_count"],"converged":seed_metrics[i]["converged"],"n_iter":seed_metrics[i]["n_iter"],"warnings":seed_metrics[i]["warnings"]})
            for j,n in enumerate(names): coef_rows.append({"holdout_site":hold,"seed":seed,"feature":n,"coefficient":c[i,j]})
        for i in range(5):
            for j in range(i+1,5): sim_rows.append({"holdout_site":hold,"seed_a":SEEDS[i],"seed_b":SEEDS[j],"coefficient_cosine":cos[i,j],"feature_sign_agreement":sign[i,j],"pair_overlap":len(seed_pairs[i]&seed_pairs[j])/max(1,len(seed_pairs[i]|seed_pairs[j]))})
    fold=pd.DataFrame(fold_rows); fold.to_csv(OUT/"fold_metrics.csv",index=False); pd.DataFrame(pair_rows).to_csv(OUT/"pair_counts.csv",index=False); pd.DataFrame(sample_rows).to_csv(OUT/"pair_sampling_summary.csv",index=False); pd.DataFrame(coef_rows).to_csv(OUT/"ranker_coefficients.csv",index=False); pd.DataFrame(sim_rows).to_csv(OUT/"ranker_similarity.csv",index=False)
    agg={"sites":{},"baseline":BASELINE}
    for s in SITES:
        q=fold[fold.holdout_site==s]; agg["sites"][s]={"ac_mean":float(q.ac_auroc.mean()),"ac_sd":float(q.ac_auroc.std(ddof=1)),"auroc_mean":float(q.auroc.mean()),"auprc_mean":float(q.auprc.mean()),"age_weighted_mean":float(q.age_weighted_auroc.mean()),"delta":float(q.ac_auroc.mean()-BASELINE[s])}
    piv=fold.pivot(index="seed",columns="holdout_site",values="ac_auroc"); macro=piv.mean(1); worst=piv.min(1); agg.update({"macro_ac_mean":float(macro.mean()),"macro_ac_sd":float(macro.std(ddof=1)),"worst_ac_mean":float(worst.mean()),"worst_ac_sd":float(worst.std(ddof=1)),"macro_delta":float(macro.mean()-np.mean(list(BASELINE.values()))),"worst_delta":float(worst.mean()-min(BASELINE.values())),"classification":"Strong positive" if macro.mean()>np.mean(list(BASELINE.values())) and worst.mean()>=min(BASELINE.values()) and sum(agg["sites"][s]["delta"]>=0 for s in SITES)>=2 else ("Negative" if macro.mean()<np.mean(list(BASELINE.values())) and worst.mean()<min(BASELINE.values()) else "Mixed")})
    (OUT/"aggregate_metrics.json").write_text(json.dumps(agg,indent=2)+"\n")
    lines=["# DATA2 compact30 Rank32x5 results","",f"Frozen baseline Macro/Worst: {np.mean(list(BASELINE.values())):.12f} / {min(BASELINE.values()):.12f}","","| Site | Rank AC mean ± SD | Baseline AC | Delta |","|---|---:|---:|---:|"]
    for s in SITES: lines.append(f"| {s} | {agg['sites'][s]['ac_mean']:.6f} ± {agg['sites'][s]['ac_sd']:.6f} | {BASELINE[s]:.6f} | {agg['sites'][s]['delta']:+.6f} |")
    lines += ["",f"Macro AC: {agg['macro_ac_mean']:.6f} ± {agg['macro_ac_sd']:.6f}; delta {agg['macro_delta']:+.6f}",f"Worst AC: {agg['worst_ac_mean']:.6f} ± {agg['worst_ac_sd']:.6f}; delta {agg['worst_delta']:+.6f}",f"Pre-specified classification: **{agg['classification']}**.","","Ranker: l2 LogisticRegression, C=0.03, no intercept; five age-windowed within-site subsamples; symmetric differences; equal total site weights; raw-score mean ensemble."]
    (OUT/"RANK32X5_RESULTS.md").write_text("\n".join(lines)+"\n")
    return fold, agg

def main():
    if OUT.exists(): raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    m,mats,names=load_inputs(); (OUT/"run_context.json").write_text(json.dumps({"source_commit":git("rev-parse","HEAD"),"manifest":str(MANIFEST),"cache":str(CACHE),"seeds":SEEDS,"feature_names":names},indent=2)+"\n")
    baseline_check(m,mats)
    fold,agg=run_rankers(m,mats,names)
    print(json.dumps(agg,indent=2),flush=True)
if __name__ == "__main__": main()
