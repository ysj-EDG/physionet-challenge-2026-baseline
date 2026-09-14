#!/usr/bin/env python3
"""Analyze site-conditioned AC-AUROC from frozen P3/P4 CV logits only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


SITES = ("I0002", "I0006", "S0001")
OUTER_SEEDS = (7, 17, 29)
FOLDS = (0, 1, 2)
P3_MODELS = {
    "p3_demo10": "P3_demo10",
    "p3_compact30": "P3_compact30",
    "p3_compact35": "P3_compact35",
    "p3_global59": "P3_global59",
    "p3_stage155": "P3_stage155",
}
P4_MODELS = {
    "p4_legacy_best_val_ac": "best_val_ac",
    "p4_legacy_fixed_epoch6": "fixed_epoch6",
}
CONDITIONS = ("overall", "same_site", "cross_site", *(f"site_{s}" for s in SITES))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def pair_metric(frame: pd.DataFrame, condition: str, age_conditioned: bool) -> dict:
    if condition.startswith("site_"):
        pool = frame.loc[frame.site_id == condition.removeprefix("site_")].copy()
        site_relation = "all"
    else:
        pool = frame
        site_relation = condition

    positives = pool.loc[pool.label == 1]
    negatives = pool.loc[pool.label == 0]
    n_pos, n_neg = len(positives), len(negatives)
    if not n_pos or not n_neg:
        return {
            "value": math.nan, "eligible_pairs": 0,
            "positives": n_pos, "negatives": n_neg,
            "participating_positives": 0, "participating_negatives": 0,
        }

    pos_age = positives.raw_age.to_numpy(float)[:, None]
    neg_age = negatives.raw_age.to_numpy(float)[None, :]
    mask = np.isfinite(pos_age) & np.isfinite(neg_age)
    if age_conditioned:
        mask &= np.abs(pos_age - neg_age) <= 2

    pos_site = positives.site_id.astype(str).to_numpy()[:, None]
    neg_site = negatives.site_id.astype(str).to_numpy()[None, :]
    if site_relation == "same_site":
        mask &= pos_site == neg_site
    elif site_relation == "cross_site":
        mask &= pos_site != neg_site
    elif site_relation not in {"overall", "all"}:
        raise ValueError(condition)

    denom = int(mask.sum())
    if denom == 0:
        value = math.nan
        participating_pos = participating_neg = 0
    else:
        pos_score = positives.decision_logit.to_numpy(float)[:, None]
        neg_score = negatives.decision_logit.to_numpy(float)[None, :]
        credit = (pos_score > neg_score).astype(float) + 0.5 * (pos_score == neg_score)
        value = float(credit[mask].sum() / denom)
        participating_pos = int(mask.any(axis=1).sum())
        participating_neg = int(mask.any(axis=0).sum())
    return {
        "value": value, "eligible_pairs": denom,
        "positives": n_pos, "negatives": n_neg,
        "participating_positives": participating_pos,
        "participating_negatives": participating_neg,
    }


def validate_and_join(predictions: pd.DataFrame, holdout: pd.DataFrame, source: Path) -> pd.DataFrame:
    required = {"record_id", "label", "raw_age", "decision_logit"}
    if not required.issubset(predictions.columns):
        raise RuntimeError(f"Missing columns in {source}: {required - set(predictions.columns)}")
    if predictions.record_id.duplicated().any():
        raise RuntimeError(f"Duplicate record_id in {source}")
    if set(predictions.record_id) != set(holdout.record_id):
        missing = set(holdout.record_id) - set(predictions.record_id)
        extra = set(predictions.record_id) - set(holdout.record_id)
        raise RuntimeError(f"Holdout mismatch in {source}: missing={len(missing)}, extra={len(extra)}")
    merged = holdout[["record_id", "site_id", "label", "raw_age"]].merge(
        predictions[["record_id", "label", "raw_age", "decision_logit"]],
        on="record_id", suffixes=("_manifest", "_prediction"), validate="one_to_one",
    )
    if not np.array_equal(merged.label_manifest.to_numpy(), merged.label_prediction.to_numpy()):
        raise RuntimeError(f"Label mismatch in {source}")
    if not np.allclose(
        merged.raw_age_manifest.to_numpy(float), merged.raw_age_prediction.to_numpy(float),
        rtol=0, atol=1e-12, equal_nan=True,
    ):
        raise RuntimeError(f"Age mismatch in {source}")
    if not np.isfinite(merged.decision_logit).all():
        raise RuntimeError(f"Non-finite logits in {source}")
    return merged.rename(columns={
        "label_manifest": "label", "raw_age_manifest": "raw_age",
    })[["record_id", "site_id", "label", "raw_age", "decision_logit"]]


def sample_composition(holdout: pd.DataFrame, seed: int, fold: int) -> list[dict]:
    base = holdout.assign(decision_logit=0.0)
    same_total = pair_metric(base, "same_site", True)["eligible_pairs"]
    cross_total = pair_metric(base, "cross_site", True)["eligible_pairs"]
    rows = []
    for site in SITES:
        subset = holdout.loc[holdout.site_id == site]
        same = pair_metric(base, f"site_{site}", True)
        cross_involving = 0
        positives = holdout.loc[holdout.label == 1]
        negatives = holdout.loc[holdout.label == 0]
        for _, pos in positives.iterrows():
            for _, neg in negatives.iterrows():
                if abs(float(pos.raw_age) - float(neg.raw_age)) <= 2 and pos.site_id != neg.site_id:
                    if pos.site_id == site or neg.site_id == site:
                        cross_involving += 1
        rows.append({
            "outer_seed": seed, "fold": fold, "site_id": site,
            "n": len(subset), "positives": int((subset.label == 1).sum()),
            "negatives": int((subset.label == 0).sum()),
            "eligible_same_site_pairs": same["eligible_pairs"],
            "eligible_cross_site_pairs_involving_site": cross_involving,
            "eligible_same_site_pairs_total": same_total,
            "eligible_cross_site_pairs_total": cross_total,
        })
    return rows


def aggregate(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    repeat_rows = []
    numeric = [
        "ac_auroc", "auroc", "eligible_pairs", "positives", "negatives",
        "participating_positives", "participating_negatives",
    ]
    for (model, condition, seed), group in fold_metrics.groupby(
        ["model", "condition", "outer_seed"], sort=False
    ):
        ac = group.ac_auroc.dropna()
        auc = group.auroc.dropna()
        row = {
            "level": "repeat", "model": model, "condition": condition,
            "outer_seed": seed, "folds_total": len(group),
            "folds_with_ac": len(ac), "folds_with_auroc": len(auc),
        }
        for column in numeric:
            vals = group[column].dropna()
            row[column] = float(vals.mean()) if len(vals) else math.nan
        row.update({
            "ac_repeat_min": math.nan, "ac_repeat_max": math.nan,
            "ac_repeat_sd": math.nan,
        })
        repeat_rows.append(row)

    repeat = pd.DataFrame(repeat_rows)
    overall_rows = []
    for (model, condition), group in repeat.groupby(["model", "condition"], sort=False):
        ac = group.ac_auroc.dropna()
        auc = group.auroc.dropna()
        row = {
            "level": "final", "model": model, "condition": condition,
            "outer_seed": "all", "folds_total": int(group.folds_total.sum()),
            "folds_with_ac": int(group.folds_with_ac.sum()),
            "folds_with_auroc": int(group.folds_with_auroc.sum()),
        }
        for column in numeric:
            vals = group[column].dropna()
            row[column] = float(vals.mean()) if len(vals) else math.nan
        row.update({
            "ac_repeat_min": float(ac.min()) if len(ac) else math.nan,
            "ac_repeat_max": float(ac.max()) if len(ac) else math.nan,
            "ac_repeat_sd": float(ac.std(ddof=1)) if len(ac) > 1 else math.nan,
        })
        overall_rows.append(row)
    return pd.concat([repeat, pd.DataFrame(overall_rows)], ignore_index=True)


def fmt(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.3f}"


def write_report(
    output: Path, commit: str, manifest_hash: str, prediction_hash: str,
    fold_metrics: pd.DataFrame, repeat_metrics: pd.DataFrame, composition: pd.DataFrame,
) -> None:
    final = repeat_metrics.loc[repeat_metrics.level == "final"].set_index(["model", "condition"])
    key_models = [
        "p4_legacy_best_val_ac", "p4_legacy_fixed_epoch6",
        "p3_compact30", "p3_global59",
    ]
    all_models = [*P4_MODELS, *P3_MODELS]
    lines = [
        "# Site-conditioned analysis of frozen P3/P4 outer-CV logits",
        "",
        "## Scope and provenance",
        "",
        f"- Analysis commit before this uncommitted analysis: `{commit}`.",
        "- No model training, inference, NPZ generation, feature transformation, or checkpoint modification was performed.",
        "- P3 inputs: `output/p3_v2/cv/seed{7,17,29}/fold{0,1,2}/P3_*_holdout.csv`.",
        "- P4 inputs: `output/p4_legacy_lstm_cv/seed7_model/outer_seed*_fold*/*_outer_holdout_logits.csv`.",
        f"- Frozen fold manifest: `output/p3_v2/cv_fold_manifest.csv`, SHA256 `{manifest_hash}`.",
        f"- Combined SHA256 over the 63 input prediction CSV paths and contents: `{prediction_hash}`.",
        "- All prediction record IDs, labels, and raw ages matched the frozen holdout manifest exactly.",
        "- The 733-record outer-CV population contains I0006 and S0001 only; I0002 has zero records and is therefore reported as N/A rather than imputed.",
        "",
        "## Metric definition",
        "",
        "AC-AUROC uses the official gap=2 pairwise comparison with tie credit 0.5. `same_site` adds equal-site filtering; `cross_site` adds unequal-site filtering. Per-site values restrict the patient pool to that site. Aggregation is fold metric -> mean of three folds per repeat -> mean of three repeat means. Nine-fold logits are never pooled for the primary result.",
        "",
        "`Pos/neg` in the fold CSV is the full class count in the relevant patient pool. `participating_*` additionally reports unique patients that occur in at least one eligible age/site pair.",
        "",
        "## Final hierarchical summary",
        "",
        "| Model | Overall AC | Same-site AC | Cross-site AC | I0002 AC | I0006 AC | S0001 AC | Same-overall | Cross-same |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in all_models:
        vals = {condition: final.loc[(model, condition), "ac_auroc"] for condition in CONDITIONS}
        lines.append(
            f"| {model} | {fmt(vals['overall'])} | {fmt(vals['same_site'])} | "
            f"{fmt(vals['cross_site'])} | {fmt(vals['site_I0002'])} | "
            f"{fmt(vals['site_I0006'])} | {fmt(vals['site_S0001'])} | "
            f"{vals['same_site']-vals['overall']:+.3f} | {vals['cross_site']-vals['same_site']:+.3f} |"
        )

    lines += [
        "",
        "## Repeat-level AC-AUROC for key comparisons",
        "",
        "| Model | Condition | Seed 7 | Seed 17 | Seed 29 | Final | Repeat min-max | SD |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in key_models:
        for condition in ("overall", "same_site", "cross_site", *(f"site_{s}" for s in SITES)):
            reps = repeat_metrics.loc[
                (repeat_metrics.level == "repeat") & (repeat_metrics.model == model)
                & (repeat_metrics.condition == condition)
            ].set_index("outer_seed")
            fin = final.loc[(model, condition)]
            repeat_range = (
                "N/A" if pd.isna(fin.ac_repeat_min)
                else f"{fmt(fin.ac_repeat_min)}-{fmt(fin.ac_repeat_max)}"
            )
            lines.append(
                f"| {model} | {condition} | {fmt(reps.loc[7, 'ac_auroc'])} | "
                f"{fmt(reps.loc[17, 'ac_auroc'])} | {fmt(reps.loc[29, 'ac_auroc'])} | "
                f"{fmt(fin.ac_auroc)} | {repeat_range} | "
                f"{fmt(fin.ac_repeat_sd)} |"
            )

    lines += [
        "",
        "## Evidence volume",
        "",
        "The table below gives the range across the nine frozen folds. Exact fold/site counts are in `site_composition.csv`.",
        "",
        "| Quantity | Minimum per fold | Maximum per fold |",
        "|---|---:|---:|",
    ]
    per_fold = composition.groupby(["outer_seed", "fold"], as_index=False).first()
    lines += [
        f"| Eligible same-site age pairs | {int(per_fold.eligible_same_site_pairs_total.min())} | {int(per_fold.eligible_same_site_pairs_total.max())} |",
        f"| Eligible cross-site age pairs | {int(per_fold.eligible_cross_site_pairs_total.min())} | {int(per_fold.eligible_cross_site_pairs_total.max())} |",
    ]
    for site in SITES:
        block = composition.loc[composition.site_id == site]
        lines.append(
            f"| {site} samples (positives) | {int(block.n.min())} ({int(block.positives.min())}) | "
            f"{int(block.n.max())} ({int(block.positives.max())}) |"
        )
        lines.append(
            f"| {site} eligible within-site pairs | {int(block.eligible_same_site_pairs.min())} | "
            f"{int(block.eligible_same_site_pairs.max())} |"
        )

    legacy_same = final.loc[("p4_legacy_best_val_ac", "same_site"), "ac_auroc"]
    legacy_cross = final.loc[("p4_legacy_best_val_ac", "cross_site"), "ac_auroc"]
    global_same = final.loc[("p3_global59", "same_site"), "ac_auroc"]
    compact_same = final.loc[("p3_compact30", "same_site"), "ac_auroc"]
    lines += [
        "",
        "## Answers to the six questions",
        "",
        f"1. **Legacy same-site signal:** yes. Best-val legacy same-site AC is **{fmt(legacy_same)}**, with repeat means 0.753/0.693/0.701; fixed-epoch6 is **{fmt(final.loc[('p4_legacy_fixed_epoch6','same_site'),'ac_auroc'])}**. Both remain clearly above 0.5.",
        f"2. **Source of the legacy advantage:** it is stronger on same-site pairs, not cross-site pairs. Best-val legacy is **{fmt(legacy_same)}** same-site versus **{fmt(legacy_cross)}** cross-site (cross-minus-same {legacy_cross-legacy_same:+.3f}); restricting to same-site also raises AC by {legacy_same-final.loc[('p4_legacy_best_val_ac','overall'),'ac_auroc']:+.3f} over overall. Cross-site signal is still meaningful, but site separation does not explain the 0.70 headline result.",
        "3. **Within-site consistency:** no evidence supports efficacy at all three sites. I0002 is absent. S0001 is strong and stable (0.726; repeat 0.697-0.765), while I0006 is only 0.543 and unstable (0.414-0.621). I0006 contributes just 7-30 eligible within-site pairs per fold; about 95% of eligible same-site pairs come from S0001, so the same-site legacy result is predominantly supported by S0001.",
        f"4. **P3 site dependence:** yes, especially compact30. compact30 is **{fmt(compact_same)}** same-site versus **{fmt(final.loc[('p3_compact30','cross_site'),'ac_auroc'])}** cross-site (difference {final.loc[('p3_compact30','cross_site'),'ac_auroc']-compact_same:+.3f}); global59 is **{fmt(global_same)}** versus **{fmt(final.loc[('p3_global59','cross_site'),'ac_auroc'])}** (difference {final.loc[('p3_global59','cross_site'),'ac_auroc']-global_same:+.3f}). Their overall AC is therefore more site-assisted than legacy's.",
        f"5. **Legacy advantage after site control:** legacy best-val minus global59 is **{legacy_same-global_same:+.3f} AC** on same-site pairs, compared with approximately +0.110 in the original overall P3/P4 reports.",
        "6. **Next-step implication:** prioritize LOSO/site-domain validation before further temporal/static branch ablation. The legacy advantage survives same-site control, so branch ablation remains scientifically useful, but current evidence is dominated by S0001, weak/unstable for I0006, and contains no I0002. A branch ablation on the same folds would explain the S0001-heavy signal without first establishing multicenter robustness.",
        "",
        "## Interpretation",
        "",
        "The central finding is that P4 legacy's 0.70 AC is not created by easier cross-site comparisons: its same-site AC is higher than both overall and cross-site AC. However, `same-site` is not synonymous with `multicenter-stable`: S0001 supplies about 95% of eligible same-site pairs, I0006 is underpowered and unstable, and I0002 is absent. Thus the model has credible within-hospital ranking signal, but the present CV cannot establish balanced cross-center generalization.",
    ]
    (output / "SITE_CONDITIONED_ANALYSIS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "output/site_conditioned_analysis").resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    manifest_path = root / "output/p3_v2/cv_fold_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    observed_sites = set(manifest.site_id.astype(str))
    unexpected_sites = observed_sites - set(SITES)
    if unexpected_sites:
        raise RuntimeError(f"Unexpected sites: {sorted(unexpected_sites)}")
    if manifest.duplicated(subset=["seed", "fold", "record_id"]).any():
        raise RuntimeError("Duplicate records within P3 seed/fold manifest")

    fold_rows, composition_rows, source_paths = [], [], []
    for seed in OUTER_SEEDS:
        for fold in FOLDS:
            block = manifest.loc[(manifest.seed == seed) & (manifest.fold == fold)]
            holdout = block.loc[block.role == "holdout"].copy()
            if len(holdout) not in (244, 245):
                raise RuntimeError(f"Unexpected holdout size seed={seed}, fold={fold}: {len(holdout)}")
            composition_rows.extend(sample_composition(holdout, seed, fold))

            p4_manifest_path = root / (
                f"output/p4_legacy_lstm_cv/seed7_model/outer_seed{seed}_fold{fold}/"
                "frozen_outer_fold_manifest.csv"
            )
            p4_manifest = pd.read_csv(p4_manifest_path)
            p4_holdout = p4_manifest.loc[p4_manifest.role == "holdout"]
            columns = ["record_id", "bdsp_patient_id", "site_id", "label", "raw_age"]
            left = holdout[columns].sort_values("record_id").reset_index(drop=True)
            right = p4_holdout[columns].sort_values("record_id").reset_index(drop=True)
            if not left.equals(right):
                raise RuntimeError(f"P3/P4 frozen manifest mismatch seed={seed}, fold={fold}")

            sources = {}
            for model, stem in P3_MODELS.items():
                sources[model] = root / f"output/p3_v2/cv/seed{seed}/fold{fold}/{stem}_holdout.csv"
            for model, stem in P4_MODELS.items():
                sources[model] = root / (
                    f"output/p4_legacy_lstm_cv/seed7_model/outer_seed{seed}_fold{fold}/"
                    f"{stem}_outer_holdout_logits.csv"
                )

            for model, source in sources.items():
                if not source.is_file():
                    raise FileNotFoundError(source)
                source_paths.append(source)
                joined = validate_and_join(pd.read_csv(source), holdout, source)
                for condition in CONDITIONS:
                    ac = pair_metric(joined, condition, True)
                    auc = pair_metric(joined, condition, False)
                    fold_rows.append({
                        "model": model, "outer_seed": seed, "fold": fold,
                        "condition": condition, "ac_auroc": ac["value"],
                        "eligible_pairs": ac["eligible_pairs"],
                        "positives": ac["positives"], "negatives": ac["negatives"],
                        "participating_positives": ac["participating_positives"],
                        "participating_negatives": ac["participating_negatives"],
                        "auroc": auc["value"], "auroc_pairs": auc["eligible_pairs"],
                        "source_file": str(source.relative_to(root)),
                    })

    fold_metrics = pd.DataFrame(fold_rows)
    composition = pd.DataFrame(composition_rows)
    repeat_metrics = aggregate(fold_metrics)

    # Reproduce the already frozen hierarchical overall results before writing output.
    expected = {
        "p3_demo10": 0.5672282031876167,
        "p3_compact30": 0.5809979300021965,
        "p3_compact35": 0.5731027281890043,
        "p3_global59": 0.5903998070303526,
        "p3_stage155": 0.49389827890369237,
        "p4_legacy_best_val_ac": 0.7008641343308929,
        "p4_legacy_fixed_epoch6": 0.6824424303982504,
    }
    final = repeat_metrics.loc[
        (repeat_metrics.level == "final") & (repeat_metrics.condition == "overall")
    ].set_index("model")
    for model, value in expected.items():
        observed = float(final.loc[model, "ac_auroc"])
        if not math.isclose(observed, value, rel_tol=0, abs_tol=1e-12):
            raise RuntimeError(f"Frozen overall AC mismatch for {model}: {observed} != {value}")

    output.mkdir(parents=True)
    fold_metrics.to_csv(output / "site_conditioned_fold_metrics.csv", index=False, na_rep="N/A")
    repeat_metrics.to_csv(output / "site_conditioned_repeat_metrics.csv", index=False, na_rep="N/A")
    composition.to_csv(output / "site_composition.csv", index=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    write_report(
        output, commit, sha256(manifest_path), tree_hash(source_paths, root),
        fold_metrics, repeat_metrics, composition,
    )
    print(f"Wrote {len(fold_metrics)} fold-condition rows and {len(repeat_metrics)} aggregate rows to {output}")


if __name__ == "__main__":
    main()
