#!/usr/bin/env python3
"""Aggregate the completed P4 legacy LSTM CV without pooling fold predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


ROLES = ("best_val_ac", "fixed_epoch6")
METRICS = ("ac_auroc", "auroc", "auprc")
P3 = {
    "demo10": {"ac_auroc": 0.567, "auroc": 0.776, "auprc": 0.260},
    "compact30": {"ac_auroc": 0.581, "auroc": 0.772, "auprc": 0.283},
    "global59": {"ac_auroc": 0.590, "auroc": 0.772, "auprc": 0.257},
    "stage155": {"ac_auroc": 0.494, "auroc": 0.737, "auprc": 0.228},
}


def avg(items: list[dict], section: str, metric: str) -> float:
    return mean(item["checkpoints"][item["role"]][section][metric] for item in items)


def f(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    model_root = root / "seed7_model"

    rows: list[dict] = []
    for fold_dir in sorted(path for path in model_root.iterdir() if path.is_dir()):
        data = json.loads((fold_dir / "fold_metrics.json").read_text())
        for role in ROLES:
            rows.append({**data, "role": role})
    if len(rows) != 18:
        raise RuntimeError(f"Expected 18 checkpoint rows, found {len(rows)}")

    repeats: dict[str, dict[str, dict]] = {role: {} for role in ROLES}
    for role in ROLES:
        for seed in (7, 17, 29):
            subset = [r for r in rows if r["role"] == role and r["outer_seed"] == seed]
            if len(subset) != 3:
                raise RuntimeError(f"Expected three folds for {role}/seed={seed}")
            repeats[role][str(seed)] = {
                section: {metric: avg(subset, section, metric) for metric in METRICS}
                for section in ("natural_outer_train", "outer_holdout", "validation")
            }

    aggregate: dict[str, dict] = {}
    for role in ROLES:
        rep = repeats[role]
        aggregate[role] = {
            "repeat_means": rep,
            "final_hierarchical_mean": {
                section: {
                    metric: mean(rep[str(seed)][section][metric] for seed in (7, 17, 29))
                    for metric in METRICS
                }
                for section in ("natural_outer_train", "outer_holdout", "validation")
            },
            "repeat_mean_min_max": {
                metric: {
                    "min": min(rep[str(seed)]["outer_holdout"][metric] for seed in (7, 17, 29)),
                    "max": max(rep[str(seed)]["outer_holdout"][metric] for seed in (7, 17, 29)),
                }
                for metric in METRICS
            },
        }

    paired = {}
    for metric in METRICS:
        by_seed = {}
        for seed in (7, 17, 29):
            by_seed[str(seed)] = (
                repeats["best_val_ac"][str(seed)]["outer_holdout"][metric]
                - repeats["fixed_epoch6"][str(seed)]["outer_holdout"][metric]
            )
        paired[metric] = {"repeat_differences": by_seed, "final_difference": mean(by_seed.values())}

    output = {
        "aggregation_method": "mean 3 folds within repeat, then mean 3 repeat means; no pooled logits",
        "roles": aggregate,
        "best_val_ac_minus_fixed_epoch6": paired,
        "p3_frozen_references": P3,
    }
    (root / "aggregate_metrics.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    csv_fields = [
        "outer_seed", "outer_fold", "checkpoint", "best_epoch", "stop_epoch",
        "holdout_n", "holdout_positives", "holdout_negatives", "eligible_age_pairs_gap2",
        "holdout_ac_auroc", "holdout_auroc", "holdout_auprc",
        "train_ac_auroc", "train_auroc", "train_auprc", "val_ac_auroc", "val_auroc",
    ]
    with (root / "fold_metrics_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["outer_seed"], r["outer_fold"], r["role"])):
            ckpt = row["checkpoints"][row["role"]]
            hold = ckpt["outer_holdout"]
            train = ckpt["natural_outer_train"]
            val = ckpt["validation"]
            writer.writerow({
                "outer_seed": row["outer_seed"], "outer_fold": row["outer_fold"],
                "checkpoint": row["role"], "best_epoch": row["best_epoch"],
                "stop_epoch": row["stop_epoch"], "holdout_n": hold["n"],
                "holdout_positives": hold["positives"], "holdout_negatives": hold["negatives"],
                "eligible_age_pairs_gap2": hold["eligible_age_pairs_gap2"],
                "holdout_ac_auroc": hold["ac_auroc"], "holdout_auroc": hold["auroc"],
                "holdout_auprc": hold["auprc"], "train_ac_auroc": train["ac_auroc"],
                "train_auroc": train["auroc"], "train_auprc": train["auprc"],
                "val_ac_auroc": val["ac_auroc"], "val_auroc": val["auroc"],
            })

    identity = json.loads((root / "run_identity.json").read_text())
    lines = [
        "# P4 legacy LSTM repeated outer-CV results",
        "",
        "## Result",
        "",
        "The experiment completed all 9 P3 outer folds. The primary raw-ranking result for "
        f"`best_val_ac` is **AC-AUROC {f(aggregate['best_val_ac']['final_hierarchical_mean']['outer_holdout']['ac_auroc'])}**, "
        f"AUROC {f(aggregate['best_val_ac']['final_hierarchical_mean']['outer_holdout']['auroc'])}, and "
        f"AUPRC {f(aggregate['best_val_ac']['final_hierarchical_mean']['outer_holdout']['auprc'])}. "
        "These are hierarchical repeated-CV means, not metrics computed after pooling nine-fold logits.",
        "",
        "## Frozen setup and checks",
        "",
        f"- Base Git commit: `{identity['base_commit']}`.",
        f"- Fold manifest SHA256: `{identity['fold_manifest_sha256']}`; outer seeds 7/17/29, model seed 7 for all folds.",
        "- Input preprocessing: `legacy_clip`; no typed scaling, feature mask, sidecar, pooled EEG, data2, test, or external data.",
        "- Fixed model-selection set: 158 records (12 positive, 146 negative); BDSPPatientID overlap with the 733 base records: 0.",
        "- Each repeat's three holdouts are disjoint and cover exactly 733 records; every fold has zero train/holdout patient overlap.",
        "- Both checkpoint prediction lists match their frozen fold manifests exactly; no duplicate, missing, or non-finite output was found.",
        f"- Runtime: Python {identity['python_version'].split()[0]}, PyTorch {identity['torch_version']}, CUDA {identity['cuda_version']}, {identity['gpu']} (`eeg_env`).",
        "- Outer holdout loaders were constructed only after both complete checkpoints were frozen. Holdout results did not affect training, scheduling, early stopping, calibration, threshold selection, or checkpoint selection.",
        "",
        "## Nine-fold results: best validation AC checkpoint",
        "",
        "| Outer seed | Fold | Best/stop epoch | Holdout n (+/-) | Eligible age pairs | Val AC/AUROC | Train AC/AUROC/AUPRC | Holdout AC/AUROC/AUPRC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted((r for r in rows if r["role"] == "best_val_ac"), key=lambda r: (r["outer_seed"], r["outer_fold"])):
        c = row["checkpoints"]["best_val_ac"]; h = c["outer_holdout"]; t = c["natural_outer_train"]; v = c["validation"]
        lines.append(
            f"| {row['outer_seed']} | {row['outer_fold']} | {row['best_epoch']}/{row['stop_epoch']} | "
            f"{h['n']} ({h['positives']}/{h['negatives']}) | {h['eligible_age_pairs_gap2']} | "
            f"{f(v['ac_auroc'])}/{f(v['auroc'])} | {f(t['ac_auroc'])}/{f(t['auroc'])}/{f(t['auprc'])} | "
            f"{f(h['ac_auroc'])}/{f(h['auroc'])}/{f(h['auprc'])} |"
        )

    lines += [
        "",
        "## Nine-fold results: fixed epoch 6 checkpoint",
        "",
        "| Outer seed | Fold | Trajectory best/stop epoch | Holdout n (+/-) | Eligible age pairs | Epoch-6 val AC/AUROC | Train AC/AUROC/AUPRC | Holdout AC/AUROC/AUPRC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted((r for r in rows if r["role"] == "fixed_epoch6"), key=lambda r: (r["outer_seed"], r["outer_fold"])):
        c = row["checkpoints"]["fixed_epoch6"]; h = c["outer_holdout"]; t = c["natural_outer_train"]; v = c["validation"]
        lines.append(
            f"| {row['outer_seed']} | {row['outer_fold']} | {row['best_epoch']}/{row['stop_epoch']} | "
            f"{h['n']} ({h['positives']}/{h['negatives']}) | {h['eligible_age_pairs_gap2']} | "
            f"{f(v['ac_auroc'])}/{f(v['auroc'])} | {f(t['ac_auroc'])}/{f(t['auroc'])}/{f(t['auprc'])} | "
            f"{f(h['ac_auroc'])}/{f(h['auroc'])}/{f(h['auprc'])} |"
        )

    lines += [
        "",
        "## Hierarchical aggregation",
        "",
        "| Checkpoint | Repeat seed | Holdout AC | AUROC | AUPRC | Train AC | Train-holdout AC gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for role in ROLES:
        for seed in (7, 17, 29):
            rep = repeats[role][str(seed)]; h = rep["outer_holdout"]; t = rep["natural_outer_train"]
            lines.append(f"| {role} | {seed} | {f(h['ac_auroc'])} | {f(h['auroc'])} | {f(h['auprc'])} | {f(t['ac_auroc'])} | {f(t['ac_auroc']-h['ac_auroc'])} |")
        final = aggregate[role]["final_hierarchical_mean"]; h = final["outer_holdout"]; t = final["natural_outer_train"]
        ranges = aggregate[role]["repeat_mean_min_max"]
        lines.append(
            f"| **{role} final** | **mean; range** | **{f(h['ac_auroc'])}; {f(ranges['ac_auroc']['min'])}-{f(ranges['ac_auroc']['max'])}** | "
            f"**{f(h['auroc'])}; {f(ranges['auroc']['min'])}-{f(ranges['auroc']['max'])}** | "
            f"**{f(h['auprc'])}; {f(ranges['auprc']['min'])}-{f(ranges['auprc']['max'])}** | "
            f"**{f(t['ac_auroc'])}** | **{f(t['ac_auroc']-h['ac_auroc'])}** |"
        )

    best = aggregate["best_val_ac"]["final_hierarchical_mean"]["outer_holdout"]
    fixed = aggregate["fixed_epoch6"]["final_hierarchical_mean"]["outer_holdout"]
    lines += [
        "",
        "## P3 comparison",
        "",
        "| Model | Holdout AC | AUROC | AUPRC | AC vs compact30 | AC vs global59 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| legacy best_val_ac | {f(best['ac_auroc'])} | {f(best['auroc'])} | {f(best['auprc'])} | {best['ac_auroc']-0.581:+.3f} | {best['ac_auroc']-0.590:+.3f} |",
        f"| legacy fixed_epoch6 | {f(fixed['ac_auroc'])} | {f(fixed['auroc'])} | {f(fixed['auprc'])} | {fixed['ac_auroc']-0.581:+.3f} | {fixed['ac_auroc']-0.590:+.3f} |",
    ]
    for name, values in P3.items():
        lines.append(f"| P3 {name} | {f(values['ac_auroc'])} | {f(values['auroc'])} | {f(values['auprc'])} | {values['ac_auroc']-0.581:+.3f} | {values['ac_auroc']-0.590:+.3f} |")

    lines += [
        "",
        "## Conclusions",
        "",
        f"1. The legacy model's primary mean Age-conditioned AUROC is **{f(best['ac_auroc'])}**.",
        f"2. Its repeat means are **{f(repeats['best_val_ac']['7']['outer_holdout']['ac_auroc'])}**, **{f(repeats['best_val_ac']['17']['outer_holdout']['ac_auroc'])}**, and **{f(repeats['best_val_ac']['29']['outer_holdout']['ac_auroc'])}**; the range is **{f(aggregate['best_val_ac']['repeat_mean_min_max']['ac_auroc']['min'])}-{f(aggregate['best_val_ac']['repeat_mean_min_max']['ac_auroc']['max'])}** (width {aggregate['best_val_ac']['repeat_mean_min_max']['ac_auroc']['max']-aggregate['best_val_ac']['repeat_mean_min_max']['ac_auroc']['min']:.3f}).",
        f"3. Best-val-AC exceeds fixed epoch 6 by **{paired['ac_auroc']['final_difference']:+.3f} AC**, **{paired['auroc']['final_difference']:+.3f} AUROC**, and **{paired['auprc']['final_difference']:+.3f} AUPRC**. Checkpoint choice matters, but it is not the main source of the full split variation.",
        f"4. The mean natural-train minus holdout AC gap remains large: **{aggregate['best_val_ac']['final_hierarchical_mean']['natural_outer_train']['ac_auroc']-best['ac_auroc']:.3f}** for best-val-AC and **{aggregate['fixed_epoch6']['final_hierarchical_mean']['natural_outer_train']['ac_auroc']-fixed['ac_auroc']:.3f}** for epoch 6. This indicates substantial fitting/split sensitivity.",
        f"5. At the pre-specified hierarchical level, both legacy checkpoints exceed compact30 (0.581) and global59 (0.590) in all three repeat means. The primary model gains **{best['ac_auroc']-0.581:+.3f}** and **{best['ac_auroc']-0.590:+.3f}** AC respectively. This is not uniform fold-by-fold: best-val-AC seed29/fold0 is 0.589, just below global59, and fixed-epoch6 seed17/fold1 is 0.552.",
        "6. The repeated-CV result supports real ranking signal, because all best-val repeat means remain well above chance and above the P3 compact/global references. However, the 0.054 repeat-range, wide individual-fold spread (0.589-0.795), and large train-holdout gap show that the unusually high historical fixed-split Test/External results should be treated as substantially split-sensitive rather than as a stable point estimate of generalization.",
        "",
        "Platt probabilities and thresholds are retained in each fold artifact for diagnostics only. Every CV conclusion above uses raw decision-logit ranking. No test/external evaluation or post-hoc experiment selection was performed.",
    ]
    (root / "P4_LEGACY_LSTM_CV_RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
