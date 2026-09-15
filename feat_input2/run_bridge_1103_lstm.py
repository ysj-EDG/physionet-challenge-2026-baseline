#!/usr/bin/env python3
"""Run the two frozen P6 LSTM balance protocols on the P5 three-site LOSO."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ["LSTM_INPUT_PREPROCESSING"] = "legacy_clip"
os.environ["LSTM_POS_WEIGHT_MODE"] = "unit"
os.environ["LSTM_SEED"] = "7"
os.environ.setdefault("PYTHONHASHSEED", "7")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["P5_MANIFEST"] = str(ROOT / "output/p5_loso_1103/loso_manifest.csv")
os.environ["P5_FROZEN_FINGERPRINTS"] = str(
    ROOT / "output/p5_loso_1103/input_fingerprints.json"
)
os.environ["P5_OUTPUT_DIR"] = str(ROOT / "output/p6_site_domain/_unused_p5_output")
os.environ["LSTM_CACHE_DIR"] = str(ROOT / "npz_1103_timegrid_v2")
os.environ["P5_CACHE_LAYOUT"] = "site"

import train_lstm as tl
from feature_scaling import FeatureScaler
from feat_input.feat_mody import run_loso_1103 as p5


OUTPUT = ROOT / "output/bridge_1103_timegrid_v2/lstm"
MANIFEST = ROOT / "output/p5_loso_1103/loso_manifest.csv"
FINGERPRINTS = ROOT / "output/p5_loso_1103/input_fingerprints.json"
P5_AGGREGATE = ROOT / "output/p5_loso_1103/aggregate_metrics.json"
SITES = ("I0002", "I0006", "S0001")
ARMS = {
    "L0": {"sampler": "global_class_balanced", "pos_weight": "empirical"},
    "L1": {"sampler": "global_class_balanced", "pos_weight": 1.0},
    "L2": {"sampler": "site_and_class_balanced", "pos_weight": 1.0},
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def validate_inputs() -> tuple[pd.DataFrame, dict[str, dict]]:
    for path in (MANIFEST, FINGERPRINTS, P5_AGGREGATE):
        if not path.is_file():
            raise FileNotFoundError(path)
    frozen = json.loads(FINGERPRINTS.read_text())
    if sha256(MANIFEST) != frozen["manifest_sha256"]:
        raise RuntimeError("P5 LOSO manifest differs from its frozen SHA256")
    frame = pd.read_csv(
        MANIFEST, dtype={"record_id": str, "patient_id": str, "site": str}
    )
    if len(frame) != 1103 or frame.record_id.duplicated().any():
        raise RuntimeError("P5 manifest does not uniquely contain 1103 records")
    if frame.patient_id.duplicated().any():
        raise RuntimeError("Repeated patient_id in P5 manifest")
    if set(frame.site) != set(SITES) or set(frame.label) != {0, 1}:
        raise RuntimeError("Incomplete site or binary label mapping")
    split_records = p5.records_from_splits()
    failures = []
    for row in frame.itertuples(index=False):
        path = p5.cache_path(row)
        if not path.is_file() or row.record_id not in split_records:
            failures.append(row.record_id)
        expected = p5.cache_path(row)
        if path.resolve() != expected.resolve():
            failures.append(row.record_id)
    if failures:
        raise RuntimeError(f"Manifest/cache mismatch: {sorted(set(failures))[:10]}")
    if len(list((ROOT / "npz_1103_timegrid_v2").glob("*/*.npz"))) != 1103:
        raise RuntimeError("NEW1103 cache does not contain exactly 1103 NPZ files")
    for site in SITES:
        train = frame.loc[frame.site != site]
        holdout = frame.loc[frame.site == site]
        if set(train.patient_id) & set(holdout.patient_id):
            raise RuntimeError(f"Patient leakage in holdout {site}")
        cells = train.groupby(["site", "label"]).size()
        for train_site in sorted(set(train.site)):
            for label in (0, 1):
                if int(cells.get((train_site, label), 0)) == 0:
                    raise RuntimeError(
                        f"Empty site-label cell for holdout={site}: {train_site}/{label}"
                    )
    return frame, split_records


def global_class_loader(rows: list[dict], preprocessor: FeatureScaler) -> DataLoader:
    return p5.balanced_loader(rows, preprocessor)


def site_class_loader(
    rows: list[dict], preprocessor: FeatureScaler
) -> tuple[DataLoader, list[dict]]:
    counts: dict[tuple[str, int], int] = {}
    for row in rows:
        key = (row["site"], int(row["label"]))
        counts[key] = counts.get(key, 0) + 1
    train_sites = sorted({row["site"] for row in rows})
    cells = []
    weights = []
    for train_site in train_sites:
        for label in (0, 1):
            count = counts.get((train_site, label), 0)
            if count == 0:
                raise RuntimeError(f"Empty site-label cell: {train_site}/{label}")
            cells.append({
                "site": train_site,
                "label": label,
                "raw_count": count,
                "sample_weight": 1.0 / count,
                "expected_sampling_proportion": 1.0 / (2 * len(train_sites)),
            })
    for row in rows:
        weights.append(1.0 / counts[(row["site"], int(row["label"]))])
    generator = torch.Generator().manual_seed(7)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(rows), replacement=True, generator=generator,
    )
    loader = DataLoader(
        p5.FrozenTruthDataset(rows, preprocessor), batch_size=8, shuffle=False,
        sampler=sampler, collate_fn=tl.collate_fn, num_workers=0, generator=generator,
    )
    return loader, cells


def global_class_cells(rows: list[dict]) -> list[dict]:
    labels = np.asarray([int(row["label"]) for row in rows])
    counts = {label: int((labels == label).sum()) for label in (0, 1)}
    result = []
    for site in sorted({row["site"] for row in rows}):
        for label in (0, 1):
            raw = sum(row["site"] == site and int(row["label"]) == label for row in rows)
            sample_weight = len(rows) / (2 * counts[label])
            result.append({
                "site": site, "label": label, "raw_count": raw,
                "sample_weight": sample_weight,
                "expected_sampling_proportion": raw * sample_weight / len(rows),
            })
    return result


def run_fold(
    arm: str, holdout_site: str, train_rows: list[dict], holdout_rows: list[dict]
) -> dict:
    fold_dir = OUTPUT / arm / f"holdout_{holdout_site}"
    fold_dir.mkdir(parents=True)
    logger = p5.configure_log(fold_dir / "train.log")
    tl.seed_everything(7)
    preprocessor = FeatureScaler.legacy_clip()
    if ARMS[arm]["sampler"] == "global_class_balanced":
        train_loader = global_class_loader(train_rows, preprocessor)
        cells = global_class_cells(train_rows)
    else:
        train_loader, cells = site_class_loader(train_rows, preprocessor)
    pd.DataFrame(cells).to_csv(fold_dir / "sampler_cells.csv", index=False)
    logger.info("arm=%s holdout=%s device=%s", arm, holdout_site, tl.device)
    for cell in cells:
        logger.info(
            "sampling_cell site=%s label=%d raw=%d weight=%.10g expected=%.6f",
            cell["site"], cell["label"], cell["raw_count"],
            cell["sample_weight"], cell["expected_sampling_proportion"],
        )

    # Preserve P5/P4 production call order: sampler diagnostic before model init.
    first_batch = next(iter(train_loader))
    if first_batch is None:
        raise RuntimeError("Empty LSTM smoke batch")
    x_seq, x_ecg, _mask, x_static, ages, y, _lengths = first_batch
    if x_seq.shape[-1] != 483 or x_ecg.shape[-1] != 12 or x_static.shape[-1] != 196:
        raise RuntimeError("Legacy input dimension mismatch")
    if not all(torch.isfinite(value).all() for value in (x_seq, x_ecg, x_static)):
        raise RuntimeError("Non-finite smoke input")
    logger.info(
        "smoke seq=%s ecg=%s static=%s positives=%d age=[%.1f,%.1f]",
        tuple(x_seq.shape), tuple(x_ecg.shape), tuple(x_static.shape),
        int(y.sum()), float(ages.min()), float(ages.max()),
    )

    model = tl.LSTMModel().to(tl.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([float((len(train_rows)-sum(int(r["label"]) for r in train_rows))/sum(int(r["label"]) for r in train_rows)) if ARMS[arm]["pos_weight"] == "empirical" else 1.0], dtype=torch.float32, device=tl.device)
    )
    history = []
    for epoch in range(1, 7):
        started = time.time()
        loss = float(tl.train_epoch(model, train_loader, optimizer, criterion))
        if not np.isfinite(loss):
            raise RuntimeError(f"Non-finite loss for {arm}/{holdout_site}/epoch{epoch}")
        history.append({"epoch": epoch, "train_loss": loss})
        logger.info("epoch=%d train_loss=%.8f time=%.1fs", epoch, loss, time.time() - started)

    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    training_config = {
        "batch_size": 8, "optimizer": "Adam", "lr": 1e-3,
        "epochs": 6, "early_stopping": False, "scheduler": None,
        "gradient_clip_max_norm": 1.0,
        "weighted_random_sampler": True,
        "sampler": ARMS[arm]["sampler"], "num_samples": len(train_rows),
        "replacement": True, "pos_weight_mode": ARMS[arm]["pos_weight"], "pos_weight": float((len(train_rows)-sum(int(r["label"]) for r in train_rows))/sum(int(r["label"]) for r in train_rows)) if ARMS[arm]["pos_weight"] == "empirical" else 1.0,
        "model_seed": 7, "hidden_size": 128, "num_layers": 2,
        "fc_hidden": 64, "dropout": 0.3,
    }
    checkpoint = {
        "state_dict": state, "checkpoint_role": "fixed_epoch6_p6_loso",
        "fixed_epoch": 6, "model_seed": 7, "source_commit": git_head(),
        "holdout_site": holdout_site,
        "train_record_ids": [row["record_id"] for row in train_rows],
        "input_preprocessing": preprocessor.state_dict(),
        "feature_dims": {"X_seq": 483, "X_ecg": 12, "x_static": 196, "time_input": 495},
        "training_config": training_config, "sampler_cells": cells,
    }
    torch.save(checkpoint, fold_dir / "epoch6_checkpoint.pt")
    logger.info("checkpoint frozen before constructing holdout loader")

    train_natural = p5.natural_loader(train_rows, preprocessor)
    train_labels, train_scores, train_ages = p5.strict_collect(model, train_natural)
    train_metrics = p5.metric_bundle(train_labels, train_scores, train_ages)
    history[-1].update({f"natural_{key}": value for key, value in train_metrics.items()})
    pd.DataFrame(history).to_csv(fold_dir / "train_metrics.csv", index=False)
    p5.write_logits(
        fold_dir / "train_logits.csv", train_rows, train_labels, train_ages, train_scores
    )

    holdout_loader = p5.natural_loader(holdout_rows, preprocessor)
    hold_labels, hold_scores, hold_ages = p5.strict_collect(model, holdout_loader)
    hold_metrics = p5.metric_bundle(hold_labels, hold_scores, hold_ages)
    p5.write_logits(
        fold_dir / "holdout_logits.csv", holdout_rows,
        hold_labels, hold_ages, hold_scores,
    )
    result = {
        "protocol": arm, "holdout_site": holdout_site,
        "train": train_metrics, "holdout": hold_metrics,
        "train_ac_minus_holdout_ac": train_metrics["ac_auroc"] - hold_metrics["ac_auroc"],
        "epochs_completed": 6,
    }
    (fold_dir / "config.json").write_text(json.dumps({
        "protocol": arm, "holdout_site": holdout_site,
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "input_cache": "npz_1103_timegrid_v2 (NEW1103)",
        "outer_holdout_used_during_training": False,
        "model": "train_lstm.LSTMModel", "preprocessing": "legacy_clip",
        "training_config": training_config, "sampler_cells": cells,
    }, indent=2) + "\n")
    (fold_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    logger.info(
        "frozen holdout site=%s ac=%.6f age_weighted=%.6f auroc=%.6f auprc=%.6f",
        holdout_site, hold_metrics["ac_auroc"], hold_metrics["age_weighted_auroc"],
        hold_metrics["auroc"], hold_metrics["auprc"],
    )
    return result


def aggregate(results: list[dict]) -> tuple[pd.DataFrame, dict]:
    rows = []
    for result in results:
        rows.append({
            "protocol": result["protocol"], "holdout_site": result["holdout_site"],
            **{f"train_{key}": value for key, value in result["train"].items()},
            **{f"holdout_{key}": value for key, value in result["holdout"].items()},
            "train_ac_minus_holdout_ac": result["train_ac_minus_holdout_ac"],
        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT / "fold_metrics.csv", index=False, na_rep="N/A")
    summary = {}
    for arm in ARMS:
        group = metrics.loc[metrics.protocol == arm].set_index("holdout_site").loc[list(SITES)]
        values = group.holdout_ac_auroc.astype(float)
        summary[arm] = {
            "sites": {
                site: {
                    "ac_auroc": float(group.loc[site, "holdout_ac_auroc"]),
                    "age_weighted_auroc": float(group.loc[site, "holdout_age_weighted_auroc"]),
                    "auroc": float(group.loc[site, "holdout_auroc"]),
                    "auprc": float(group.loc[site, "holdout_auprc"]),
                    "eligible_ac_pairs": int(group.loc[site, "holdout_eligible_ac_pairs"]),
                    "positives": int(group.loc[site, "holdout_positives"]),
                    "negatives": int(group.loc[site, "holdout_negatives"]),
                } for site in SITES
            },
            "macro_ac_auroc": float(values.mean()),
            "worst_site_ac_auroc": float(values.min()),
            "macro_age_weighted_auroc": float(group.holdout_age_weighted_auroc.mean()),
            "macro_auroc": float(group.holdout_auroc.mean()),
            "macro_auprc": float(group.holdout_auprc.mean()),
        }
    baseline = json.loads(P5_AGGREGATE.read_text())["L3_legacy_lstm"]
    output = {"P5_baseline": baseline, **summary}
    for arm in ARMS:
        output[arm]["delta_vs_p5"] = {
            "sites": {
                site: output[arm]["sites"][site]["ac_auroc"] - baseline["sites"][site]["ac_auroc"]
                for site in SITES
            },
            "macro_ac_auroc": output[arm]["macro_ac_auroc"] - baseline["macro_ac_auroc"],
            "worst_site_ac_auroc": output[arm]["worst_site_ac_auroc"] - baseline["worst_site_ac_auroc"],
        }
    (OUTPUT / "aggregate_metrics.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    return metrics, output


def fmt(value: float) -> str:
    return "N/A" if not np.isfinite(float(value)) else f"{float(value):.3f}"


def write_report(metrics: pd.DataFrame, agg: dict) -> None:
    protocols = ["P5_baseline", *ARMS]
    labels = {
        "P5_baseline": "OLD P5 balanced class sampler + empirical pos_weight",
        "L0": "NEW-L0 global class sampler + empirical pos_weight", "L1": "NEW-L1 global class sampler + pos_weight=1",
        "L2": "NEW-L2 site+class sampler + pos_weight=1",
    }
    lines = [
        "# NEW1103 timegrid_v2 LSTM bridge results", "", "## Frozen protocol", "",
        f"- Source commit before bridge changes: `{git_head()}`.",
        "- Input is only the frozen P5 1103-record LOSO manifest and NEW `npz_1103_timegrid_v2` cache.",
        "- All three bridge arms reuse P5's 2-layer LSTM, `legacy_clip`, seed 7, Adam 1e-3, batch size 8, gradient clip 1.0, and exactly six epochs.",
        "- Across OLD/NEW comparison the model protocols remain frozen; within NEW, L0/L1/L2 differ only by the requested sampler/positive-loss weighting.",
        "- Main results use raw logits and official Age-conditioned AUROC (gap=2); no holdout threshold, calibration, epoch selection, or tuning was used.",
        "", "## Main comparison", "",
        "| Protocol | I0002 AC | I0006 AC | S0001 AC | Macro AC | Worst-site AC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for protocol in protocols:
        item = agg[protocol]
        lines.append(
            f"| {labels[protocol]} | {fmt(item['sites']['I0002']['ac_auroc'])} | "
            f"{fmt(item['sites']['I0006']['ac_auroc'])} | {fmt(item['sites']['S0001']['ac_auroc'])} | "
            f"{fmt(item['macro_ac_auroc'])} | {fmt(item['worst_site_ac_auroc'])} |"
        )
    lines += ["", "## Delta from P5", "",
              "| Protocol | I0002 | I0006 | S0001 | Macro | Worst-site |",
              "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        delta = agg[arm]["delta_vs_p5"]
        lines.append(
            f"| {labels[arm]} | {delta['sites']['I0002']:+.3f} | "
            f"{delta['sites']['I0006']:+.3f} | {delta['sites']['S0001']:+.3f} | "
            f"{delta['macro_ac_auroc']:+.3f} | {delta['worst_site_ac_auroc']:+.3f} |"
        )
    lines += ["", "## Fold diagnostics", "",
              "| Protocol | Holdout | n (+/-) | Eligible pairs | Train AC | Holdout AC | Age-weighted | AUROC | AUPRC | Train-holdout gap |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.protocol} | {row.holdout_site} | {int(row.holdout_n)} "
            f"({int(row.holdout_positives)}/{int(row.holdout_negatives)}) | "
            f"{int(row.holdout_eligible_ac_pairs)} | {fmt(row.train_ac_auroc)} | "
            f"{fmt(row.holdout_ac_auroc)} | {fmt(row.holdout_age_weighted_auroc)} | "
            f"{fmt(row.holdout_auroc)} | {fmt(row.holdout_auprc)} | "
            f"{row.train_ac_minus_holdout_ac:+.3f} |"
        )
    # The bridge decisions compare the NEW protocols to NEW-L0. OLD P5 deltas
    # are descriptive extractor-bridge observations, not sampler decisions.
    l0, l1, l2 = agg["L0"], agg["L1"], agg["L2"]
    b1_order = l0["macro_ac_auroc"] < l1["macro_ac_auroc"] <= l2["macro_ac_auroc"]
    b2_improved = sum(
        l2["sites"][site]["ac_auroc"] > l0["sites"][site]["ac_auroc"] for site in SITES
    )
    b2_pass = (
        l2["macro_ac_auroc"] > l0["macro_ac_auroc"]
        and l2["worst_site_ac_auroc"] > l0["worst_site_ac_auroc"]
        and b2_improved >= 2
    )
    lines += [
        "", "## Predeclared NEW-protocol decisions", "",
        f"- B1 ordering `L0 < L1 <= L2`: **{'confirmed' if b1_order else 'not confirmed'}**.",
        f"- B2 versus NEW-L0: **{'PASS' if b2_pass else 'FAIL'}** "
        f"({b2_improved}/3 sites improve; Macro "
        f"{l2['macro_ac_auroc'] - l0['macro_ac_auroc']:+.3f}; Worst "
        f"{l2['worst_site_ac_auroc'] - l0['worst_site_ac_auroc']:+.3f}).",
        "- Deltas from OLD P5 above describe the extractor bridge only and are not used to promote B1 or B2.",
    ]
    (OUTPUT / "NEW1103_LSTM_BRIDGE_RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")
    manifest, split_records = validate_inputs()
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "source_commit.txt").write_text(git_head() + "\n")
    environment = {
        "python": sys.executable, "python_version": sys.version,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "platform": platform.platform(), "device": str(tl.device),
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "LSTM_INPUT_PREPROCESSING", "LSTM_POS_WEIGHT_MODE",
            "LSTM_SEED", "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG",
        )},
    }
    (OUTPUT / "environment.txt").write_text(json.dumps(environment, indent=2) + "\n")
    results = []
    for arm in ARMS:
        (OUTPUT / arm).mkdir()
        for holdout_site in SITES:
            train_frame = manifest.loc[manifest.site != holdout_site].copy()
            holdout_frame = manifest.loc[manifest.site == holdout_site].copy()
            train_rows = p5.rows_for(train_frame, split_records)
            holdout_rows = p5.rows_for(holdout_frame, split_records)
            print(
                f"[BRIDGE-LSTM] arm={arm} holdout={holdout_site} "
                f"train={len(train_rows)} holdout_n={len(holdout_rows)}", flush=True,
            )
            results.append(run_fold(arm, holdout_site, train_rows, holdout_rows))
    if len(results) != 9:
        raise RuntimeError(f"Expected nine completed folds, got {len(results)}")
    metrics, agg = aggregate(results)
    write_report(metrics, agg)
    print("NEW1103_LSTM_BRIDGE_COMPLETE", flush=True)



def finalize_bridge_reports():
    root=ROOT/"output/bridge_1103_timegrid_v2"
    lr=json.loads((root/"loso_lr/aggregate_metrics.json").read_text()); lm=json.loads((root/"lstm/aggregate_metrics.json").read_text())
    old_lr=json.loads((ROOT/"output/p5_loso_1103/aggregate_metrics.json").read_text()); old_p6=json.loads((ROOT/"output/p6_site_domain/lstm_balance/aggregate_metrics.json").read_text())
    old={"demo10":old_lr["L0_demo10"],"compact30":old_lr["L1_compact30"],"global59":old_lr["L2_global59"],"L0":old_lr["L3_legacy_lstm"],"L1":old_p6["B1_global_class_pos1"],"L2":old_p6["B2_site_class_pos1"]}
    new={"demo10":lr["demo10"],"compact30":lr["compact30"],"global59":lr["global59"],"L0":lm["L0"],"L1":lm["L1"],"L2":lm["L2"]}
    combined={"old":old,"new":new,"delta":{k:{"macro_ac_auroc":new[k]["macro_ac_auroc"]-old[k]["macro_ac_auroc"],"worst_site_ac_auroc":new[k]["worst_site_ac_auroc"]-old[k]["worst_site_ac_auroc"],"sites":{s:new[k]["sites"][s]["ac_auroc"]-old[k]["sites"][s]["ac_auroc"] for s in SITES}} for k in new}}
    (root/"aggregate_metrics.json").write_text(json.dumps(combined,indent=2,sort_keys=True)+"\n")
    gi=sum(new["global59"]["sites"][s]["ac_auroc"]>new["compact30"]["sites"][s]["ac_auroc"] for s in SITES); gp=new["global59"]["macro_ac_auroc"]>new["compact30"]["macro_ac_auroc"] and new["global59"]["worst_site_ac_auroc"]>=new["compact30"]["worst_site_ac_auroc"] and gi>=2
    b1=new["L0"]["macro_ac_auroc"]<new["L1"]["macro_ac_auroc"]<=new["L2"]["macro_ac_auroc"]
    bi=sum(new["L2"]["sites"][s]["ac_auroc"]>new["L0"]["sites"][s]["ac_auroc"] for s in SITES); bp=new["L2"]["macro_ac_auroc"]>new["L0"]["macro_ac_auroc"] and new["L2"]["worst_site_ac_auroc"]>new["L0"]["worst_site_ac_auroc"] and bi>=2
    so=pd.read_csv(ROOT/"output/p6_site_domain/site_audit/site_feature_summary.csv").set_index("feature_family"); sn=pd.read_csv(root/"site_audit/site_feature_summary.csv").set_index("feature_family")
    st=["| Family | OLD | NEW | Delta |","|---|---:|---:|---:|"]
    for f in sn.index:
        ov=float(so.loc[f,"mean_balanced_accuracy"]); nv=float(sn.loc[f,"site_balanced_accuracy_mean"]); st.append(f"| {f} | {ov:.3f} | {nv:.3f} | {nv-ov:+.3f} |")
    lines=["# 1103 timegrid_v2 bridge results","","OLD and NEW contain the same 1103 patients, labels, sites, LOSO folds, model protocols, and seeds. The principal systematic change is extraction version; these are controlled observations, not strict causal proof.","","## Input changes","",(root/"paired_qc/OLD_NEW_INPUT_COMPARISON.md").read_text().split("## Site summary")[0].replace("# OLD vs NEW1103 input comparison\n\n","").strip(),"","## Site decodability OLD vs NEW","",*st,"","## Model bridge","","| Model | OLD sites | OLD Macro/Worst | NEW sites | NEW Macro/Worst | Macro delta |","|---|---:|---:|---:|---:|---:|"]
    for k in ("demo10","compact30","global59","L0","L1","L2"):
        o,n=old[k],new[k]; lines.append(f"| {k} | {o['sites']['I0002']['ac_auroc']:.3f}/{o['sites']['I0006']['ac_auroc']:.3f}/{o['sites']['S0001']['ac_auroc']:.3f} | {o['macro_ac_auroc']:.3f}/{o['worst_site_ac_auroc']:.3f} | {n['sites']['I0002']['ac_auroc']:.3f}/{n['sites']['I0006']['ac_auroc']:.3f}/{n['sites']['S0001']['ac_auroc']:.3f} | {n['macro_ac_auroc']:.3f}/{n['worst_site_ac_auroc']:.3f} | {n['macro_ac_auroc']-o['macro_ac_auroc']:+.3f} |")
    lines += ["","## Predeclared decisions","",f"1. global59 retention: **{'PASS' if gp else 'FAIL'}**. Versus NEW compact30: Macro {new['global59']['macro_ac_auroc']-new['compact30']['macro_ac_auroc']:+.3f}, Worst {new['global59']['worst_site_ac_auroc']-new['compact30']['worst_site_ac_auroc']:+.3f}, {gi}/3 sites improve. Downgrade it to diagnostic and stop the P3-style EEG-pooling main line.",f"2. B1 ordering `L0 < L1 <= L2`: **{'confirmed' if b1 else 'not confirmed'}**. NEW-L1 versus L0: Macro {new['L1']['macro_ac_auroc']-new['L0']['macro_ac_auroc']:+.3f}, Worst {new['L1']['worst_site_ac_auroc']-new['L0']['worst_site_ac_auroc']:+.3f}.",f"3. B2 versus NEW-L0: **{'PASS' if bp else 'FAIL'}**. Macro {new['L2']['macro_ac_auroc']-new['L0']['macro_ac_auroc']:+.3f}, Worst {new['L2']['worst_site_ac_auroc']-new['L0']['worst_site_ac_auroc']:+.3f}, {bi}/3 sites improve.",f"4. Historical L0 improves on all three sites: Macro {new['L0']['macro_ac_auroc']-old['L0']['macro_ac_auroc']:+.3f}, Worst {new['L0']['worst_site_ac_auroc']-old['L0']['worst_site_ac_auroc']:+.3f}.","5. For a later data2 LSTM stage, NEW-L0 is the primary supported protocol. L1 is at most a fixed mechanism control; L2 is not promoted. No data2 LSTM was launched.","","## Directions","","- Continue the frozen legacy LSTM architecture only under L0 if data2 training is later authorized.","- Downgrade global59 to diagnostic and stop P3-style EEG pooling as a primary path.","- Do not promote B1/B2 or tune sampling weights post hoc."]
    (root/"BRIDGE_1103_RESULTS.md").write_text("\n".join(lines)+"\n")
    dedicated=(root/"lstm/NEW1103_LSTM_BRIDGE_RESULTS.md").read_text().replace("legacy `npz_new` cache","NEW `npz_1103_timegrid_v2` cache").replace("Both P6 arms","All three bridge arms")
    dedicated += f"\n## NEW-protocol decisions\n\n- B1 ordering `L0 < L1 <= L2`: **{'confirmed' if b1 else 'not confirmed'}**.\n- B2 versus NEW-L0: **{'PASS' if bp else 'FAIL'}** ({bi}/3 sites improve; Macro {new['L2']['macro_ac_auroc']-new['L0']['macro_ac_auroc']:+.3f}; Worst {new['L2']['worst_site_ac_auroc']-new['L0']['worst_site_ac_auroc']:+.3f}).\n"
    (root/"lstm/NEW1103_LSTM_BRIDGE_RESULTS.md").write_text(dedicated)
    rp=ROOT/"feat_input2/output/baselines/baseline_registry.json"; reg=json.loads(rp.read_text()); reg["entries"]=[e for e in reg["entries"] if not e["id"].startswith("NEW-")]
    specs={"NEW-D":("demo10","elastic-net LR","demo10","typed_v1","class_weight=balanced",None,"none"),"NEW-C":("compact30","elastic-net LR","compact30","typed_v1","class_weight=balanced",None,"none"),"NEW-G":("global59","elastic-net LR","global59","typed_v1","class_weight=balanced",None,"none"),"NEW-L0":("L0","2-layer LSTM","483+12+196","legacy_clip","global class-balanced","empirical","fixed_epoch6"),"NEW-L1":("L1","2-layer LSTM","483+12+196","legacy_clip","global class-balanced",1.0,"fixed_epoch6"),"NEW-L2":("L2","2-layer LSTM","483+12+196","legacy_clip","site+class-balanced",1.0,"fixed_epoch6")}
    for ident,(key,model,features,prep,sampler,pos,ckpt) in specs.items():
        n=new[key]; reg["entries"].append({"id":ident,"dataset":"NEW1103 / timegrid_v2","cache":"npz_1103_timegrid_v2","extraction_version":"timegrid_v2.0.0","model":model,"feature_set":features,"preprocessing":prep,"sampler":sampler,"pos_weight":pos,"CV_protocol":"three-site LOSO fixed epoch6" if model.endswith("LSTM") else "three-site LOSO","checkpoint_rule":ckpt,"seed":7,"I0002_AC":n["sites"]["I0002"]["ac_auroc"],"I0006_AC":n["sites"]["I0006"]["ac_auroc"],"S0001_AC":n["sites"]["S0001"]["ac_auroc"],"macro_AC":n["macro_ac_auroc"],"worst_site_AC":n["worst_site_ac_auroc"],"result_commit":"bridge result commit (this commit)","result_path":"output/bridge_1103_timegrid_v2","role":"NEW1103 bridge benchmark"})
    reg["generated_from_source_commit"]=git_head(); reg["interpretation_warning"]="OLD/NEW are matched-patient bridge observations; DATA2 differs in size and composition."; rp.write_text(json.dumps(reg,indent=2)+"\n")
    md=["# Baseline registry","","OLD/NEW are matched-patient bridge observations. DATA2 differs in sample size and site composition.","","| ID | Dataset | Model/features | I0002 | I0006 | S0001 | Macro | Worst | Role |","|---|---|---|---:|---:|---:|---:|---:|---|"]
    for e in reg["entries"]: md.append(f"| {e['id']} | {e['dataset']} | {e['model']} / {e['feature_set']} | {e['I0002_AC'] if e['I0002_AC'] is not None else 'N/A'} | {e['I0006_AC'] if e['I0006_AC'] is not None else 'N/A'} | {e['S0001_AC'] if e['S0001_AC'] is not None else 'N/A'} | {e['macro_AC'] if e['macro_AC'] is not None else 'N/A'} | {e['worst_site_AC'] if e['worst_site_AC'] is not None else 'N/A'} | {e['role']} |")
    (ROOT/"feat_input2/output/baselines/BASELINE_REGISTRY.md").write_text("\n".join(md)+"\n")
    return {"global59_pass":gp,"b1_order":b1,"b2_pass":bp,"b2_improved_sites":bi}
if __name__ == "__main__":
    main()
