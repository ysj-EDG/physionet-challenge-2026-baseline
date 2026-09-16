#!/usr/bin/env python3
"""Run DATA2-L0 historical legacy LSTM with frozen three-site LOSO."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ["LSTM_INPUT_PREPROCESSING"] = "legacy_clip"
os.environ["LSTM_POS_WEIGHT_MODE"] = "empirical"
os.environ["LSTM_SEED"] = "7"
os.environ.setdefault("PYTHONHASHSEED", "7")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "feat_input2/output/data2_lowcost/data2_manifest.csv"
CACHE = ROOT / "npz_data2"
OUTPUT = Path(os.environ.get("DATA2_L0_OUTPUT_DIR", ROOT / "output/data2_l0_lstm"))
SPLITS = OUTPUT / "splits"
SITES = ("I0002", "I0006", "S0001")

os.environ["P5_MANIFEST"] = str(MANIFEST)
os.environ["P5_OUTPUT_DIR"] = str(OUTPUT / "_unused_p5_output")
os.environ["LSTM_CACHE_DIR"] = str(CACHE)
os.environ["P5_CACHE_LAYOUT"] = "site"

import train_lstm as tl
from feature_scaling import FeatureScaler
from feat_input.feat_mody import run_loso_1103 as p5


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        preflight = OUTPUT / "preflight.json"
        if preflight.is_file():
            return str(json.loads(preflight.read_text())["head"])
        raise RuntimeError("Git metadata unavailable and no frozen preflight HEAD found")


def load_manifest() -> pd.DataFrame:
    frame = pd.read_csv(MANIFEST, dtype={"record_id": str, "patient_id": str, "site": str})
    expected = {"I0002": (319, 52, 267), "I0006": (1142, 112, 1030), "S0001": (5139, 334, 4805)}
    if len(frame) != 6600 or frame.record_id.nunique() != 6600 or frame.patient_id.nunique() != 6600:
        raise RuntimeError("Data2 manifest must contain 6600 unique records and patients")
    if set(frame.site) != set(SITES) or set(frame.label) != {0, 1}:
        raise RuntimeError("Incomplete site or binary labels")
    if set(frame.extraction_version) != {"timegrid_v2.0.0"}:
        raise RuntimeError("Mixed extraction versions")
    for site, (n, pos, neg) in expected.items():
        part = frame.loc[frame.site == site]
        actual = (len(part), int(part.label.sum()), int((part.label == 0).sum()))
        if actual != (n, pos, neg):
            raise RuntimeError(f"Unexpected {site} counts: {actual}")
    missing = []
    for row in frame.itertuples(index=False):
        path = CACHE / row.site / f"{row.record_id}.npz"
        if not path.is_file():
            missing.append(str(path))
    if missing or len(list(CACHE.glob("*/*.npz"))) != 6600:
        raise RuntimeError(f"Cache coverage failure: missing={missing[:3]}")
    return frame


def create_split(frame: pd.DataFrame, holdout_site: str) -> pd.DataFrame:
    pieces = []
    for site in SITES:
        part = frame.loc[frame.site == site].copy()
        if site == holdout_site:
            part["role"] = "outer_holdout"
            pieces.append(part)
            continue
        inner_idx, val_idx = train_test_split(
            np.arange(len(part)), test_size=0.20, random_state=7,
            shuffle=True, stratify=part.label.to_numpy(),
        )
        role = np.full(len(part), "inner_train", dtype=object)
        role[val_idx] = "inner_val"
        part["role"] = role
        pieces.append(part)
    split = pd.concat(pieces, ignore_index=True)
    if split.record_id.duplicated().any() or split.patient_id.duplicated().any():
        raise RuntimeError(f"Duplicate record/patient in split {holdout_site}")
    roles = {role: set(split.loc[split.role == role].patient_id) for role in split.role.unique()}
    if any(roles[a] & roles[b] for a in roles for b in roles if a < b):
        raise RuntimeError(f"Patient leakage in split {holdout_site}")
    for site in SITES:
        if site == holdout_site:
            continue
        for role in ("inner_train", "inner_val"):
            cell = split.loc[(split.site == site) & (split.role == role)]
            if set(cell.label) != {0, 1}:
                raise RuntimeError(f"Missing class in {holdout_site}/{site}/{role}")
            if role == "inner_val" and p5.eligible_pairs(cell.label, cell.age) <= 0:
                raise RuntimeError(f"No eligible AC pair in {holdout_site}/{site}/inner_val")
    return split[["record_id", "patient_id", "site", "label", "age", "role"]]


def rows(frame: pd.DataFrame) -> list[dict]:
    return [
        {"record_id": r.record_id, "patient_id": r.patient_id, "site": r.site,
         "label": int(r.label), "age": float(r.age), "npz_partition": r.site}
        for r in frame.itertuples(index=False)
    ]


def configure_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"data2_l0_{path.parent.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.handlers = [handler, logging.StreamHandler(sys.stdout)]
    return logger


def collect_metrics(model, data_rows: list[dict], preprocessor):
    loader = p5.natural_loader(data_rows, preprocessor)
    labels, scores, ages = p5.strict_collect(model, loader)
    return p5.metric_bundle(labels, scores, ages), labels, scores, ages


def clone_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def checkpoint(state, role, holdout_site, epoch, stop_epoch, best_score, train_rows, val_rows, pos_weight, preprocessor):
    return {
        "state_dict": state, "checkpoint_role": role, "epoch": int(epoch),
        "stop_epoch": int(stop_epoch), "best_seen_site_macro_ac": float(best_score),
        "model_seed": 7, "source_commit": git_head(), "holdout_site": holdout_site,
        "inner_train_record_ids": [r["record_id"] for r in train_rows],
        "inner_val_record_ids": [r["record_id"] for r in val_rows],
        "outer_holdout_used_during_training": False,
        "input_preprocessing": preprocessor.state_dict(),
        "feature_dims": {"X_seq": 483, "X_ecg": 12, "x_static": 196, "time_input": 495},
        "training_config": {
            "model": "train_lstm.LSTMModel", "hidden_size": 128, "num_layers": 2,
            "fc_hidden": 64, "dropout": 0.3, "batch_size": 8,
            "optimizer": "Adam", "lr": 1e-3, "max_epochs": 80,
            "early_stopping_patience": 15, "scheduler": "ReduceLROnPlateau",
            "scheduler_mode": "max", "scheduler_factor": 0.5,
            "scheduler_patience": 8, "scheduler_monitor": "seen_site_validation_macro_ac",
            "gradient_clip_max_norm": 1.0, "sampler": "global_class_balanced",
            "replacement": True, "num_samples": len(train_rows),
            "pos_weight_mode": "empirical", "pos_weight": float(pos_weight),
            "preprocessing": "legacy_clip", "seed": 7,
        },
    }


def prepare() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")
    frame = load_manifest()
    SPLITS.mkdir(parents=True)
    summary = {}
    for holdout in SITES:
        split = create_split(frame, holdout)
        split.to_csv(SPLITS / f"outer_{holdout}.csv", index=False)
        summary[holdout] = split.groupby(["role", "site", "label"]).size().to_dict()

    # One-batch forward/backward smoke test in this independent process.
    split = pd.read_csv(SPLITS / "outer_I0002.csv", dtype={"record_id": str, "patient_id": str})
    train_rows = rows(split.loc[split.role == "inner_train"])
    tl.seed_everything(7)
    preprocessor = FeatureScaler.legacy_clip()
    loader = p5.balanced_loader(train_rows, preprocessor)
    batch = next(iter(loader))
    if batch is None:
        raise RuntimeError("Empty smoke batch")
    x_seq, x_ecg, mask, x_static, _ages, y, lengths = batch
    if (x_seq.shape[-1], x_ecg.shape[-1], x_static.shape[-1]) != (483, 12, 196):
        raise RuntimeError("Input dimension mismatch")
    if not all(torch.isfinite(x).all() for x in (x_seq, x_ecg, x_static)):
        raise RuntimeError("Non-finite smoke inputs")
    model = tl.LSTMModel().to(tl.device)
    n_pos = sum(int(r["label"]) for r in train_rows)
    pos_weight = (len(train_rows) - n_pos) / n_pos
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=tl.device))
    logits = model(x_seq.to(tl.device), x_ecg.to(tl.device), mask.to(tl.device), x_static.to(tl.device), lengths)
    loss = criterion(logits.view(-1), y.to(tl.device).float().view(-1))
    loss.backward()
    if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise RuntimeError("Non-finite smoke forward/backward")

    env = {
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "head": git_head(), "python": sys.executable, "python_version": sys.version,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "platform": platform.platform(), "scheduler_monitor": "seen_site_validation_macro_ac",
        "smoke_loss": float(loss.detach().cpu()),
    }
    (OUTPUT / "preflight.json").write_text(json.dumps(env, indent=2) + "\n")
    audit = [
        "# DATA2-L0 reuse audit", "", f"- Branch: `{env['branch']}`.", f"- Starting HEAD: `{env['head']}`.",
        "- Reuses `train_lstm.LSTMModel`, `train_epoch`, `PSGDataset`, `collate_fn`, and `legacy_clip`.",
        "- Reuses P5 `FrozenTruthDataset`, global class-balanced loader, natural loader, raw-logit collection, and official metric bundle.",
        "- Reuses the frozen data2 low-cost manifest; no EDF, label reconstruction, feature extraction, or NPZ write occurred.",
        "- Thin runner adds only per-seen-site stratified inner validation, macro seen-site AC selection, checkpointing, and report aggregation.",
        "- Scheduler is historical `ReduceLROnPlateau(mode=max,factor=0.5,patience=8)` monitoring seen-site validation macro AC.",
        f"- Environment: Python `{sys.version.split()[0]}`, PyTorch `{torch.__version__}`, CUDA `{torch.version.cuda}`, device `{env['gpu']}`.",
        f"- Independent one-batch forward/backward smoke loss `{env['smoke_loss']:.6f}` was finite.",
    ]
    (OUTPUT / "REUSE_AUDIT.md").write_text("\n".join(audit) + "\n")
    print("DATA2_L0_PREPARE_PASS", json.dumps({k: {str(x): int(v) for x, v in val.items()} for k, val in summary.items()}))


def run_fold(holdout_site: str) -> None:
    split_path = SPLITS / f"outer_{holdout_site}.csv"
    if not split_path.is_file():
        raise FileNotFoundError("Run --prepare first")
    fold_dir = OUTPUT / holdout_site
    if fold_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {fold_dir}")
    fold_dir.mkdir()
    log = configure_logger(fold_dir / "train.log")
    split = pd.read_csv(split_path, dtype={"record_id": str, "patient_id": str})
    train_rows = rows(split.loc[split.role == "inner_train"])
    val_rows = rows(split.loc[split.role == "inner_val"])
    holdout_rows = rows(split.loc[split.role == "outer_holdout"])
    val_sites = tuple(site for site in SITES if site != holdout_site)
    val_by_site = {site: [r for r in val_rows if r["site"] == site] for site in val_sites}

    tl.seed_everything(7)
    preprocessor = FeatureScaler.legacy_clip()
    train_loader = p5.balanced_loader(train_rows, preprocessor)
    first_batch = next(iter(train_loader))
    if first_batch is None:
        raise RuntimeError("Empty first sampled batch")
    for tensor in (first_batch[0], first_batch[1], first_batch[3]):
        if not torch.isfinite(tensor).all():
            raise RuntimeError("Non-finite first batch")
    model = tl.LSTMModel().to(tl.device)
    n_pos = sum(int(r["label"]) for r in train_rows)
    n_neg = len(train_rows) - n_pos
    pos_weight = n_neg / n_pos
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=tl.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)

    best_score = -float("inf")
    best_state = None
    best_epoch = 0
    epoch6_state = None
    patience_counter = 0
    history = []
    stop_epoch = 0
    log.info("holdout=%s device=%s inner_train=%d inner_val=%d outer_holdout=%d pos_weight=%.8f", holdout_site, tl.device, len(train_rows), len(val_rows), len(holdout_rows), pos_weight)
    for epoch in range(1, 81):
        started = time.time()
        loss = float(tl.train_epoch(model, train_loader, optimizer, criterion))
        if not np.isfinite(loss):
            raise RuntimeError(f"Non-finite loss epoch={epoch}")
        train_metrics, *_ = collect_metrics(model, train_rows, preprocessor)
        site_metrics = {}
        for site in val_sites:
            site_metrics[site], *_ = collect_metrics(model, val_by_site[site], preprocessor)
            if site_metrics[site]["eligible_ac_pairs"] <= 0 or not np.isfinite(site_metrics[site]["ac_auroc"]):
                raise RuntimeError(f"Invalid validation AC for {site}")
        selection_score = float(np.mean([site_metrics[site]["ac_auroc"] for site in val_sites]))
        scheduler.step(selection_score)
        if epoch == 6:
            epoch6_state = clone_state(model)
        improved = best_state is None or selection_score > best_score
        if improved:
            best_score, best_state, best_epoch, patience_counter = selection_score, clone_state(model), epoch, 0
        else:
            patience_counter += 1
        row = {
            "epoch": epoch, "train_loss": loss,
            "train_natural_ac": train_metrics["ac_auroc"], "train_natural_auroc": train_metrics["auroc"], "train_natural_auprc": train_metrics["auprc"],
            "seen_site_macro_ac": selection_score, "lr": float(optimizer.param_groups[0]["lr"]),
            "is_best_checkpoint": bool(improved), "elapsed_seconds": time.time() - started,
        }
        for site in val_sites:
            for key in ("ac_auroc", "auroc", "auprc", "eligible_ac_pairs"):
                row[f"val_{site}_{key}"] = site_metrics[site][key]
        history.append(row)
        pd.DataFrame(history).to_csv(fold_dir / "metrics_by_epoch.csv", index=False)
        log.info("epoch=%d loss=%.7f train_ac=%.6f val_%s=%.6f val_%s=%.6f macro=%.6f lr=%.8g best=%s time=%.1fs", epoch, loss, train_metrics["ac_auroc"], val_sites[0], site_metrics[val_sites[0]]["ac_auroc"], val_sites[1], site_metrics[val_sites[1]]["ac_auroc"], selection_score, optimizer.param_groups[0]["lr"], improved, row["elapsed_seconds"])
        stop_epoch = epoch
        if patience_counter >= 15:
            log.info("early_stop epoch=%d best_epoch=%d best_macro=%.6f", epoch, best_epoch, best_score)
            break
    if best_state is None or epoch6_state is None:
        raise RuntimeError("Missing best or epoch6 state")

    best_payload = checkpoint(best_state, "best_seen_site_macro_ac", holdout_site, best_epoch, stop_epoch, best_score, train_rows, val_rows, pos_weight, preprocessor)
    epoch6_payload = checkpoint(epoch6_state, "fixed_epoch6_diagnostic_only", holdout_site, 6, stop_epoch, best_score, train_rows, val_rows, pos_weight, preprocessor)
    torch.save(best_payload, fold_dir / "best_checkpoint.pt")
    torch.save(epoch6_payload, fold_dir / "epoch6_checkpoint.pt")
    log.info("checkpoints frozen before outer holdout loader construction")

    evaluations = {}
    for role, state, logits_name in (
        ("best", best_state, "outer_holdout_logits.csv"),
        ("epoch6_diagnostic", epoch6_state, "outer_holdout_epoch6_logits.csv"),
    ):
        model.load_state_dict(state)
        train_metrics, *_ = collect_metrics(model, train_rows, preprocessor)
        validation = {}
        for site in val_sites:
            validation[site], *_ = collect_metrics(model, val_by_site[site], preprocessor)
        hold_metrics, labels, scores, ages = collect_metrics(model, holdout_rows, preprocessor)
        p5.write_logits(fold_dir / logits_name, holdout_rows, labels, ages, scores)
        val_macro = float(np.mean([validation[s]["ac_auroc"] for s in val_sites]))
        evaluations[role] = {
            "epoch": best_epoch if role == "best" else 6, "train": train_metrics,
            "validation_by_site": validation, "seen_site_macro_ac": val_macro,
            "outer_holdout": hold_metrics,
            "train_minus_val_macro_ac": train_metrics["ac_auroc"] - val_macro,
            "val_macro_minus_outer_ac": val_macro - hold_metrics["ac_auroc"],
        }
    result = {
        "protocol": "DATA2-L0", "holdout_site": holdout_site,
        "best_epoch": best_epoch, "stop_epoch": stop_epoch,
        "inner_train_n": len(train_rows), "inner_val_n": len(val_rows),
        "outer_holdout_n": len(holdout_rows), "val_sites": list(val_sites),
        "best": evaluations["best"], "epoch6_diagnostic": evaluations["epoch6_diagnostic"],
    }
    (fold_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (fold_dir / "config.json").write_text(json.dumps({
        "protocol": "DATA2-L0 historical legacy LSTM", "holdout_site": holdout_site,
        "manifest": str(MANIFEST.relative_to(ROOT)), "split": str(split_path.relative_to(ROOT)),
        "cache": str(CACHE), "outer_holdout_used_during_training": False,
        "selection_metric": "unweighted mean AC across two seen-site validation subsets",
        "training_config": best_payload["training_config"],
    }, indent=2) + "\n")
    log.info("complete best_epoch=%d stop_epoch=%d outer_ac=%.6f epoch6_outer_ac=%.6f", best_epoch, stop_epoch, evaluations["best"]["outer_holdout"]["ac_auroc"], evaluations["epoch6_diagnostic"]["outer_holdout"]["ac_auroc"])


def finalize() -> None:
    results = []
    for site in SITES:
        path = OUTPUT / site / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        results.append(json.loads(path.read_text()))
    rows_out = []
    for result in results:
        best, e6 = result["best"], result["epoch6_diagnostic"]
        row = {
            "holdout_site": result["holdout_site"], "inner_train_n": result["inner_train_n"],
            "inner_val_n": result["inner_val_n"], "outer_holdout_n": result["outer_holdout_n"],
            "best_epoch": result["best_epoch"], "stop_epoch": result["stop_epoch"],
            "train_ac": best["train"]["ac_auroc"], "seen_site_val_macro_ac": best["seen_site_macro_ac"],
            "outer_ac": best["outer_holdout"]["ac_auroc"], "outer_age_weighted_auroc": best["outer_holdout"]["age_weighted_auroc"],
            "outer_auroc": best["outer_holdout"]["auroc"], "outer_auprc": best["outer_holdout"]["auprc"],
            "outer_positives": best["outer_holdout"]["positives"], "outer_negatives": best["outer_holdout"]["negatives"],
            "outer_eligible_ac_pairs": best["outer_holdout"]["eligible_ac_pairs"],
            "train_minus_val_macro_ac": best["train_minus_val_macro_ac"], "val_macro_minus_outer_ac": best["val_macro_minus_outer_ac"],
            "epoch6_outer_ac": e6["outer_holdout"]["ac_auroc"], "best_minus_epoch6_outer_ac": best["outer_holdout"]["ac_auroc"] - e6["outer_holdout"]["ac_auroc"],
        }
        for site, metrics in best["validation_by_site"].items():
            row[f"val_{site}_ac"] = metrics["ac_auroc"]
        rows_out.append(row)
    folds = pd.DataFrame(rows_out)
    folds.to_csv(OUTPUT / "fold_metrics.csv", index=False)
    ordered = folds.set_index("holdout_site").loc[list(SITES)]
    summary = {
        "protocol": "DATA2-L0", "sites": {
            site: {"ac_auroc": float(ordered.loc[site, "outer_ac"]), "age_weighted_auroc": float(ordered.loc[site, "outer_age_weighted_auroc"]), "auroc": float(ordered.loc[site, "outer_auroc"]), "auprc": float(ordered.loc[site, "outer_auprc"]), "epoch6_ac_auroc": float(ordered.loc[site, "epoch6_outer_ac"]), "best_epoch": int(ordered.loc[site, "best_epoch"]), "stop_epoch": int(ordered.loc[site, "stop_epoch"])} for site in SITES
        },
        "macro_ac_auroc": float(ordered.outer_ac.mean()), "worst_site_ac_auroc": float(ordered.outer_ac.min()),
        "macro_age_weighted_auroc": float(ordered.outer_age_weighted_auroc.mean()),
        "macro_auroc": float(ordered.outer_auroc.mean()), "macro_auprc": float(ordered.outer_auprc.mean()),
        "epoch6_macro_ac_auroc": float(ordered.epoch6_outer_ac.mean()), "epoch6_worst_site_ac_auroc": float(ordered.epoch6_outer_ac.min()),
    }
    compact = {"I0002": .748, "I0006": .681, "S0001": .654}
    improved = sum(summary["sites"][s]["ac_auroc"] >= compact[s] for s in SITES)
    success = summary["macro_ac_auroc"] > .694 and summary["worst_site_ac_auroc"] >= .654 and improved >= 2
    robustness = abs(summary["macro_ac_auroc"] - .694) < .01 and summary["worst_site_ac_auroc"] > .654
    summary["predeclared_decision"] = {"success": success, "robustness_tradeoff": robustness, "sites_not_below_compact30": improved}
    (OUTPUT / "aggregate_metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = ["# DATA2-L0 legacy LSTM LOSO results", "", "Outer holdout sites were never used for training, preprocessing, scheduler, checkpoint selection, thresholding, or calibration. The primary result is the best seen-site validation macro-AC checkpoint; epoch 6 is diagnostic only.", "", "| Holdout | Inner train | Inner val | Outer n | Best/stop epoch | Train AC | Seen-val macro AC | Outer AC | Epoch6 AC | Train-val gap | Val-outer gap |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in ordered.itertuples():
        lines.append(f"| {row.Index} | {row.inner_train_n} | {row.inner_val_n} | {row.outer_holdout_n} | {row.best_epoch}/{row.stop_epoch} | {row.train_ac:.3f} | {row.seen_site_val_macro_ac:.3f} | {row.outer_ac:.3f} | {row.epoch6_outer_ac:.3f} | {row.train_minus_val_macro_ac:+.3f} | {row.val_macro_minus_outer_ac:+.3f} |")
    lines += ["", "## Aggregate and fixed comparisons", "", f"- Primary Macro AC: **{summary['macro_ac_auroc']:.3f}**; Worst-site AC: **{summary['worst_site_ac_auroc']:.3f}**.", f"- Epoch6 diagnostic Macro/Worst: {summary['epoch6_macro_ac_auroc']:.3f}/{summary['epoch6_worst_site_ac_auroc']:.3f}.", f"- DATA2 compact30 reference: I0002/I0006/S0001 = 0.748/0.681/0.654; Macro/Worst = 0.694/0.654.", f"- Sites not below compact30: {improved}/3.", f"- Predeclared stable-value success: **{'PASS' if success else 'FAIL'}**.", f"- Predeclared robustness-tradeoff condition: **{'YES' if robustness else 'NO'}**.", "- NEW1103-L0 reference Macro/Worst = 0.622/0.537; OLD1103-L0 = 0.541/0.474. Cross-size differences are descriptive because sample size and composition change.", "", "## Interpretation", "", "- The S0001 holdout fold trains only on I0002+I0006 and therefore has by far the smallest inner-training set; interpret its domain gap together with this fixed data-size limitation.", "- Metric-aligned ranking or typed LSTM may be considered only after reviewing this frozen result; neither experiment was launched here."]
    (OUTPUT / "DATA2_L0_LSTM_RESULTS.md").write_text("\n".join(lines) + "\n")
    print("DATA2_L0_FINALIZED", json.dumps(summary))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--fold", choices=SITES)
    group.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.fold:
        run_fold(args.fold)
    else:
        finalize()


if __name__ == "__main__":
    main()
