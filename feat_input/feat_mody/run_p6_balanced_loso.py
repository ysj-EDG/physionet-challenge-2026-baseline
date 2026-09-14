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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["P5_MANIFEST"] = str(ROOT / "output/p5_loso_1103/loso_manifest.csv")
os.environ["P5_FROZEN_FINGERPRINTS"] = str(
    ROOT / "output/p5_loso_1103/input_fingerprints.json"
)
os.environ["P5_OUTPUT_DIR"] = str(ROOT / "output/p6_site_domain/_unused_p5_output")
os.environ["LSTM_CACHE_DIR"] = str(ROOT / "npz_new")

import train_lstm as tl
from feature_scaling import FeatureScaler
from feat_input.feat_mody import run_loso_1103 as p5


OUTPUT = ROOT / "output/p6_site_domain/lstm_balance"
MANIFEST = ROOT / "output/p5_loso_1103/loso_manifest.csv"
FINGERPRINTS = ROOT / "output/p5_loso_1103/input_fingerprints.json"
P5_AGGREGATE = ROOT / "output/p5_loso_1103/aggregate_metrics.json"
SITES = ("I0002", "I0006", "S0001")
ARMS = {
    "B1_global_class_pos1": {
        "sampler": "global_class_balanced",
        "pos_weight": 1.0,
    },
    "B2_site_class_pos1": {
        "sampler": "site_and_class_balanced",
        "pos_weight": 1.0,
    },
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
        path = ROOT / row.npz_path
        if not path.is_file() or row.record_id not in split_records:
            failures.append(row.record_id)
        expected = p5.CACHE / row.npz_partition / f"{row.record_id}.npz"
        if path.resolve() != expected.resolve():
            failures.append(row.record_id)
    if failures:
        raise RuntimeError(f"Manifest/cache mismatch: {sorted(set(failures))[:10]}")
    if len(list((ROOT / "npz_new").glob("*/*.npz"))) != 1103:
        raise RuntimeError("Old npz_new no longer contains exactly 1103 NPZ files")
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
        pos_weight=torch.tensor([1.0], dtype=torch.float32, device=tl.device)
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
        "replacement": True, "pos_weight_mode": "unit", "pos_weight": 1.0,
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
        "input_cache": "npz_new (legacy 1103)",
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
        "P5_baseline": "P5 balanced class sampler + empirical pos_weight",
        "B1_global_class_pos1": "P6-B1 global class sampler + pos_weight=1",
        "B2_site_class_pos1": "P6-B2 site+class sampler + pos_weight=1",
    }
    lines = [
        "# P6-B: site-balanced LSTM LOSO results", "", "## Frozen protocol", "",
        f"- Source commit before P6 changes: `{git_head()}`.",
        "- Input is only the frozen P5 1103-record LOSO manifest and legacy `npz_new` cache.",
        "- Both P6 arms reuse P5's 2-layer LSTM, `legacy_clip`, seed 7, Adam 1e-3, batch size 8, gradient clip 1.0, and exactly six epochs.",
        "- The only changes are the requested sampler/positive-loss weighting. Holdout sites were constructed and evaluated only after epoch-6 checkpoints were frozen.",
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
    # Mechanically derive the success criteria without altering the experiment.
    lines += ["", "## Interpretation", ""]
    for arm in ARMS:
        d = agg[arm]["delta_vs_p5"]
        improved = sum(d["sites"][site] > 0 for site in SITES)
        success = d["macro_ac_auroc"] > 0 and d["worst_site_ac_auroc"] > 0 and improved >= 2
        lines.append(
            f"- `{arm}` changed macro AC by {d['macro_ac_auroc']:+.3f} and worst-site AC by "
            f"{d['worst_site_ac_auroc']:+.3f}; {improved}/3 sites improved. Under the predeclared "
            f"criteria this is **{'supportive' if success else 'not supportive'}** of carrying the strategy forward."
        )
    (OUTPUT / "P6_SITE_BALANCED_LSTM_RESULTS.md").write_text("\n".join(lines) + "\n")


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
                f"[P6-B] arm={arm} holdout={holdout_site} "
                f"train={len(train_rows)} holdout_n={len(holdout_rows)}", flush=True,
            )
            results.append(run_fold(arm, holdout_site, train_rows, holdout_rows))
    if len(results) != 6:
        raise RuntimeError(f"Expected six completed folds, got {len(results)}")
    metrics, agg = aggregate(results)
    write_report(metrics, agg)
    print("P6_BALANCED_LOSO_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
