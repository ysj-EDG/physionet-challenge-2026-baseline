#!/usr/bin/env python3
"""Physio Fusion V1 Gate 1/2 command-line entrypoint."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from dataset import (
    EXPECTED_SITE_COUNTS,
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
from train import summarize_and_select, train_coherence_ae

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("smoke", "d-search"))
    parser.add_argument("--npz-root", type=Path, default=DEFAULT_NPZ_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("output/physio_fusion_v1/d_selection"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.command == "smoke":
        run_smoke(args.npz_root)
    else:
        run_d_search(args.npz_root, args.output_dir, args.device)


if __name__ == "__main__":
    main()
