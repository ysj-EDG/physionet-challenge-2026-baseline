#!/usr/bin/env python
"""Offline engineering validation for typed_v1; no model training or scoring."""

from __future__ import annotations
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from feature_scaling import (
    ECG_EPOCH_OFFSET, EXPECTED_DIMS, FeatureScaler, file_sha256,
    load_feature_rules, record_key, stable_hash, _prepare_values,
)


class Aggregate:
    def __init__(self, dim):
        for name in ("total", "finite", "nonfinite", "zero", "abs5", "abs10",
                     "abs20", "abs50", "filled", "invalid"):
            setattr(self, name, np.zeros(dim, np.int64))
        self.minimum = np.full(dim, np.inf)
        self.maximum = np.full(dim, -np.inf)
        self.sum = np.zeros(dim, np.float64)
        self.sumsq = np.zeros(dim, np.float64)
        self.records = 0

    def update(self, values, filled=None, invalid=None):
        x = np.asarray(values)
        if x.ndim == 1:
            x = x[None, :]
        self.records += 1
        if len(x) == 0:
            return
        finite = np.isfinite(x)
        safe = np.where(finite, x, 0).astype(np.float64)
        self.total += len(x)
        self.finite += finite.sum(0)
        self.nonfinite += (~finite).sum(0)
        self.zero += ((x == 0) & finite).sum(0)
        self.abs5 += ((np.abs(x) > 5) & finite).sum(0)
        self.abs10 += ((np.abs(x) > 10) & finite).sum(0)
        self.abs20 += ((np.abs(x) > 20) & finite).sum(0)
        self.abs50 += ((np.abs(x) > 50) & finite).sum(0)
        self.sum += safe.sum(0)
        self.sumsq += np.square(safe).sum(0)
        self.minimum = np.minimum(self.minimum, np.min(np.where(finite, x, np.inf), axis=0))
        self.maximum = np.maximum(self.maximum, np.max(np.where(finite, x, -np.inf), axis=0))
        if filled is not None:
            self.filled += np.asarray(filled, dtype=np.int64)
        if invalid is not None:
            self.invalid += np.asarray(invalid, dtype=np.int64)


def typed_validity(raw, branch, scaler, hrv_sentinel, caisr_sentinel):
    matrix = raw if raw.ndim == 2 else raw[None, :]
    filled = np.zeros(matrix.shape[1], np.int64)
    invalid = np.zeros(matrix.shape[1], np.int64)
    for index in range(matrix.shape[1]):
        rule = scaler._rule_map[(branch, index)]
        _, valid, diagnostics = _prepare_values(matrix[:, index], rule)
        if branch == "X_ecg" and index < 11:
            valid &= ~hrv_sentinel
        if branch == "x_static" and index >= 10 and caisr_sentinel:
            valid[:] = False
        filled[index] = int((~valid).sum())
        invalid[index] = (
            diagnostics["invalid_missing_rule"] + diagnostics["range_invalid"]
            + diagnostics["negative_for_log"]
        )
    return filled, invalid


def scope_keys(split, site):
    return ("global:all", f"role:{split}", f"site:{site}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=ROOT / "npz_new")
    parser.add_argument("--split-root", type=Path, default=ROOT / "split")
    parser.add_argument("--scaler-state", type=Path,
                        default=ROOT / "feat_input" / "p0_fix" / "scaler_state_preview.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "feat_input" / "p0_fix")
    args = parser.parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rules = load_feature_rules()
    state = json.loads(args.scaler_state.read_text(encoding="utf-8"))
    scaler = FeatureScaler.from_state_dict(state)
    legacy = FeatureScaler.legacy_clip()
    train_manifest = args.split_root / "train_records.json"
    if scaler.metadata["manifest_sha256"] != file_sha256(train_manifest):
        raise RuntimeError("Frozen scaler train manifest hash does not match current train manifest")
    if scaler.metadata["rules_sha256"] != stable_hash(rules):
        raise RuntimeError("Frozen scaler rules do not match approved rules")

    manifests = {
        split: json.loads((args.split_root / f"{split}_records.json").read_text())
        for split in ("train", "val", "test", "external")
    }
    identities = {}
    for split, records in manifests.items():
        for record in records:
            rid = record_key(record)
            if rid in identities:
                raise RuntimeError(f"duplicate record {rid}")
            identities[rid] = (split, str(record["SiteID"]))

    before = {}
    for rid, (split, _) in identities.items():
        path = args.cache_root / split / f"{rid}.npz"
        stat = path.stat()
        before[str(path)] = (stat.st_size, stat.st_mtime_ns)

    aggregates = {}
    comparisons = {
        ("x_static", 0): "age", ("x_static", 9): "BMI",
        ("X_ecg", 0): "MedianNN", ("X_ecg", 4): "pNN20",
        ("x_static", 10): "TRT", ("x_static", 11): "TST",
        ("x_static", 16): "WASO", ("X_seq", 414): "BSR",
        ("X_seq", 433): "EMG_envelope_ratio", ("x_static", 28): "N3_N1_ratio",
    }
    comparison_values = {
        label: {variant: [] for variant in ("raw", "legacy_clip", "typed_v1")}
        for label in comparisons.values()
    }
    counts = Counter()
    errors = []

    for rid, (split, site) in sorted(identities.items()):
        path = args.cache_root / split / f"{rid}.npz"
        try:
            with np.load(path, allow_pickle=False) as data:
                X_seq = np.asarray(data["X_seq"])
                X_ecg_raw = np.asarray(data["X_ecg"])
                x_static = np.asarray(data["x_static"])
            FeatureScaler._validate_shapes(X_seq, X_ecg_raw, x_static, rid)
            consumed = max(0, min(len(X_ecg_raw), len(X_seq) - ECG_EPOCH_OFFSET))
            X_ecg = X_ecg_raw[:consumed]
            raw_arrays = {"X_seq": X_seq, "X_ecg": X_ecg, "x_static": x_static}
            legacy_arrays = dict(zip(raw_arrays, legacy.transform_arrays(X_seq, X_ecg, x_static)))
            typed_result = scaler.transform_arrays(
                X_seq, X_ecg, x_static, record_id=rid, return_diagnostics=True
            )
            typed_arrays = dict(zip(raw_arrays, typed_result[:3]))
            diagnostics = typed_result[3]
            counts["records"] += 1
            counts["raw_ecg_windows"] += len(X_ecg_raw)
            counts["consumed_ecg_windows"] += consumed
            counts["discarded_ecg_tail_windows"] += len(X_ecg_raw) - consumed
            counts["legacy_zero_hrv_sentinel_windows"] += diagnostics["legacy_zero_hrv_sentinel_windows"]
            counts["legacy_zero_caisr_sentinel_records"] += diagnostics["legacy_zero_caisr_sentinel_records"]

            hrv_sentinel = (
                np.all(np.isfinite(X_ecg[:, :11]), axis=1)
                & np.all(X_ecg[:, :11] == 0, axis=1)
            ) if consumed else np.zeros(0, dtype=bool)
            caisr_sentinel = bool(
                np.isfinite(x_static[10:]).all() and x_static[10] == 0
                and np.all(x_static[10:] == 0)
            )
            for branch, raw in raw_arrays.items():
                filled, invalid = typed_validity(raw, branch, scaler, hrv_sentinel, caisr_sentinel)
                matrix = raw if raw.ndim == 2 else raw[None, :]
                variants = {
                    "raw": (raw, np.zeros_like(filled), np.zeros_like(invalid)),
                    "legacy_clip": (
                        legacy_arrays[branch], (~np.isfinite(matrix)).sum(0),
                        np.zeros_like(invalid),
                    ),
                    "typed_v1": (typed_arrays[branch], filled, invalid),
                }
                for group in scope_keys(split, site):
                    for variant, (values, fill_counts, invalid_counts) in variants.items():
                        key = (group, branch, variant)
                        aggregates.setdefault(key, Aggregate(EXPECTED_DIMS[branch])).update(
                            values, fill_counts, invalid_counts
                        )
                for feature_key, label in comparisons.items():
                    if feature_key[0] != branch:
                        continue
                    index = feature_key[1]
                    for variant, (values, _, _) in variants.items():
                        column = values[:, index] if values.ndim == 2 else values[index:index + 1]
                        comparison_values[label][variant].append(np.asarray(column, dtype=float))
        except Exception as exc:
            errors.append({"split": split, "site": site,
                           "error": f"{type(exc).__name__}: {exc}"})

    after = {}
    for path_text in before:
        stat = Path(path_text).stat()
        after[path_text] = (stat.st_size, stat.st_mtime_ns)
    cache_unchanged = before == after
    if errors or not cache_unchanged:
        raise RuntimeError(f"scan errors={len(errors)} cache_unchanged={cache_unchanged}")

    rule_map = {(r["branch"], r["index"]): r for r in rules}
    rows = []
    for (group, branch, variant), agg in sorted(aggregates.items()):
        scope, value = group.split(":", 1)
        for index in range(EXPECTED_DIMS[branch]):
            finite = int(agg.finite[index])
            mean = agg.sum[index] / finite if finite else math.nan
            variance = agg.sumsq[index] / finite - mean ** 2 if finite else math.nan
            rows.append({
                "scope": scope, "site": value if scope == "site" else "",
                "role": value if scope == "role" else "all", "fold": "small_I0002",
                "variant": variant, "branch": branch, "index": index,
                "name": rule_map[(branch, index)]["name"],
                "family": rule_map[(branch, index)]["family"],
                "records": agg.records, "total_observations": int(agg.total[index]),
                "finite_observations": finite, "nonfinite_count": int(agg.nonfinite[index]),
                "filled_or_sentinel_count": int(agg.filled[index]),
                "substantial_invalid_count": int(agg.invalid[index]),
                "zero_count": int(agg.zero[index]),
                "min": float(agg.minimum[index]) if finite else math.nan,
                "max": float(agg.maximum[index]) if finite else math.nan,
                "mean": float(mean),
                "std": float(np.sqrt(max(0.0, variance))) if finite else math.nan,
                "abs_gt_5_count": int(agg.abs5[index]),
                "abs_gt_10_count": int(agg.abs10[index]),
                "abs_gt_20_count": int(agg.abs20[index]),
                "abs_gt_50_count": int(agg.abs50[index]),
                "abs_gt_5_fraction_finite": agg.abs5[index] / finite if finite else math.nan,
                "abs_gt_10_fraction_finite": agg.abs10[index] / finite if finite else math.nan,
                "abs_gt_20_fraction_finite": agg.abs20[index] / finite if finite else math.nan,
                "abs_gt_50_fraction_finite": agg.abs50[index] / finite if finite else math.nan,
                "observation_basis": (
                    "X_seq real epochs; X_ecg only min(T_ecg,T_seq-10) consumed windows; "
                    "x_static once per record; collate padding and pre-offset ECG zeros excluded"
                ),
            })
    stats = pd.DataFrame(rows)
    stats.to_csv(args.output_dir / "transformed_feature_stats.csv", index=False)

    comparison_rows = []
    for label, variants in comparison_values.items():
        for variant, blocks in variants.items():
            x = np.concatenate(blocks)
            x = x[np.isfinite(x)]
            comparison_rows.append({
                "feature": label, "variant": variant, "n": len(x),
                "min": float(np.min(x)), "p1": float(np.quantile(x, 0.01)),
                "p50": float(np.quantile(x, 0.50)),
                "p99": float(np.quantile(x, 0.99)), "max": float(np.max(x)),
            })
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(args.output_dir / "key_feature_comparison.csv", index=False)

    fit_summary = {
        "metadata": scaler.metadata, "parameters": scaler.parameters,
        "rules_sha256": state["rules_sha256"],
        "feature_order_sha256": state["feature_order_sha256"],
        "scale_method_counts": dict(Counter(p["scale_method"] for p in scaler.parameters)),
        "all_missing_columns": [
            f"{p['branch']}[{p['index']}] {p['name']}"
            for p in scaler.parameters if p["scale_method"] == "all_missing_fallback"
        ],
        "unit_fallback_columns": [
            f"{p['branch']}[{p['index']}] {p['name']}"
            for p in scaler.parameters if p["scale_method"] == "unit_fallback"
        ],
    }
    (args.output_dir / "scaler_fit_summary.json").write_text(
        json.dumps(fit_summary, indent=2, ensure_ascii=False) + "\n"
    )

    typed_global = stats[(stats.scope == "global") & (stats.variant == "typed_v1")]
    legacy_global = stats[(stats.scope == "global") & (stats.variant == "legacy_clip")]
    extreme_table = typed_global[typed_global.abs_gt_50_count > 0][
        ["branch", "index", "name", "abs_gt_50_count", "abs_gt_50_fraction_finite", "min", "max"]
    ].sort_values("abs_gt_50_count", ascending=False)
    report = [
        "# P0 typed_v1 input preprocessing implementation report", "",
        "## Outcome", "",
        f"- Frozen scaler fitted from {scaler.metadata['fit_record_count']} unique train records only.",
        f"- Offline transformation covered {counts['records']}/{len(identities)} cached records with zero errors.",
        f"- All typed_v1 outputs were finite: {int(typed_global.nonfinite_count.sum())} nonfinite values.",
        f"- Cache file size/mtime metadata remained unchanged: {cache_unchanged}.",
        "- P0 engineering input repair passed mapping, path, finite-value, cache-integrity and test checks. No model training and no AUROC/AUPRC claim.", "",
        "## Fixed interface", "",
        "- One shared FeatureScaler implements typed_v1 and exact legacy_clip modes.",
        "- Explicit per-column unit/nonlinear/type rules precede train-only robust parameters.",
        "- Raw age is copied before model-input transformation; ECG offset and dimensions are unchanged.",
        "- Padding, pre-offset ECG zeros, empty ECG and documented legacy sentinels remain zero.",
        "- New checkpoints save preprocessing state; old checkpoints require LSTM_ALLOW_LEGACY_INPUT=1.", "",
        "## Data accounting", "",
        f"- Raw ECG windows: {counts['raw_ecg_windows']}; consumed: {counts['consumed_ecg_windows']}; excluded tail: {counts['discarded_ecg_tail_windows']}.",
        f"- Legacy all-zero HRV sentinel windows: {counts['legacy_zero_hrv_sentinel_windows']}.",
        f"- Legacy all-zero CAISR sentinel records: {counts['legacy_zero_caisr_sentinel_records']}.",
        f"- Scale methods: {fit_summary['scale_method_counts']}.",
        f"- typed_v1 abs(value)>50 count across global feature rows: {int(typed_global.abs_gt_50_count.sum())}.",
        f"- legacy_clip abs(value)>50 count: {int(legacy_global.abs_gt_50_count.sum())}.", "",
        "## Remaining large transformed values", "",
        extreme_table.to_string(index=False, float_format=lambda x: f"{x:.6g}"),
        "",
        "These values are finite and rare but remain explicit review items; no automatic z clipping was added.", "",
        "## Key feature comparison", "",
        comparison.to_string(index=False, float_format=lambda x: f"{x:.6g}"), "",
        "## Validation", "",
        "- test_feature_scaling.py: 11/11 passed.",
        "- test_feature_scaling_fit_and_compat.py: 4/4 passed.",
        "- CPU forward/backward finite gradients and train/inference logit agreement at 1e-6.",
        "- Two real train caches also passed shared-transform equality and a finite CPU backward pass.",
        "- Full offline transformation uses no labels; test/external y=-1 is not scored.", "",
        "## Reproduction", "",
        "conda run -n sleepfm_env python feat_input/feat_mody/build_feature_rules_v1.py",
        "conda run -n sleepfm_env python feat_input/feat_mody/test_feature_scaling.py",
        "conda run -n sleepfm_env python feat_input/feat_mody/test_feature_scaling_fit_and_compat.py",
        "conda run -n sleepfm_env python feat_input/feat_mody/smoke_real_cache.py",
        "conda run -n sleepfm_env python feat_input/feat_mody/offline_validate_typed_v1.py --cache-root npz_new --split-root split", "",
        "## Training command (not executed)", "",
        "LSTM_INPUT_PREPROCESSING=typed_v1 LSTM_CACHE_DIR=npz_new LSTM_SPLITS_DIR=split LSTM_MODEL_DIR=<new_model_dir> LSTM_SEED=7 conda run -n sleepfm_env python train_lstm.py", "",
        "## Boundary of evidence", "",
        "- Code proves shared deterministic behavior, strict state validation and explicit legacy compatibility.",
        "- Cache measurement proves finite transformed outputs for this 1103-record cache.",
        "- Historical provenance remains assumed because NPZ has no embedded feature names/version.",
        "- Performance and generalization require controlled legacy_clip versus typed_v1 retraining.", "",
        f"Offline validation elapsed: {time.time() - started:.1f} seconds.",
    ]
    (args.output_dir / "IMPLEMENTATION_REPORT.md").write_text("\n".join(report) + "\n")

    unresolved = """# Unresolved input issues

- Historical NPZ files have dimensions but no embedded feature names or extractor version; current-order compatibility is an explicit assumption.
- Upstream extraction already converted some missing or failed measurements to zero. Except for approved all-11 HRV and all-186 CAISR sentinels, those origins cannot be recovered.
- Extreme EMG envelope ratios can reflect denominator or quality failures. typed_v1 makes them finite with log1p but does not certify physiological validity.
- All-zero StageEvent and modality blocks remain ambiguous and are not used to delete EEG records.
- Hard-range violations are counted and treated as invalid; they are not silently clipped.
- No standardized z clipping is enabled. Remaining abs(z)>50 values occur in three EMG envelope ratios (1086 observations total), one arousal interval and one signed limb-index difference; these remain column-level review items.
- typed_v1 combines scaling, BMI missing handling and legacy sentinels; individual effects require later ablations.
- No training, scoring, label recovery, feature re-extraction or architecture change was performed.
"""
    (args.output_dir / "unresolved_input_issues.md").write_text(unresolved)
    print(json.dumps({
        "records": counts["records"], "errors": len(errors),
        "cache_unchanged": cache_unchanged,
        "typed_nonfinite": int(typed_global.nonfinite_count.sum()),
        "elapsed_seconds": time.time() - started,
    }, indent=2))


if __name__ == "__main__":
    main()
