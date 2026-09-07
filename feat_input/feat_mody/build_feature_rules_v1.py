#!/usr/bin/env python
"""Build the reviewed typed_v1 rule table; fail on every unmapped column."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SCHEMA_PATH = ROOT / "feat_input" / "feature_schema.json"
OUTPUT_PATH = ROOT / "feat_input" / "feature_rules_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    from feat_input.audit_feature_scaling import build_schema
    runtime_schema = build_schema()
    stored_order = [(x["branch"], int(x["index"]), x["name"]) for x in schema]
    runtime_order = [(x["branch"], int(x["index"]), x["name"]) for x in runtime_schema]
    if stored_order != runtime_order:
        raise RuntimeError("Audited feature schema no longer matches current extractor order")
    current_names = {(x["branch"], int(x["index"])): x for x in schema}
    if len(current_names) != 691:
        raise RuntimeError("feature_schema.json must contain exactly 691 unique columns")

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    rules = {}

    def add(
        branch, index, *, family, nonlinear="identity", scaling="robust",
        unit="unitless", unit_scale=1.0, reference_unit="1",
        expected=None, hard=False, missing_rule="nonfinite",
        zero_semantics="valid zero unless a documented whole-block sentinel applies",
        formula_summary,
    ):
        key = (branch, int(index))
        if key in rules:
            raise RuntimeError(f"duplicate rule {key}")
        source = current_names.get(key)
        if source is None:
            raise RuntimeError(f"schema has no column {key}")
        rules[key] = {
            "branch": branch,
            "index": int(index),
            "name": source["name"],
            "family": family,
            "formula_source": {
                "file": source["source_file"],
                "line": int(source.get("source_line", -1)),
                "function": source.get("source_function", "unknown"),
            },
            "formula_summary": formula_summary,
            "original_unit": unit,
            "unit_scale": float(unit_scale),
            "nonlinear": nonlinear,
            "reference_unit": reference_unit,
            "scaling": scaling,
            "expected_range": {
                "min": expected[0] if expected else None,
                "max": expected[1] if expected else None,
                "hard": bool(hard),
                "coordinate": "original_input",
            },
            "range_tolerance": 1e-6,
            "missing_rule": missing_rule,
            "identity_fill": 0.0,
            "zero_semantics": zero_semantics,
            "audit_source": "feat_input/AUDIT_REPORT.md and current extractor formula review",
            "source_uncertainty": (
                "Rule order matches current extractor and audited dimensions. Historical NPZ "
                "has no embedded names/version, so historical extraction provenance remains an explicit assumption."
            ),
            "approved_rule_version": "typed_v1",
            "code_head_at_rule_build": head,
        }

    # X_seq[0:54]: six channels x nine spectral features.
    for index in range(54):
        position = index % 9
        if position == 5:
            add("X_seq", index, family="eeg_spectral_sef50", unit="Hz",
                formula_summary="SEF50 frequency; identity then train robust scaling")
        else:
            add("X_seq", index, family="eeg_spectral_log10", unit="log10 power or log10 ratio",
                formula_summary="Extractor already applies log10; preserve signed value and robust-scale")

    # X_seq[54:414]: fifteen pairs x twenty-four explicit coherence statistics.
    bounded_positions = {0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 22, 23}
    auc_positions = {5, 6, 7, 8, 9}
    ratio_positions = {15, 16, 17, 18}
    hz_positions = {19, 21}
    for index in range(54, 414):
        position = (index - 54) % 24
        if position in bounded_positions:
            add("X_seq", index, family="eeg_coherence_bounded", scaling="identity",
                expected=(0.0, 1.0), hard=True,
                formula_summary="Mean/IQR/normalized entropy/sigma-band coherence proven within [0,1]")
        elif position in auc_positions:
            add("X_seq", index, family="eeg_coherence_auc", unit="Hz-weighted coherence",
                formula_summary="Frequency-integrated coherence AUC; not a [0,1] mean")
        elif position in ratio_positions:
            add("X_seq", index, family="eeg_coherence_ratio", nonlinear="log1p",
                expected=(0.0, None), hard=True, reference_unit="ratio=1",
                formula_summary="Unbounded nonnegative coherence ratio; log1p(x/1) then robust scaling")
        elif position in hz_positions:
            add("X_seq", index, family="eeg_coherence_frequency", unit="Hz",
                expected=(0.0, None), hard=True,
                formula_summary="Coherence spectral centroid or bandwidth in Hz")
        else:
            raise RuntimeError(f"unclassified coherence position {position}")

    for index in range(414, 432):
        add("X_seq", index, family="eeg_bsr_percent", scaling="identity",
            unit="percent", unit_scale=100.0, reference_unit="fraction",
            expected=(0.0, 100.0), hard=True,
            zero_semantics="valid no-suppression value or unavailable-channel fallback; never treated as BMI missing",
            formula_summary="BSR extractor returns percent; divide by 100 and pass through")

    for index in range(432, 456):
        position = (index - 432) % 8
        if position in {0, 7}:
            add("X_seq", index, family="emg_existing_log", unit="natural-log ratio",
                formula_summary="Extractor already applies natural log; preserve signed value and robust-scale")
        elif position == 1:
            add("X_seq", index, family="emg_envelope_ratio", nonlinear="log1p",
                expected=(0.0, None), hard=True, reference_unit="ratio=1",
                formula_summary="Nonnegative envelope IQR ratio; log1p(x/1) controls audited extreme tail")
        elif position == 4:
            add("X_seq", index, family="emg_burst_rate", nonlinear="log1p",
                unit="events/min", expected=(0.0, None), hard=True,
                reference_unit="1 event/min",
                formula_summary="Nonnegative burst rate; log1p per 1 event/min then robust-scale")
        elif position in {2, 3, 5, 6}:
            add("X_seq", index, family="emg_fraction", scaling="identity",
                expected=(0.0, 1.0), hard=True,
                formula_summary="Formula-derived activity/tonic/duty/phasic fraction in [0,1]")
        else:
            raise RuntimeError(f"unclassified EMG position {position}")

    resp_log = {457, 458, 459, 463, 465}
    resp_robust = {456, 462, 468}
    resp_fraction = {460, 461, 464, 466, 469}
    for index in range(456, 470):
        if index in resp_log:
            add("X_seq", index, family="respiratory_nonnegative_ratio", nonlinear="log1p",
                expected=(0.0, None), hard=True, reference_unit="ratio=1",
                formula_summary="Approved nonnegative unbounded respiratory ratio; log1p(x/1)")
        elif index in resp_robust:
            unit = "breaths/min" if index == 456 else ("seconds" if index in {462, 468} else "unitless")
            add("X_seq", index, family="respiratory_continuous", unit=unit,
                formula_summary="Signed/physical respiratory value; identity then robust scaling")
        elif index == 467:
            add("X_seq", index, family="respiratory_correlation", scaling="identity",
                expected=(-1.0, 1.0), hard=True,
                formula_summary="Thorax-abdomen Pearson correlation; pass through [-1,1]")
        elif index in resp_fraction:
            add("X_seq", index, family="respiratory_fraction", scaling="identity",
                expected=(0.0, 1.0), hard=True,
                formula_summary="Formula-derived respiratory coverage/paradox fraction in [0,1]")
        else:
            raise RuntimeError(f"unclassified respiratory index {index}")

    for index in range(470, 483):
        add("X_seq", index, family="stage_event_indicator_or_coverage", scaling="identity",
            expected=(0.0, 1.0), hard=True,
            zero_semantics="valid stage-off/no-event value; all-zero block remains unresolved and is not deleted",
            formula_summary="Five stage one-hot plus eight event coverage fractions; pass through")

    ecg = {
        0: ("ecg_median_nn", "identity", "robust", "ms", 1000.0, "seconds", (0.0, None), True,
            "Median NN interval: convert milliseconds to seconds then robust-scale"),
        1: ("ecg_ratio", "log1p", "robust", "ratio", 1.0, "ratio=1", (0.0, None), True, "MCVNN nonnegative ratio"),
        2: ("ecg_ratio", "log1p", "robust", "ratio", 1.0, "ratio=1", (0.0, None), True, "CVNN nonnegative ratio"),
        3: ("ecg_ratio", "log1p", "robust", "ratio", 1.0, "ratio=1", (0.0, None), True, "CVSD nonnegative ratio"),
        4: ("ecg_percent", "identity", "identity", "percent", 100.0, "fraction", (0.0, 100.0), True, "pNN20 percent divided by 100"),
        5: ("ecg_existing_log", "identity", "robust", "natural-log power", 1.0, "1", None, False, "LF already natural-log transformed"),
        6: ("ecg_existing_log", "identity", "robust", "natural-log power", 1.0, "1", None, False, "HF already natural-log transformed"),
        7: ("ecg_fraction", "identity", "identity", "fraction", 1.0, "1", (0.0, 1.0), True, "HF/(LF+HF) fraction"),
        8: ("ecg_ratio", "log1p", "robust", "ratio", 1.0, "ratio=1", (0.0, None), True, "SD1/SD2 unbounded ratio"),
        9: ("ecg_fraction", "identity", "identity", "fraction", 1.0, "1", (0.0, 1.0), True, "Symbolic 0V fraction"),
        10: ("ecg_fraction", "identity", "identity", "fraction", 1.0, "1", (0.0, 1.0), True, "Symbolic 2UV fraction"),
        11: ("ecg_circadian", "identity", "identity", "cosine", 1.0, "1", (-1.0, 1.0), True, "Circadian cosine"),
    }
    for index, spec in ecg.items():
        family, nonlinear, scaling, unit, unit_scale, reference, expected, hard, summary = spec
        add("X_ecg", index, family=family, nonlinear=nonlinear, scaling=scaling,
            unit=unit, unit_scale=unit_scale, reference_unit=reference,
            expected=expected, hard=hard,
            zero_semantics=(
                "first 11 simultaneous zeros are legacy HRV failure sentinel and stay zero"
                if index < 11 else "valid circadian phase zero; not cleared by HRV sentinel"
            ), formula_summary=summary)

    add("x_static", 0, family="demographic_age", unit="years",
        expected=(0.0, None), hard=False, missing_rule="nonfinite_or_nonpositive",
        formula_summary="Model age robust-scaled; independent raw age remains untouched for scoring")
    for index in range(1, 9):
        add("x_static", index, family="demographic_one_hot", scaling="identity",
            expected=(0.0, 1.0), hard=True,
            formula_summary="Existing sex/race categorical indicator passed through without recoding")
    add("x_static", 9, family="demographic_bmi", unit="kg/m^2",
        expected=(0.0, None), hard=False, missing_rule="nonfinite_or_nonpositive",
        zero_semantics="zero/nonfinite/nonpositive is explicit unavailable BMI and is excluded from fit",
        formula_summary="Positive finite BMI robust-scaled; unavailable BMI fills to train median -> z=0")

    A_hours = {10, 11, 13, 14, 15, 16, 22, 23, 24, 25, 26}
    B_duration = set(range(39, 74)) | set(range(80, 85))
    C_counts = {31, 33, 35, 36, 37, 38, 79, 94, 124, 128, 129, 130, 131, 132, 166, 170, 171}
    D_rates = {32, 96, *range(108, 115), 126, *range(133, 138), 151, 152, *range(154, 159), 163, 168, 172, 173, *range(183, 189)}
    E_ratios = {28, 29, 30, 97, 127, 153, 169}
    F_durations = {95, *range(98, 107), 125, *range(141, 149), 150, 161, 167, *range(175, 182)}
    G_bounded = {12, *range(17, 22), 27, 34, 74, 75, *range(85, 91), 107, *range(116, 124), *range(138, 141), 149, 160, 162, 164, 165, 174, 182, *range(190, 194)}
    H_signed_direct = {76, 77, 78}
    H_signed_robust = {92, 93, 115, 159, 189}
    I_composite = {91, 194, 195}
    groups = [A_hours, B_duration, C_counts, D_rates, E_ratios, F_durations,
              G_bounded, H_signed_direct, H_signed_robust, I_composite]
    union = set().union(*groups)
    duplicates = sum(len(group) for group in groups) - len(union)
    expected_static = set(range(10, 196))
    if union != expected_static or duplicates:
        raise RuntimeError(
            f"CAISR classification mismatch missing={sorted(expected_static-union)} "
            f"extra={sorted(union-expected_static)} duplicates={duplicates}"
        )

    for index in sorted(A_hours):
        add("x_static", index, family="caisr_total_duration_hours", unit="seconds",
            unit_scale=3600.0, reference_unit="hours",
            expected=(0.0, None), hard=True,
            formula_summary="Large sleep total duration converted seconds/3600 then robust-scaled")
    for index in sorted(B_duration | F_durations):
        add("x_static", index, family="caisr_duration_long_tail", nonlinear="log1p",
            unit="seconds", unit_scale=60.0, reference_unit="1 minute",
            expected=(0.0, None), hard=True,
            formula_summary="Approved nonnegative bout/event/interval duration; log1p(seconds/60)")
    for index in sorted(C_counts):
        add("x_static", index, family="caisr_count", nonlinear="log1p",
            unit="count", reference_unit="1 count", expected=(0.0, None), hard=True,
            formula_summary="Nonnegative event/bout/cycle count; log1p(count/1)")
    for index in sorted(D_rates):
        add("x_static", index, family="caisr_rate_per_hour", nonlinear="log1p",
            unit="events/hour", reference_unit="1 event/hour",
            expected=(0.0, None), hard=True,
            formula_summary="Formula-derived nonnegative hourly rate; log1p(rate/1 event/hour)")
    for index in sorted(E_ratios):
        add("x_static", index, family="caisr_unbounded_ratio", nonlinear="log1p",
            unit="ratio", reference_unit="ratio=1", expected=(0.0, None), hard=True,
            formula_summary="Formula permits ratio above one; log1p(ratio/1), never clip to one")
    for index in sorted(G_bounded):
        upper = math.log(5.0) if index == 87 else (2.0 if index == 90 else 1.0)
        add("x_static", index, family="caisr_bounded", scaling="identity",
            expected=(0.0, upper), hard=True,
            formula_summary=(
                "Stage entropy in [0,log(5)] passed through"
                if index == 87 else
                "Posterior L1 volatility in [0,2] passed through"
                if index == 90 else
                "Formula-derived efficiency/share/probability/coverage quantity in [0,1]"
            ))
    for index in sorted(H_signed_direct):
        add("x_static", index, family="caisr_signed_fraction_difference", scaling="identity",
            expected=(-1.0, 1.0), hard=True,
            formula_summary="Signed early/late proportion difference; preserve sign and pass through")
    for index in sorted(H_signed_robust):
        add("x_static", index, family="caisr_signed_index", unit="formula-specific signed index",
            formula_summary="Signed rate difference or composite index; identity then robust-scale, no abs/log")
    for index in sorted(I_composite):
        add("x_static", index, family="caisr_nonnegative_composite", nonlinear="log1p",
            unit="formula-defined composite index", reference_unit="index=1",
            expected=(0.0, None), hard=True,
            formula_summary="Nonnegative fragmentation/burden composite; log1p(index/1) then robust-scale")

    ordered = []
    for branch in ("X_seq", "X_ecg", "x_static"):
        dim = {"X_seq": 483, "X_ecg": 12, "x_static": 196}[branch]
        for index in range(dim):
            key = (branch, index)
            if key not in rules:
                raise RuntimeError(f"unmapped feature {key}: {current_names.get(key)}")
            rule = rules[key]
            if rule["name"] != current_names[key]["name"]:
                raise RuntimeError(f"name mismatch at {key}")
            ordered.append(rule)
    if len(ordered) != 691:
        raise RuntimeError(len(ordered))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "rules": len(ordered),
        "branch_counts": {
            branch: sum(r["branch"] == branch for r in ordered)
            for branch in ("X_seq", "X_ecg", "x_static")
        },
        "head": head,
    }, indent=2))


if __name__ == "__main__":
    main()
