"""Position-preserving overnight EEG organization from the frozen EEG54 block."""
from __future__ import annotations

import numpy as np

from .common import linear_quantile, relative_eeg_power, stage_mask
from .contracts import FamilyInputContract, FeatureResult, FieldRequirement
from .registry import (
    SOURCE_ADAPTER_NUMERICAL_CONSTANTS,
    SOURCE_ADAPTER_NUMERICAL_SYMBOLS,
    FamilySpec,
    FeatureDefinition,
)

FAMILY_ID = "eeg_temporal"
FAMILY_VERSION = "1.0.0"
LEADS = ("F3", "F4", "C3", "C4", "O1", "O2")
BANDS = ("delta", "theta", "alpha", "sigma")
BAND_OFFSETS = {"delta": 4, "theta": 3, "alpha": 2, "sigma": 1}
STATISTICS = (
    ("range80", "range80_v1"),
    ("drift", "raw_thirds_drift_v1"),
    ("tail_hour_concentration", "clock_hour_tail_v1"),
    ("tail_entry_rate_per_hour", "tail_entry_rate_v1"),
)
EEG_BLOCK_WIDTH = 9
BETA_OFFSET = 7
P0_OFFSET = 6
DEFAULT_REFERENCE_SUPPORT = 20
DEFAULT_DRIFT_SIDE_SUPPORT = 20
DEFAULT_HOUR_POSITIONS = 120
DEFAULT_HOUR_SLEEP_SUPPORT = 96
EPOCH_SECONDS = 30.0
TAIL_QUANTILE = 0.88


def _definitions() -> tuple[FeatureDefinition, ...]:
    rows = []
    for lead_index, lead in enumerate(LEADS):
        base = lead_index * EEG_BLOCK_WIDTH
        for band in BANDS:
            columns = (base + BAND_OFFSETS[band], base + BETA_OFFSET, base + P0_OFFSET)
            for statistic, formula in STATISTICS:
                rows.append(
                    FeatureDefinition(
                        feature_id=(
                            f"featurelab_v2.eeg_temporal.{lead.lower()}.{band}.{statistic}"
                        ),
                        source_field="X_seq",
                        source_columns=columns,
                        formula_id=formula,
                        parameters={"sleep_scope": "Sleep"},
                        validity_expression=(
                            "valid sleep & lead available & clean_subsegments>0 & "
                            "finite(inputs) on the uncompressed time grid"
                        ),
                        minimum_support=(
                            DEFAULT_HOUR_SLEEP_SUPPORT
                            if statistic == "tail_hour_concentration"
                            else 1
                            if statistic == "tail_entry_rate_per_hour"
                            else DEFAULT_REFERENCE_SUPPORT
                        ),
                        unit=(
                            "per_hour" if statistic.endswith("per_hour")
                            else "fraction" if statistic == "tail_hour_concentration"
                            else "log10_power"
                        ),
                    )
                )
    return tuple(rows)


def extract_family(source, config) -> FeatureResult:
    reference_support = int(config["minimum_reference_support"])
    drift_support = int(config["minimum_drift_side_support"])
    hour_positions = int(config["hour_positions"])
    hour_support = int(config["minimum_hour_sleep_support"])
    epoch_seconds = float(config["epoch_seconds"])
    tail_quantile = float(config["tail_quantile"])
    tail_comparison = str(config["tail_comparison"])
    if tail_comparison != "strict_greater_than":
        raise ValueError("Only the frozen strict-greater-than tail rule is supported")
    definitions = _definitions()
    values = np.full(len(definitions), np.nan, dtype=np.float64)
    valid = np.zeros(len(definitions), dtype=bool)
    support = np.zeros(len(definitions), dtype=np.int64)
    reasons = ["INSUFFICIENT_REFERENCE_SUPPORT"] * len(definitions)

    x_seq = np.asarray(source.arrays["X_seq"])
    stages = np.asarray(source.arrays["stage_code_aligned"])
    stage_valid = np.asarray(source.arrays["stage_valid"])
    starts = np.asarray(source.arrays["epoch_start_sec"], dtype=np.float64)
    durations = np.asarray(source.arrays["epoch_duration_sec"], dtype=np.float64)
    available = np.asarray(source.arrays["eeg_channel_available"])
    clean = np.asarray(source.arrays["eeg_clean_subsegment_count"])
    sleep = stage_mask(stages, stage_valid, "Sleep")
    thirds = np.array_split(np.arange(len(stages)), 3)

    position = 0
    for lead_index, _lead in enumerate(LEADS):
        base = lead_index * EEG_BLOCK_WIDTH
        for band in BANDS:
            columns = (base + BAND_OFFSETS[band], base + BETA_OFFSET, base + P0_OFFSET)
            series = relative_eeg_power(
                x_seq[:, columns[0]], x_seq[:, columns[1]], x_seq[:, columns[2]]
            )
            finite = np.isfinite(series)
            for column in columns:
                finite &= np.isfinite(x_seq[:, column])
            if bool(available[lead_index]):
                selected = sleep & (clean[:, lead_index] > 0) & finite
                unavailable_reason = "INSUFFICIENT_REFERENCE_SUPPORT"
            else:
                selected = np.zeros(len(stages), dtype=bool)
                unavailable_reason = "CHANNEL_UNAVAILABLE"
            reference = np.asarray(series[selected], dtype=np.float64)
            n_reference = len(reference)

            support[position] = n_reference
            reasons[position] = unavailable_reason
            if n_reference >= reference_support:
                values[position] = (
                    linear_quantile(reference, 0.90) - linear_quantile(reference, 0.10)
                )
                valid[position] = True
                reasons[position] = ""
            position += 1

            first_indices = thirds[0] if thirds else np.asarray([], dtype=int)
            last_indices = thirds[-1] if thirds else np.asarray([], dtype=int)
            first_values = series[first_indices][selected[first_indices]]
            last_values = series[last_indices][selected[last_indices]]
            limiting_support = min(len(first_values), len(last_values))
            support[position] = limiting_support
            reasons[position] = (
                "CHANNEL_UNAVAILABLE" if not bool(available[lead_index])
                else "INSUFFICIENT_DRIFT_SIDE_SUPPORT"
            )
            if len(first_values) >= drift_support and len(last_values) >= drift_support:
                values[position] = float(np.mean(last_values) - np.mean(first_values))
                valid[position] = True
                reasons[position] = ""
            position += 1

            best_ratio = None
            best_support = 0
            threshold = (
                linear_quantile(reference, tail_quantile)
                if n_reference >= reference_support else np.nan
            )
            if np.isfinite(threshold):
                for start in range(0, len(stages) - hour_positions + 1):
                    stop = start + hour_positions
                    if not (
                        np.all(np.isfinite(durations[start:stop]))
                        and np.all(durations[start:stop] == epoch_seconds)
                        and np.all(np.isfinite(starts[start:stop]))
                        and np.all(np.diff(starts[start:stop]) == epoch_seconds)
                    ):
                        continue
                    local_valid = selected[start:stop]
                    local_support = int(np.sum(local_valid))
                    if local_support < hour_support:
                        continue
                    ratio = float(
                        np.sum(local_valid & (series[start:stop] > threshold)) / local_support
                    )
                    if best_ratio is None or ratio > best_ratio:
                        best_ratio = ratio
                        best_support = local_support
            support[position] = best_support
            reasons[position] = (
                "CHANNEL_UNAVAILABLE" if not bool(available[lead_index])
                else "NO_QUALIFYING_HOUR"
            )
            if best_ratio is not None:
                values[position] = best_ratio
                valid[position] = True
                reasons[position] = ""
            position += 1

            legal_pairs = (
                selected[:-1]
                & selected[1:]
                & np.isfinite(starts[:-1])
                & np.isfinite(starts[1:])
                & (np.diff(starts) == epoch_seconds)
            )
            pair_count = int(np.sum(legal_pairs))
            support[position] = pair_count
            reasons[position] = (
                "CHANNEL_UNAVAILABLE" if not bool(available[lead_index])
                else "INSUFFICIENT_LEGAL_PAIRS"
            )
            if np.isfinite(threshold) and pair_count > 0:
                high = series > threshold
                entries = int(np.sum(legal_pairs & (~high[:-1]) & high[1:]))
                observed_hours = pair_count * epoch_seconds / 3600.0
                values[position] = entries / observed_hours
                valid[position] = True
                reasons[position] = ""
            position += 1

    return FeatureResult(
        family_id=FAMILY_ID,
        family_version=FAMILY_VERSION,
        feature_ids=tuple(item.feature_id for item in definitions),
        values=values,
        valid_mask=valid,
        support_count=support,
        missing_reason=tuple(reasons),
        feature_units=tuple(item.unit for item in definitions),
        source_identity=source.identity,
        processing_parameters={
            "minimum_reference_support": reference_support,
            "minimum_drift_side_support": drift_support,
            "hour_positions": hour_positions,
            "minimum_hour_sleep_support": hour_support,
            "epoch_seconds": epoch_seconds,
            "tail_quantile": tail_quantile,
            "tail_comparison": tail_comparison,
        },
    )


EEG_TEMPORAL_FEATURES = _definitions()
EEG_TEMPORAL_SPEC = FamilySpec(
    family_id=FAMILY_ID,
    family_version=FAMILY_VERSION,
    extractor_symbol="extensions.eeg_temporal.extract_family",
    default_config={
        "minimum_reference_support": DEFAULT_REFERENCE_SUPPORT,
        "minimum_drift_side_support": DEFAULT_DRIFT_SIDE_SUPPORT,
        "hour_positions": DEFAULT_HOUR_POSITIONS,
        "minimum_hour_sleep_support": DEFAULT_HOUR_SLEEP_SUPPORT,
        "epoch_seconds": EPOCH_SECONDS,
        "tail_quantile": TAIL_QUANTILE,
        "tail_comparison": "strict_greater_than",
    },
    input_contract=FamilyInputContract(
        family_id=FAMILY_ID,
        contract_version="1",
        requirements=(
            FieldRequirement("X_seq", ("T", 483), ("float32",), columns=tuple(range(54))),
            FieldRequirement("stage_code_aligned", ("T",), ("float32",)),
            FieldRequirement("stage_valid", ("T",), ("bool",)),
            FieldRequirement("epoch_start_sec", ("T",), ("float32",)),
            FieldRequirement("epoch_duration_sec", ("T",), ("float32",)),
            FieldRequirement("eeg_channel_available", (6,), ("bool",)),
            FieldRequirement("eeg_clean_subsegment_count", ("T", 6), ("int32",)),
        ),
        stage_time_rules=("raw positions; exact 30-second clock continuity",),
        validity_rules=("sleep stage, lead availability, clean count, and finite inputs",),
    ),
    feature_definitions=EEG_TEMPORAL_FEATURES,
    numerical_symbols=(
        "extensions.eeg_temporal.extract_family",
        "extensions.eeg_temporal._definitions",
        "extensions.common.linear_quantile",
        "extensions.common.relative_eeg_power",
        "extensions.common.stage_mask",
        "extensions.contracts.FeatureResult",
        "extensions.registry.FeatureDefinition",
    ) + SOURCE_ADAPTER_NUMERICAL_SYMBOLS,
    numerical_constants=(
        "extensions.eeg_temporal.FAMILY_ID",
        "extensions.eeg_temporal.FAMILY_VERSION",
        "extensions.eeg_temporal.LEADS",
        "extensions.eeg_temporal.BANDS",
        "extensions.eeg_temporal.BAND_OFFSETS",
        "extensions.eeg_temporal.STATISTICS",
        "extensions.eeg_temporal.EEG_BLOCK_WIDTH",
        "extensions.eeg_temporal.BETA_OFFSET",
        "extensions.eeg_temporal.P0_OFFSET",
        "extensions.eeg_temporal.DEFAULT_REFERENCE_SUPPORT",
        "extensions.eeg_temporal.DEFAULT_HOUR_SLEEP_SUPPORT",
        "extensions.common.STAGE_CODES",
    ) + SOURCE_ADAPTER_NUMERICAL_CONSTANTS,
    numerical_dependencies={"numpy": np.__version__},
    extractor=extract_family,
)
