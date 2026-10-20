"""Stage-specific high quantiles derived from the frozen EEG54 block."""
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

FAMILY_ID = "eeg_tail"
FAMILY_VERSION = "1.0.0"
LEADS = ("F3", "F4", "C3", "C4", "O1", "O2")
BANDS = ("delta", "theta", "alpha", "sigma")
BAND_OFFSETS = {"delta": 4, "theta": 3, "alpha": 2, "sigma": 1}
STAGE_SCOPES = ("Wake", "REM", "N2", "N3", "NREM", "All-stage")
QUANTILES = (("q88", 0.88), ("q95", 0.95))
EEG_BLOCK_WIDTH = 9
BETA_OFFSET = 7
P0_OFFSET = 6
DEFAULT_MINIMUM_SUPPORT = 20


def _definitions() -> tuple[FeatureDefinition, ...]:
    rows = []
    for lead_index, lead in enumerate(LEADS):
        base = lead_index * EEG_BLOCK_WIDTH
        for band in BANDS:
            columns = (base + BAND_OFFSETS[band], base + BETA_OFFSET, base + P0_OFFSET)
            for scope in STAGE_SCOPES:
                for quantile_name, quantile in QUANTILES:
                    rows.append(
                        FeatureDefinition(
                            feature_id=(
                                f"featurelab_v2.eeg_tail.{lead.lower()}.{band}."
                                f"{scope.lower()}.{quantile_name}"
                            ),
                            source_field="X_seq",
                            source_columns=columns,
                            formula_id="relative_power_linear_quantile_v1",
                            parameters={"quantile": quantile},
                            validity_expression=(
                                "stage_valid & stage_scope & eeg_channel_available & "
                                "eeg_clean_subsegment_count>0 & finite(inputs)"
                            ),
                            minimum_support=DEFAULT_MINIMUM_SUPPORT,
                            unit="log10_power",
                        )
                    )
    return tuple(rows)


def extract_family(source, config) -> FeatureResult:
    minimum_support = int(config["minimum_support"])
    quantiles = tuple(float(item) for item in config["quantiles"])
    if quantiles != tuple(item[1] for item in QUANTILES):
        raise ValueError("EEG-tail quantiles must preserve the frozen feature order")
    x_seq = np.asarray(source.arrays["X_seq"])
    stages = np.asarray(source.arrays["stage_code_aligned"])
    stage_valid = np.asarray(source.arrays["stage_valid"])
    available = np.asarray(source.arrays["eeg_channel_available"])
    clean = np.asarray(source.arrays["eeg_clean_subsegment_count"])

    definitions = _definitions()
    count = len(definitions)
    values = np.full(count, np.nan, dtype=np.float64)
    valid = np.zeros(count, dtype=bool)
    support = np.zeros(count, dtype=np.int64)
    reasons = ["INSUFFICIENT_SUPPORT"] * count
    position = 0
    for lead_index, _lead in enumerate(LEADS):
        base = lead_index * EEG_BLOCK_WIDTH
        for band in BANDS:
            band_column = base + BAND_OFFSETS[band]
            beta_column = base + BETA_OFFSET
            p0_column = base + P0_OFFSET
            series = relative_eeg_power(
                x_seq[:, band_column], x_seq[:, beta_column], x_seq[:, p0_column]
            )
            finite = (
                np.isfinite(x_seq[:, band_column])
                & np.isfinite(x_seq[:, beta_column])
                & np.isfinite(x_seq[:, p0_column])
                & np.isfinite(series)
            )
            for scope in STAGE_SCOPES:
                mask = stage_mask(stages, stage_valid, scope)
                if not bool(available[lead_index]):
                    selected = np.zeros(len(stages), dtype=bool)
                    missing_reason = "CHANNEL_UNAVAILABLE"
                else:
                    selected = mask & (clean[:, lead_index] > 0) & finite
                    missing_reason = "INSUFFICIENT_SUPPORT"
                selected_values = np.asarray(series[selected], dtype=np.float64)
                n_valid = len(selected_values)
                for quantile in quantiles:
                    support[position] = n_valid
                    reasons[position] = missing_reason
                    if n_valid >= minimum_support:
                        values[position] = linear_quantile(selected_values, quantile)
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
            "minimum_support": minimum_support,
            "quantiles": list(quantiles),
        },
    )


EEG_TAIL_FEATURES = _definitions()
EEG_TAIL_SPEC = FamilySpec(
    family_id=FAMILY_ID,
    family_version=FAMILY_VERSION,
    extractor_symbol="extensions.eeg_tail.extract_family",
    default_config={
        "minimum_support": DEFAULT_MINIMUM_SUPPORT,
        "quantiles": [item[1] for item in QUANTILES],
    },
    input_contract=FamilyInputContract(
        family_id=FAMILY_ID,
        contract_version="1",
        requirements=(
            FieldRequirement("X_seq", ("T", 483), ("float32",), columns=tuple(range(54))),
            FieldRequirement("stage_code_aligned", ("T",), ("float32",)),
            FieldRequirement("stage_valid", ("T",), ("bool",)),
            FieldRequirement("eeg_channel_available", (6,), ("bool",)),
            FieldRequirement("eeg_clean_subsegment_count", ("T", 6), ("int32",)),
        ),
        validity_rules=(
            "stage membership is authoritative",
            "lead availability and positive clean-subsegment count exclude invalid zero fills",
        ),
    ),
    feature_definitions=EEG_TAIL_FEATURES,
    numerical_symbols=(
        "extensions.eeg_tail.extract_family",
        "extensions.eeg_tail._definitions",
        "extensions.common.linear_quantile",
        "extensions.common.relative_eeg_power",
        "extensions.common.stage_mask",
        "extensions.contracts.FeatureResult",
        "extensions.registry.FeatureDefinition",
    ) + SOURCE_ADAPTER_NUMERICAL_SYMBOLS,
    numerical_constants=(
        "extensions.eeg_tail.FAMILY_ID",
        "extensions.eeg_tail.FAMILY_VERSION",
        "extensions.eeg_tail.LEADS",
        "extensions.eeg_tail.BANDS",
        "extensions.eeg_tail.BAND_OFFSETS",
        "extensions.eeg_tail.STAGE_SCOPES",
        "extensions.eeg_tail.QUANTILES",
        "extensions.eeg_tail.EEG_BLOCK_WIDTH",
        "extensions.eeg_tail.BETA_OFFSET",
        "extensions.eeg_tail.P0_OFFSET",
        "extensions.eeg_tail.DEFAULT_MINIMUM_SUPPORT",
        "extensions.common.STAGE_CODES",
    ) + SOURCE_ADAPTER_NUMERICAL_CONSTANTS,
    numerical_dependencies={"numpy": np.__version__},
    extractor=extract_family,
)
