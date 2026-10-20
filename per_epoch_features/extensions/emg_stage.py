"""Stage descriptions derived only from the frozen EMG24 block."""
from __future__ import annotations

import numpy as np

from .common import pool_mean_iqr, stage_mask
from .contracts import FamilyInputContract, FeatureResult, FieldRequirement
from .registry import (
    SOURCE_ADAPTER_NUMERICAL_CONSTANTS,
    SOURCE_ADAPTER_NUMERICAL_SYMBOLS,
    FamilySpec,
    FeatureDefinition,
)

FAMILY_ID = "emg_stage"
FAMILY_VERSION = "1.0.0"
CHANNELS = (("chin", 0), ("left_leg", 1), ("right_leg", 2))
FEATURES = (
    ("log_rms_norm", 0, "log_ratio"),
    ("tonic_fraction", 3, "fraction"),
    ("burst_rate_per_min", 4, "per_minute"),
    ("phasic_mini_epoch_fraction", 6, "fraction"),
)
STAGE_SCOPES = ("Wake", "REM", "NREM")
STATISTICS = ("mean", "iqr")
EMG_START = 432
CHANNEL_BLOCK_WIDTH = 8
DEFAULT_MINIMUM_SUPPORT = 5


def _definitions() -> tuple[FeatureDefinition, ...]:
    rows = []
    for channel, channel_index in CHANNELS:
        for feature, offset, unit in FEATURES:
            column = EMG_START + channel_index * CHANNEL_BLOCK_WIDTH + offset
            for scope in STAGE_SCOPES:
                for statistic in STATISTICS:
                    rows.append(
                        FeatureDefinition(
                            feature_id=(
                                f"featurelab_v2.emg_stage.{channel}.{feature}."
                                f"{scope.lower()}.{statistic}"
                            ),
                            source_field="X_seq",
                            source_columns=(column,),
                            formula_id=f"stage_{statistic}_v1",
                            parameters={},
                            validity_expression=(
                                "stage & channel_available & preprocessing_success & "
                                "epoch_success & finite(value)"
                            ),
                            minimum_support=DEFAULT_MINIMUM_SUPPORT,
                            unit=unit,
                        )
                    )
    return tuple(rows)


def extract_family(source, config) -> FeatureResult:
    minimum_support = int(config["minimum_support"])
    definitions = _definitions()
    values = np.full(len(definitions), np.nan, dtype=np.float64)
    valid_out = np.zeros(len(definitions), dtype=bool)
    support = np.zeros(len(definitions), dtype=np.int64)
    reasons = ["INSUFFICIENT_SUPPORT"] * len(definitions)
    x_seq = np.asarray(source.arrays["X_seq"])
    stages = np.asarray(source.arrays["stage_code_aligned"])
    stage_valid = np.asarray(source.arrays["stage_valid"])
    available = np.asarray(source.arrays["emg_channel_available"])
    preprocessing = np.asarray(source.arrays["emg_preprocessing_success"])
    epoch_success = np.asarray(source.arrays["emg_epoch_success"])

    position = 0
    for _channel, channel_index in CHANNELS:
        channel_available = bool(available[channel_index])
        preprocessing_success = bool(preprocessing[channel_index])
        for _feature, offset, _unit in FEATURES:
            column = EMG_START + channel_index * CHANNEL_BLOCK_WIDTH + offset
            series = np.asarray(x_seq[:, column], dtype=np.float64)
            for scope in STAGE_SCOPES:
                mask = (
                    stage_mask(stages, stage_valid, scope)
                    & epoch_success[:, channel_index]
                    & np.isfinite(series)
                )
                if not channel_available or not preprocessing_success:
                    mask = np.zeros(len(stages), dtype=bool)
                mean, iqr, n_valid = pool_mean_iqr(series, mask, minimum_support)
                for statistic_value in (mean, iqr):
                    support[position] = n_valid
                    if not channel_available:
                        reasons[position] = "CHANNEL_UNAVAILABLE"
                    elif not preprocessing_success:
                        reasons[position] = "PREPROCESSING_FAILED"
                    else:
                        reasons[position] = "INSUFFICIENT_SUPPORT"
                    if np.isfinite(statistic_value):
                        values[position] = statistic_value
                        valid_out[position] = True
                        reasons[position] = ""
                    position += 1
    return FeatureResult(
        family_id=FAMILY_ID,
        family_version=FAMILY_VERSION,
        feature_ids=tuple(item.feature_id for item in definitions),
        values=values,
        valid_mask=valid_out,
        support_count=support,
        missing_reason=tuple(reasons),
        feature_units=tuple(item.unit for item in definitions),
        source_identity=source.identity,
        processing_parameters={"minimum_support": minimum_support},
    )


EMG_STAGE_FEATURES = _definitions()
EMG_STAGE_SPEC = FamilySpec(
    family_id=FAMILY_ID,
    family_version=FAMILY_VERSION,
    extractor_symbol="extensions.emg_stage.extract_family",
    default_config={"minimum_support": DEFAULT_MINIMUM_SUPPORT},
    input_contract=FamilyInputContract(
        family_id=FAMILY_ID,
        contract_version="1",
        requirements=(
            FieldRequirement("X_seq", ("T", 483), ("float32",), columns=tuple(EMG_START + channel * CHANNEL_BLOCK_WIDTH + offset for channel in range(3) for _name, offset, _unit in FEATURES)),
            FieldRequirement("stage_code_aligned", ("T",), ("float32",)),
            FieldRequirement("stage_valid", ("T",), ("bool",)),
            FieldRequirement("emg_channel_available", (3,), ("bool",)),
            FieldRequirement("emg_preprocessing_success", (3,), ("bool",)),
            FieldRequirement("emg_epoch_success", ("T", 3), ("bool",)),
        ),
        validity_rules=(
            "per-channel availability and preprocessing success",
            "per-epoch channel success excludes upstream invalid zero fills",
        ),
    ),
    feature_definitions=EMG_STAGE_FEATURES,
    numerical_symbols=(
        "extensions.emg_stage.extract_family",
        "extensions.emg_stage._definitions",
        "extensions.common.pool_mean_iqr",
        "extensions.common.stage_mask",
        "extensions.contracts.FeatureResult",
        "extensions.registry.FeatureDefinition",
    ) + SOURCE_ADAPTER_NUMERICAL_SYMBOLS,
    numerical_constants=(
        "extensions.emg_stage.FAMILY_ID",
        "extensions.emg_stage.FAMILY_VERSION",
        "extensions.emg_stage.CHANNELS",
        "extensions.emg_stage.FEATURES",
        "extensions.emg_stage.STAGE_SCOPES",
        "extensions.emg_stage.STATISTICS",
        "extensions.emg_stage.EMG_START",
        "extensions.emg_stage.CHANNEL_BLOCK_WIDTH",
        "extensions.emg_stage.DEFAULT_MINIMUM_SUPPORT",
        "extensions.common.STAGE_CODES",
    ) + SOURCE_ADAPTER_NUMERICAL_CONSTANTS,
    numerical_dependencies={"numpy": np.__version__},
    extractor=extract_family,
)
