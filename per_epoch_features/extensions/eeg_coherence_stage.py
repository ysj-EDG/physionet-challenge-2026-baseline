"""Stage pooling of selected frozen coherence360 mean-band columns."""
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

FAMILY_ID = "eeg_coherence_stage"
FAMILY_VERSION = "1.0.0"
PAIRS = (("F3-F4", 0, (0, 1)), ("C3-C4", 9, (2, 3)), ("O1-O2", 14, (4, 5)))
BANDS = ("delta", "theta", "alpha", "sigma", "beta")
STAGE_SCOPES = ("REM", "N2", "N3")
STATISTICS = ("mean", "iqr")
COHERENCE_START = 54
PAIR_BLOCK_WIDTH = 24
DEFAULT_MINIMUM_SUPPORT = 5


def _definitions() -> tuple[FeatureDefinition, ...]:
    rows = []
    for pair, pair_position, _lead_indices in PAIRS:
        for band_offset, band in enumerate(BANDS):
            column = COHERENCE_START + pair_position * PAIR_BLOCK_WIDTH + band_offset
            for scope in STAGE_SCOPES:
                for statistic in STATISTICS:
                    rows.append(
                        FeatureDefinition(
                            feature_id=(
                                f"featurelab_v2.eeg_coherence_stage.{pair.lower()}."
                                f"{band}.{scope.lower()}.{statistic}"
                            ),
                            source_field="X_seq",
                            source_columns=(column,),
                            formula_id=f"stage_{statistic}_v1",
                            parameters={},
                            validity_expression=(
                                "stage_valid & pair leads available & common_clean>0 & finite(value)"
                            ),
                            minimum_support=DEFAULT_MINIMUM_SUPPORT,
                            unit="coherence",
                        )
                    )
    return tuple(rows)


def extract_family(source, config) -> FeatureResult:
    minimum_support = int(config["minimum_support"])
    definitions = _definitions()
    values = np.full(len(definitions), np.nan, dtype=np.float64)
    valid = np.zeros(len(definitions), dtype=bool)
    support = np.zeros(len(definitions), dtype=np.int64)
    reasons = ["INSUFFICIENT_SUPPORT"] * len(definitions)
    x_seq = np.asarray(source.arrays["X_seq"])
    stages = np.asarray(source.arrays["stage_code_aligned"])
    stage_valid = np.asarray(source.arrays["stage_valid"])
    available = np.asarray(source.arrays["eeg_channel_available"])
    common_clean = np.asarray(source.arrays["eeg_common_clean_subsegment_count"])

    position = 0
    for _pair, pair_position, lead_indices in PAIRS:
        pair_available = bool(available[lead_indices[0]] and available[lead_indices[1]])
        for band_offset, _band in enumerate(BANDS):
            column = COHERENCE_START + pair_position * PAIR_BLOCK_WIDTH + band_offset
            series = np.asarray(x_seq[:, column], dtype=np.float64)
            for scope in STAGE_SCOPES:
                mask = (
                    stage_mask(stages, stage_valid, scope)
                    & (common_clean > 0)
                    & np.isfinite(series)
                )
                if not pair_available:
                    mask = np.zeros(len(stages), dtype=bool)
                mean, iqr, n_valid = pool_mean_iqr(series, mask, minimum_support)
                for statistic_value in (mean, iqr):
                    support[position] = n_valid
                    reasons[position] = (
                        "PAIR_UNAVAILABLE" if not pair_available else "INSUFFICIENT_SUPPORT"
                    )
                    if np.isfinite(statistic_value):
                        values[position] = statistic_value
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
        processing_parameters={"minimum_support": minimum_support},
    )


EEG_COHERENCE_STAGE_FEATURES = _definitions()
EEG_COHERENCE_STAGE_SPEC = FamilySpec(
    family_id=FAMILY_ID,
    family_version=FAMILY_VERSION,
    extractor_symbol="extensions.eeg_coherence_stage.extract_family",
    default_config={"minimum_support": DEFAULT_MINIMUM_SUPPORT},
    input_contract=FamilyInputContract(
        family_id=FAMILY_ID,
        contract_version="1",
        requirements=(
            FieldRequirement(
                "X_seq",
                ("T", 483),
                ("float32",),
                columns=tuple(
                    COHERENCE_START + pair_position * PAIR_BLOCK_WIDTH + band_offset
                    for _pair, pair_position, _leads in PAIRS
                    for band_offset in range(5)
                ),
            ),
            FieldRequirement("stage_code_aligned", ("T",), ("float32",)),
            FieldRequirement("stage_valid", ("T",), ("bool",)),
            FieldRequirement("eeg_channel_available", (6,), ("bool",)),
            FieldRequirement("eeg_common_clean_subsegment_count", ("T",), ("int32",)),
        ),
        validity_rules=(
            "both physical leads must be available",
            "positive common-clean count excludes upstream invalid zero fills",
        ),
    ),
    feature_definitions=EEG_COHERENCE_STAGE_FEATURES,
    numerical_symbols=(
        "extensions.eeg_coherence_stage.extract_family",
        "extensions.eeg_coherence_stage._definitions",
        "extensions.common.pool_mean_iqr",
        "extensions.common.stage_mask",
        "extensions.contracts.FeatureResult",
        "extensions.registry.FeatureDefinition",
    ) + SOURCE_ADAPTER_NUMERICAL_SYMBOLS,
    numerical_constants=(
        "extensions.eeg_coherence_stage.FAMILY_ID",
        "extensions.eeg_coherence_stage.FAMILY_VERSION",
        "extensions.eeg_coherence_stage.PAIRS",
        "extensions.eeg_coherence_stage.BANDS",
        "extensions.eeg_coherence_stage.STAGE_SCOPES",
        "extensions.eeg_coherence_stage.STATISTICS",
        "extensions.eeg_coherence_stage.COHERENCE_START",
        "extensions.eeg_coherence_stage.PAIR_BLOCK_WIDTH",
        "extensions.eeg_coherence_stage.DEFAULT_MINIMUM_SUPPORT",
        "extensions.common.STAGE_CODES",
    ) + SOURCE_ADAPTER_NUMERICAL_CONSTANTS,
    numerical_dependencies={"numpy": np.__version__},
    extractor=extract_family,
)
