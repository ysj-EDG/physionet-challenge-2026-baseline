"""Stage descriptions derived only from the frozen Resp14 block."""
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

FAMILY_ID = "resp_stage"
FAMILY_VERSION = "1.0.0"
FEATURES = (
    ("airflow_rate_bpm", "beats_per_minute"),
    ("airflow_cycle_cv", "ratio"),
    ("airflow_amp_local_norm", "ratio"),
    ("airflow_amp_iqr_over_median", "ratio"),
    ("airflow_reduction30_fraction", "fraction"),
    ("airflow_near_absent_fraction", "fraction"),
    ("airflow_longest_reduction_sec", "seconds"),
    ("thorax_amp_local_norm", "ratio"),
    ("thorax_reduction30_fraction", "fraction"),
    ("abdomen_amp_local_norm", "ratio"),
    ("abdomen_reduction30_fraction", "fraction"),
    ("thorax_abd_corr", "correlation"),
    ("thorax_abd_lag_sec", "seconds"),
    ("thorax_abd_paradox_fraction", "fraction"),
)
STAGE_SCOPES = ("REM", "NREM")
STATISTICS = ("mean", "iqr")
RESP_START = 456
DEFAULT_MINIMUM_SUPPORT = 5


def _definitions() -> tuple[FeatureDefinition, ...]:
    rows = []
    for offset, (feature, unit) in enumerate(FEATURES):
        column = RESP_START + offset
        for scope in STAGE_SCOPES:
            for statistic in STATISTICS:
                rows.append(
                    FeatureDefinition(
                        feature_id=(
                            f"featurelab_v2.resp_stage.{feature}.{scope.lower()}.{statistic}"
                        ),
                        source_field="X_seq",
                        source_columns=(column,),
                        formula_id=f"stage_{statistic}_v1",
                        parameters={},
                        validity_expression="stage & resp_feature_valid[column] & finite(value)",
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
    feature_valid = np.asarray(source.arrays["resp_feature_valid"])

    position = 0
    for offset, (_feature, _unit) in enumerate(FEATURES):
        column = RESP_START + offset
        series = np.asarray(x_seq[:, column], dtype=np.float64)
        for scope in STAGE_SCOPES:
            mask = (
                stage_mask(stages, stage_valid, scope)
                & feature_valid[:, offset]
                & np.isfinite(series)
            )
            mean, iqr, n_valid = pool_mean_iqr(series, mask, minimum_support)
            for statistic_value in (mean, iqr):
                support[position] = n_valid
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


RESP_STAGE_FEATURES = _definitions()
RESP_STAGE_SPEC = FamilySpec(
    family_id=FAMILY_ID,
    family_version=FAMILY_VERSION,
    extractor_symbol="extensions.resp_stage.extract_family",
    default_config={"minimum_support": DEFAULT_MINIMUM_SUPPORT},
    input_contract=FamilyInputContract(
        family_id=FAMILY_ID,
        contract_version="1",
        requirements=(
            FieldRequirement("X_seq", ("T", 483), ("float32",), columns=tuple(range(456, 470))),
            FieldRequirement("stage_code_aligned", ("T",), ("float32",)),
            FieldRequirement("stage_valid", ("T",), ("bool",)),
            FieldRequirement("resp_feature_valid", ("T", 14), ("bool",), columns=tuple(range(14))),
        ),
        validity_rules=("per-feature validity excludes upstream invalid zero fills",),
    ),
    feature_definitions=RESP_STAGE_FEATURES,
    numerical_symbols=(
        "extensions.resp_stage.extract_family",
        "extensions.resp_stage._definitions",
        "extensions.common.pool_mean_iqr",
        "extensions.common.stage_mask",
        "extensions.contracts.FeatureResult",
        "extensions.registry.FeatureDefinition",
    ) + SOURCE_ADAPTER_NUMERICAL_SYMBOLS,
    numerical_constants=(
        "extensions.resp_stage.FAMILY_ID",
        "extensions.resp_stage.FAMILY_VERSION",
        "extensions.resp_stage.FEATURES",
        "extensions.resp_stage.STAGE_SCOPES",
        "extensions.resp_stage.STATISTICS",
        "extensions.resp_stage.RESP_START",
        "extensions.resp_stage.DEFAULT_MINIMUM_SUPPORT",
        "extensions.common.STAGE_CODES",
    ) + SOURCE_ADAPTER_NUMERICAL_CONSTANTS,
    numerical_dependencies={"numpy": np.__version__},
    extractor=extract_family,
)
