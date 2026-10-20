"""CAISR transition dynamics and verified reuse of frozen static bout summaries."""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .common import linear_quantile
from .contracts import FamilyInputContract, FeatureResult, FieldRequirement
from .registry import (
    SOURCE_ADAPTER_NUMERICAL_CONSTANTS,
    SOURCE_ADAPTER_NUMERICAL_SYMBOLS,
    FamilySpec,
    FeatureDefinition,
)

FAMILY_ID = "caisr_stage"
FAMILY_VERSION = "1.0.0"
STAGE_CODES = (("N3", 1), ("N2", 2), ("N1", 3), ("REM", 4), ("Wake", 5))
STAGE_VALUES = (1, 2, 3, 4, 5)
STATIC_P50 = {5: 44, 3: 45, 2: 46, 1: 47, 4: 48}
STATIC_P95 = {5: 69, 3: 70, 2: 71, 1: 72, 4: 73}
EPOCH_SECONDS = 30.0
DEFAULT_STATIC_TOLERANCE = 1e-5


def bout_lengths_by_stage(
    stage_code: np.ndarray,
    stage_valid: np.ndarray,
    epoch_start_sec: np.ndarray,
) -> Mapping[int, np.ndarray]:
    codes = np.asarray(stage_code)
    valid = np.asarray(stage_valid)
    starts = np.asarray(epoch_start_sec, dtype=np.float64)
    if codes.ndim != 1 or valid.shape != codes.shape or starts.shape != codes.shape:
        raise ValueError("stage and time vectors must have equal one-dimensional shape")
    if valid.dtype != np.dtype(bool):
        raise ValueError("stage_valid must be boolean")
    bouts = {code: [] for code in STAGE_VALUES}
    active_code = None
    active_epochs = 0
    previous_index = None
    for index in range(len(codes)):
        code = int(codes[index]) if np.isfinite(codes[index]) else -1
        lawful = bool(valid[index] and code in STAGE_VALUES and np.isfinite(starts[index]))
        contiguous = (
            lawful
            and active_code == code
            and previous_index is not None
            and index == previous_index + 1
            and starts[index] - starts[previous_index] == EPOCH_SECONDS
        )
        if contiguous:
            active_epochs += 1
        else:
            if active_code is not None:
                bouts[active_code].append(active_epochs * EPOCH_SECONDS)
            active_code = code if lawful else None
            active_epochs = 1 if lawful else 0
        previous_index = index if lawful else None
    if active_code is not None:
        bouts[active_code].append(active_epochs * EPOCH_SECONDS)
    return {
        code: np.asarray(lengths, dtype=np.float64)
        for code, lengths in bouts.items()
    }


def _definitions() -> tuple[FeatureDefinition, ...]:
    rows = []
    for source_name, source_code in STAGE_CODES:
        for destination_name, _destination_code in STAGE_CODES:
            rows.append(
                FeatureDefinition(
                    feature_id=(
                        f"featurelab_v2.caisr_stage.transition."
                        f"{source_name.lower()}_to_{destination_name.lower()}"
                    ),
                    source_field="stage_code_aligned",
                    source_columns=(),
                    formula_id="source_normalized_adjacent_transition_v1",
                    parameters={"source_stage": source_code},
                    validity_expression="both stages valid and exactly 30 seconds adjacent",
                    minimum_support=1,
                    unit="probability",
                )
            )
    for stage_name, stage_code in STAGE_CODES:
        for statistic, mapping in (("p50", STATIC_P50), ("p95", STATIC_P95)):
            rows.append(
                FeatureDefinition(
                    feature_id=(
                        f"featurelab_v2.caisr_stage.bout.{stage_name.lower()}.{statistic}"
                    ),
                    source_field="x_static",
                    source_columns=(mapping[stage_code],),
                    formula_id="reuse_static_bout_quantile_v1",
                    parameters={"quantile": 0.50 if statistic == "p50" else 0.95},
                    validity_expression=(
                        "annotation available & stage bout support & static/grid semantic match"
                    ),
                    minimum_support=1,
                    unit="seconds",
                )
            )
    return tuple(rows)


def extract_family(source, config) -> FeatureResult:
    tolerance = float(config["static_tolerance"])
    definitions = _definitions()
    values = np.full(len(definitions), np.nan, dtype=np.float64)
    valid_out = np.zeros(len(definitions), dtype=bool)
    support = np.zeros(len(definitions), dtype=np.int64)
    reasons = ["NO_LEGAL_TRANSITIONS"] * len(definitions)
    stages = np.asarray(source.arrays["stage_code_aligned"])
    stage_valid = np.asarray(source.arrays["stage_valid"])
    starts = np.asarray(source.arrays["epoch_start_sec"], dtype=np.float64)
    x_static = np.asarray(source.arrays["x_static"])
    annotation_status = str(np.asarray(source.arrays["annotation_status"]).item())

    legal = (
        stage_valid[:-1]
        & stage_valid[1:]
        & np.isfinite(stages[:-1])
        & np.isfinite(stages[1:])
        & np.isin(stages[:-1], STAGE_VALUES)
        & np.isin(stages[1:], STAGE_VALUES)
        & np.isfinite(starts[:-1])
        & np.isfinite(starts[1:])
        & (np.diff(starts) == EPOCH_SECONDS)
    )
    position = 0
    for _source_name, source_code in STAGE_CODES:
        source_pairs = legal & (stages[:-1] == source_code)
        outgoing = int(np.sum(source_pairs))
        for _destination_name, destination_code in STAGE_CODES:
            support[position] = outgoing
            if outgoing > 0:
                values[position] = float(
                    np.sum(source_pairs & (stages[1:] == destination_code)) / outgoing
                )
                valid_out[position] = True
                reasons[position] = ""
            position += 1

    bouts = bout_lengths_by_stage(stages, stage_valid, starts)
    for _stage_name, stage_code in STAGE_CODES:
        stage_bouts = bouts[stage_code]
        n_bouts = len(stage_bouts)
        for quantile, mapping in ((0.50, STATIC_P50), (0.95, STATIC_P95)):
            support[position] = n_bouts
            if annotation_status != "available":
                reasons[position] = "ANNOTATION_UNAVAILABLE"
            elif n_bouts == 0:
                reasons[position] = "NO_STAGE_BOUT_SUPPORT"
            else:
                stored = float(x_static[mapping[stage_code]])
                expected = linear_quantile(stage_bouts, quantile)
                if not np.isfinite(stored) or not np.isclose(
                    stored, expected, rtol=tolerance, atol=tolerance
                ):
                    reasons[position] = "STATIC_BOUT_SEMANTIC_MISMATCH"
                else:
                    values[position] = stored
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
        processing_parameters={"static_tolerance": tolerance},
    )


CAISR_STAGE_FEATURES = _definitions()
CAISR_STAGE_SPEC = FamilySpec(
    family_id=FAMILY_ID,
    family_version=FAMILY_VERSION,
    extractor_symbol="extensions.caisr_stage.extract_family",
    default_config={"static_tolerance": DEFAULT_STATIC_TOLERANCE},
    input_contract=FamilyInputContract(
        family_id=FAMILY_ID,
        contract_version="1",
        requirements=(
            FieldRequirement("stage_code_aligned", ("T",), ("float32",)),
            FieldRequirement("stage_valid", ("T",), ("bool",)),
            FieldRequirement("epoch_start_sec", ("T",), ("float32",)),
            FieldRequirement("x_static", (196,), ("float32",), columns=tuple(range(44, 49)) + tuple(range(69, 74))),
            FieldRequirement("annotation_status", (), ("<U7", "<U9")),
        ),
        stage_time_rules=("invalid epochs and non-30-second gaps terminate bouts",),
        validity_rules=("annotation available and reconstructed bout support",),
    ),
    feature_definitions=CAISR_STAGE_FEATURES,
    numerical_symbols=(
        "extensions.caisr_stage.extract_family",
        "extensions.caisr_stage.bout_lengths_by_stage",
        "extensions.caisr_stage._definitions",
        "extensions.common.linear_quantile",
        "extensions.contracts.FeatureResult",
        "extensions.registry.FeatureDefinition",
    ) + SOURCE_ADAPTER_NUMERICAL_SYMBOLS,
    numerical_constants=(
        "extensions.caisr_stage.FAMILY_ID",
        "extensions.caisr_stage.FAMILY_VERSION",
        "extensions.caisr_stage.STAGE_CODES",
        "extensions.caisr_stage.STAGE_VALUES",
        "extensions.caisr_stage.STATIC_P50",
        "extensions.caisr_stage.STATIC_P95",
        "extensions.caisr_stage.EPOCH_SECONDS",
        "extensions.caisr_stage.DEFAULT_STATIC_TOLERANCE",
    ) + SOURCE_ADAPTER_NUMERICAL_CONSTANTS,
    numerical_dependencies={"numpy": np.__version__},
    extractor=extract_family,
)
