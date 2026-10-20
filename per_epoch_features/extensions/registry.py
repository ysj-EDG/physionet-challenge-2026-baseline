"""Declarative FeatureLab family specifications and ordered registry."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .contracts import FamilyInputContract, FeatureResult, SourceRecord

FamilyExtractor = Callable[[SourceRecord, Mapping[str, object]], FeatureResult]


@dataclass(frozen=True)
class FeatureDefinition:
    feature_id: str
    source_field: str
    source_columns: tuple[int, ...]
    formula_id: str
    parameters: Mapping[str, Any]
    validity_expression: str
    minimum_support: int
    unit: str

    def material(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "source_field": self.source_field,
            "source_columns": list(self.source_columns),
            "formula_id": self.formula_id,
            "parameters": dict(self.parameters),
            "validity_expression": self.validity_expression,
            "minimum_support": self.minimum_support,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    family_version: str
    extractor_symbol: str
    default_config: Mapping[str, Any]
    input_contract: FamilyInputContract
    feature_definitions: tuple[FeatureDefinition, ...]
    numerical_symbols: tuple[str, ...]
    numerical_constants: tuple[str, ...]
    numerical_dependencies: Mapping[str, str]
    extractor: FamilyExtractor | None = None


_FAMILIES: tuple[FamilySpec, ...] = ()
_IMPLEMENTED_FAMILY_MODULES = (
    "extensions.eeg_tail",
    "extensions.eeg_temporal",
    "extensions.eeg_coherence_stage",
    "extensions.caisr_stage",
    "extensions.emg_stage",
    "extensions.resp_stage",
)
_FAMILY_EXPORTS = (
    ("extensions.eeg_tail", "EEG_TAIL_SPEC"),
    ("extensions.eeg_temporal", "EEG_TEMPORAL_SPEC"),
    ("extensions.eeg_coherence_stage", "EEG_COHERENCE_STAGE_SPEC"),
    ("extensions.caisr_stage", "CAISR_STAGE_SPEC"),
    ("extensions.emg_stage", "EMG_STAGE_SPEC"),
    ("extensions.resp_stage", "RESP_STAGE_SPEC"),
)


def list_families() -> tuple[FamilySpec, ...]:
    if _FAMILIES:
        return _FAMILIES
    return tuple(
        getattr(importlib.import_module(module), export)
        for module, export in _FAMILY_EXPORTS
    )


def get_family(family_id: str) -> FamilySpec:
    for spec in list_families():
        if spec.family_id == family_id:
            return spec
    raise KeyError(f"Unknown feature family: {family_id}")
