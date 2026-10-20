"""Deterministic public candidate manifest for FeatureLab Batch 1."""
from __future__ import annotations

import csv
from pathlib import Path

from .contracts import canonical_json
from .registry import list_families

MANIFEST_COLUMNS = (
    "feature_id",
    "family_id",
    "modality",
    "proposed_view",
    "source_file_or_field",
    "source_column",
    "stage_scope",
    "channel_or_pair",
    "physiological_meaning",
    "mathematical_formula",
    "parameters",
    "unit",
    "minimum_support",
    "validity_expression",
    "recipe_version",
    "external_reference",
    "implementation_status",
)
FROZEN_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "reproduction/siren_a/docs/G1B_FEATURE_MANIFEST_FROZEN.csv"
)
FAMILY_MODALITY = {
    "eeg_tail": "EEG",
    "eeg_temporal": "EEG",
    "eeg_coherence_stage": "EEG",
    "caisr_stage": "CAISR",
    "emg_stage": "EMG",
    "resp_stage": "Resp",
}
FAMILY_REFERENCE = {
    "eeg_tail": "FEATURELAB_V2_SPEC.md#51-eeg-tail-quantiles-288-columns",
    "eeg_temporal": "FEATURELAB_V2_SPEC.md#52-eeg-overnight-temporal-organization-96-columns",
    "eeg_coherence_stage": "FEATURELAB_V2_SPEC.md#53-stage-specific-eeg-coherence-90-columns",
    "caisr_stage": "FEATURELAB_V2_SPEC.md#54-caisr-stage-dynamics-35-columns",
    "emg_stage": "FEATURELAB_V2_SPEC.md#55-stage-specific-emg-72-columns",
    "resp_stage": "FEATURELAB_V2_SPEC.md#56-stage-specific-resp-56-columns",
}


def frozen_feature_ids(path: Path = FROZEN_MANIFEST) -> frozenset[str]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return frozenset(row["unique_feature_id"] for row in csv.DictReader(handle))


def _manifest_context(family_id: str, feature_id: str) -> tuple[str, str, str]:
    parts = feature_id.split(".")
    if family_id == "eeg_tail":
        stage = parts[4]
        view = {"wake": "V1", "rem": "V2", "n2": "V3", "n3": "V3", "nrem": "V3", "all-stage": "V4"}[stage]
        return view, stage, parts[2]
    if family_id == "eeg_temporal":
        return "V4", "Sleep", parts[2]
    if family_id == "eeg_coherence_stage":
        return "V5", parts[4], parts[2]
    if family_id == "caisr_stage":
        if parts[2] == "transition":
            return "V5", "All-stage", parts[3]
        return "V5", parts[3], "CAISR_stage"
    if family_id == "emg_stage":
        return "V5", parts[4], parts[2]
    if family_id == "resp_stage":
        return "V5", parts[-2], parts[2]
    raise ValueError(f"Unknown family: {family_id}")


def build_manifest_rows() -> list[dict[str, str]]:
    rows = []
    for spec in list_families():
        for index, definition in enumerate(spec.feature_definitions):
            view, stage, channel = _manifest_context(spec.family_id, definition.feature_id)
            status = (
                "REUSED_FROM_EXISTING_STATIC"
                if spec.family_id == "caisr_stage" and index >= 25
                else "COMPUTED_FROM_EXISTING_NPZ"
            )
            row = {
                "feature_id": definition.feature_id,
                "family_id": spec.family_id,
                "modality": FAMILY_MODALITY[spec.family_id],
                "proposed_view": view,
                "source_file_or_field": definition.source_field,
                "source_column": ";".join(str(item) for item in definition.source_columns),
                "stage_scope": stage,
                "channel_or_pair": channel,
                "physiological_meaning": (
                    f"{channel} {stage} {definition.formula_id.replace(chr(95), chr(32))}"
                ),
                "mathematical_formula": definition.formula_id,
                "parameters": canonical_json(dict(definition.parameters)),
                "unit": definition.unit,
                "minimum_support": str(definition.minimum_support),
                "validity_expression": definition.validity_expression,
                "recipe_version": spec.family_version,
                "external_reference": FAMILY_REFERENCE[spec.family_id],
                "implementation_status": status,
            }
            rows.append(row)
    return rows


def write_manifest(path: Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = build_manifest_rows()
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
