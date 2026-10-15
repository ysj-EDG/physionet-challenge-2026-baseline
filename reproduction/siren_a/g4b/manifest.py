"""G5-ready source index over immutable G4-A and G4-B artifacts (no scoring)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from reproduction.siren_a.g4.verify_g4a import sha256_file, verify_unit
from reproduction.siren_a.pipeline.run_inputs import atomic_json_no_clobber

from .config import (
    ALL_SEEDS,
    G4A_RUN_IDENTITY_SHA256,
    HOLDOUT_TO_TRAIN,
    full_model_identities,
    full_prediction_identities,
)


EXPECTED_TEST_COUNTS = {"I0002": 319, "I0006": 1142, "S0001": 5139}


def _model_key(row: Mapping[str, object]):
    return (row["train_site"], int(row["seed"]), row["view"], row["model_type"])


def _prediction_key(row: Mapping[str, object]):
    return (row["holdout_site"], row["protocol"], row["view"],
            row["model_type"], int(row["seed"]))


def validate_manifest_entries(
    models: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    expected_models = set(full_model_identities())
    expected_predictions = set(full_prediction_identities())
    model_keys = [_model_key(row) for row in models]
    prediction_keys = [_prediction_key(row) for row in predictions]
    if len(models) != 450 or len(set(model_keys)) != 450 or set(model_keys) != expected_models:
        raise ValueError("model manifest does not close 450 unique identities")
    if (len(predictions) != 1350 or len(set(prediction_keys)) != 1350
            or set(prediction_keys) != expected_predictions):
        raise ValueError("prediction manifest does not close 1350 unique identities")
    by_model = {_model_key(row): row for row in models}
    test_sha = {}
    unit_counts = Counter()
    for row in predictions:
        key = _prediction_key(row)
        holdout, protocol, view, model_type, seed = key
        train_site = HOLDOUT_TO_TRAIN[holdout]
        if row.get("train_site") != train_site:
            raise ValueError("prediction train-site trace mismatch")
        model = by_model[(train_site, seed, view, model_type)]
        for field in ("model_sha256", "adapter_sha256", "feature_order_sha256"):
            if row.get(field) != model.get(field):
                raise ValueError(f"prediction-to-model trace mismatch: {field}")
        wanted_score = "raw_ranking_score" if model_type == "lambdarank" else "positive_probability"
        if row.get("score_type") != wanted_score:
            raise ValueError("prediction score type mismatch")
        if int(row.get("prediction_length", -1)) != EXPECTED_TEST_COUNTS[holdout]:
            raise ValueError("prediction length mismatch")
        pair = (holdout, protocol)
        if pair in test_sha and test_sha[pair] != row.get("test_indices_sha256"):
            raise ValueError("test indices differ inside holdout/protocol")
        test_sha[pair] = row.get("test_indices_sha256")
        unit_counts[pair] += 1
        for field in ("prediction_reference", "prediction_sha256", "completion_sha256"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"prediction provenance missing: {field}")
    if set(unit_counts.values()) != {225} or len(unit_counts) != 6:
        raise ValueError("holdout/protocol does not contain 225 source vectors")
    for row in models:
        for field in ("checkpoint_reference", "model_sha256", "adapter_sha256",
                      "feature_order_sha256", "sample_sha256", "completion_sha256"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"model provenance missing: {field}")
    return {
        "status": "PASS", "models": len(models), "predictions": len(predictions),
        "seeds": len(set(int(row["seed"]) for row in models)),
        "prediction_units_per_holdout_protocol": {
            f"{key[0]}/{key[1]}": value for key, value in sorted(unit_counts.items())
        },
    }


def _artifact_roots(config, seed: int):
    return config.g4a_root if seed == 42 else config.private_root


def _model_path(config, site: str, seed: int, view: str, model_type: str) -> Path:
    if seed == 42:
        return config.g4a_root / "models" / site / "seed42" / view / model_type
    return config.private_root / "seeds" / f"seed{seed}" / "models" / site / view / model_type


def _prediction_path(config, holdout: str, protocol: str, view: str,
                     model_type: str, seed: int) -> Path:
    if seed == 42:
        return (config.g4a_root / "predictions" / f"holdout_{holdout}"
                / protocol / view / model_type)
    return (config.private_root / "seeds" / f"seed{seed}" / "predictions"
            / f"holdout_{holdout}" / protocol / view / model_type)


def collect_manifest_entries(config):
    models = []
    for site, seed, view, model_type in full_model_identities():
        path = _model_path(config, site, seed, view, model_type)
        if path.is_symlink() or not (path / "complete.json").is_file():
            raise ValueError("model completion evidence missing")
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        model_name = "model.txt" if model_type == "lambdarank" else "model.joblib"
        models.append({
            "source_gate": "G4-A" if seed == 42 else "G4-B",
            "train_site": site, "seed": seed, "view": view, "model_type": model_type,
            "checkpoint_reference": str(path / model_name),
            "model_sha256": sha256_file(path / model_name),
            "adapter_sha256": sha256_file(path / "adapter_state.npz"),
            "feature_order_sha256": metadata["feature_ids_sha256"],
            "sample_sha256": metadata["sample_sha256"],
            "completion_sha256": sha256_file(path / "complete.json"),
        })
    by_model = {_model_key(row): row for row in models}
    predictions = []
    for holdout, protocol, view, model_type, seed in full_prediction_identities():
        path = _prediction_path(config, holdout, protocol, view, model_type, seed)
        if path.is_symlink() or not (path / "complete.json").is_file():
            raise ValueError("prediction completion evidence missing")
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        site = HOLDOUT_TO_TRAIN[holdout]
        model = by_model[(site, seed, view, model_type)]
        predictions.append({
            "source_gate": "G4-A" if seed == 42 else "G4-B",
            "holdout_site": holdout, "train_site": site, "protocol": protocol,
            "view": view, "model_type": model_type, "seed": seed,
            "prediction_reference": str(path / "prediction.npz"),
            "prediction_sha256": metadata["prediction_sha256"],
            "test_indices_sha256": metadata["test_indices_sha256"],
            "score_type": metadata["score_type"],
            "prediction_length": metadata["test_count"],
            "model_sha256": model["model_sha256"],
            "adapter_sha256": model["adapter_sha256"],
            "feature_order_sha256": model["feature_order_sha256"],
            "completion_sha256": sha256_file(path / "complete.json"),
        })
    validate_manifest_entries(models, predictions)
    return models, predictions


def verify_g5_ready(config, run_sha: str):
    target = config.private_root / "g5_ready"
    identity = {
        "run_identity_sha256": run_sha,
        "g4a_run_identity_sha256": G4A_RUN_IDENTITY_SHA256,
        "models": 450, "predictions": 1350, "schema": "g4b_g5_manifest_v1",
    }
    verify_unit(target, "g4b_g5_manifest_v1", identity)
    model_payload = json.loads((target / "models_manifest.json").read_text(encoding="utf-8"))
    prediction_payload = json.loads((target / "predictions_manifest.json").read_text(encoding="utf-8"))
    result = validate_manifest_entries(model_payload["entries"], prediction_payload["entries"])
    current_models, current_predictions = collect_manifest_entries(config)
    if model_payload["entries"] != current_models or prediction_payload["entries"] != current_predictions:
        raise ValueError("G5 manifest no longer matches source artifacts")
    return result


def build_g5_ready(config, run_sha: str, build_unit):
    models, predictions = collect_manifest_entries(config)
    audit = validate_manifest_entries(models, predictions)
    target = config.private_root / "g5_ready"
    identity = {
        "run_identity_sha256": run_sha,
        "g4a_run_identity_sha256": G4A_RUN_IDENTITY_SHA256,
        "models": 450, "predictions": 1350, "schema": "g4b_g5_manifest_v1",
    }

    def writer(path):
        atomic_json_no_clobber(path / "models_manifest.json", {
            "schema": "g4b_models_manifest_v1", "source_g4a": G4A_RUN_IDENTITY_SHA256,
            "source_g4b": run_sha, "entries": models,
        })
        atomic_json_no_clobber(path / "predictions_manifest.json", {
            "schema": "g4b_predictions_manifest_v1", "source_g4a": G4A_RUN_IDENTITY_SHA256,
            "source_g4b": run_sha, "entries": predictions,
        })
        atomic_json_no_clobber(path / "coverage_audit.json", audit)

    action = build_unit(target, "g4b_g5_manifest_v1", identity, writer)
    verified = verify_g5_ready(config, run_sha)
    return {**verified, "action": action}


__all__ = [
    "build_g5_ready", "collect_manifest_entries", "validate_manifest_entries",
    "verify_g5_ready",
]
