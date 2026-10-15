"""Frozen G4-B schedule, paths, and seed-aware Baseline v1 parameters."""

from pathlib import Path
from typing import Dict, Tuple

from reproduction.siren_a.g4.config import (
    CANONICAL_HOLDOUT,
    DEMOGRAPHICS_SHA256,
    ENV_AUDIT_ROOT,
    ENV_PYTHON,
    EXPECTED_SELECTED,
    EXPECTED_TRAIN,
    G2_FINGERPRINT,
    G3_FINGERPRINT,
    HGB_PARAMS,
    HOLDOUTS,
    HOLDOUT_TO_TRAIN,
    LGBM_PARAMS,
    LR_PARAMS,
    MANIFEST_SHA256,
    MODEL_TYPES,
    PROTOCOLS,
    SAMPLER_VERSION,
    SPEC_SHA256,
    TRAIN_SITES,
    VIEW_SIZES,
    VIEWS,
)

NEW_SEEDS = tuple(range(43, 57))
ALL_SEEDS = tuple(range(42, 57))
G4A_SEED = 42
SCHEMA = "g4b_seeds43_56_v1"
PRIVATE_ROOT = Path(
    "/database/home/gaohaojie/workspace/python-example-2026-siren-a-g4b-private/seeds43_56_v1"
)
G4A_ROOT = Path(
    "/database/home/gaohaojie/workspace/python-example-2026-siren-a-g4a-private/seed42_smoke_v1"
)
G4A_RUN_IDENTITY_SHA256 = "4d9654cc36142b23aaa058a82a0b154f8b5d9557feb1c506354374138f13945c"
G3B_RUN_IDENTITY_SHA256 = "0936a4fcd91113f4ee76caa69c5cdfd3b501db87aa88b718d7a8b02ffccaf46d"
G3B_COMPLETE_SHA256 = "cd218356f047e39aee7950c2dab8c549616b8abaf075093bd8296e01b8320b57"
THREAD_ENV = {
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def new_model_identities() -> Tuple[Tuple[str, int, str, str], ...]:
    return tuple(
        (site, seed, view, model)
        for seed in NEW_SEEDS
        for site in TRAIN_SITES
        for view in VIEWS
        for model in MODEL_TYPES
    )


def new_prediction_identities() -> Tuple[Tuple[str, str, str, str, int], ...]:
    return tuple(
        (holdout, protocol, view, model, seed)
        for seed in NEW_SEEDS
        for holdout in HOLDOUTS
        for protocol in PROTOCOLS
        for view in VIEWS
        for model in MODEL_TYPES
    )


def full_model_identities() -> Tuple[Tuple[str, int, str, str], ...]:
    return tuple(
        (site, seed, view, model)
        for seed in ALL_SEEDS
        for site in TRAIN_SITES
        for view in VIEWS
        for model in MODEL_TYPES
    )


def full_prediction_identities() -> Tuple[Tuple[str, str, str, str, int], ...]:
    return tuple(
        (holdout, protocol, view, model, seed)
        for seed in ALL_SEEDS
        for holdout in HOLDOUTS
        for protocol in PROTOCOLS
        for view in VIEWS
        for model in MODEL_TYPES
    )


def parameters_for_seed(model_type: str, seed: int, truncation_level: int) -> Dict[str, object]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in NEW_SEEDS:
        raise ValueError("G4-B seed must be an integer in 43..56")
    if model_type == "lr":
        parameters = dict(LR_PARAMS)
    elif model_type == "hgb":
        parameters = dict(HGB_PARAMS)
    elif model_type == "lambdarank":
        if not isinstance(truncation_level, int) or truncation_level <= 0:
            raise ValueError("LambdaRank truncation must be positive")
        parameters = dict(LGBM_PARAMS)
        parameters["lambdarank_truncation_level"] = int(truncation_level)
    else:
        raise ValueError("unknown model type")
    parameters["random_state"] = seed
    return parameters


__all__ = [
    "ALL_SEEDS", "CANONICAL_HOLDOUT", "DEMOGRAPHICS_SHA256", "ENV_AUDIT_ROOT", "ENV_PYTHON",
    "EXPECTED_SELECTED", "EXPECTED_TRAIN", "G2_FINGERPRINT", "G3_FINGERPRINT",
    "G3B_COMPLETE_SHA256", "G3B_RUN_IDENTITY_SHA256", "G4A_ROOT",
    "G4A_RUN_IDENTITY_SHA256", "G4A_SEED", "HOLDOUTS", "HOLDOUT_TO_TRAIN",
    "MANIFEST_SHA256", "MODEL_TYPES", "NEW_SEEDS", "PRIVATE_ROOT", "PROTOCOLS",
    "SAMPLER_VERSION", "SCHEMA", "SPEC_SHA256", "THREAD_ENV", "TRAIN_SITES",
    "VIEW_SIZES", "VIEWS", "full_model_identities", "full_prediction_identities",
    "new_model_identities", "new_prediction_identities", "parameters_for_seed",
]
