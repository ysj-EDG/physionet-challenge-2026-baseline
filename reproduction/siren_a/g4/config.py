"""Frozen Baseline v1 configuration. No CLI hyperparameter overrides."""

from pathlib import Path

SEED = 42
ALL_SEEDS = tuple(range(42, 57))
TRAIN_SITES = ("S0001", "I0006")
VIEWS = ("V1", "V2", "V3", "V4", "V5")
VIEW_SIZES = {"V1": 113, "V2": 129, "V3": 349, "V4": 113, "V5": 123}
MODEL_TYPES = ("lr", "hgb", "lambdarank")
PROTOCOLS = ("transductive", "inductive")
HOLDOUTS = ("I0002", "I0006", "S0001")
HOLDOUT_TO_TRAIN = {"I0002": "S0001", "I0006": "S0001", "S0001": "I0006"}
CANONICAL_HOLDOUT = {"S0001": "I0002", "I0006": "S0001"}
EXPECTED_TRAIN = {"S0001": 5139, "I0006": 1142}
EXPECTED_SELECTED = {"S0001": 3597, "I0006": 799}

MANIFEST_SHA256 = "7aef4e0462936ad0fbb3ee504a71062b0f50d82de5c54e1d70df2d13691e455a"
SPEC_SHA256 = "f9f6ae6b7fd3126569914dad9c643c17ee9fa56354c460c694f46060cb174d9a"
G2_FINGERPRINT = "cf88b4e87f8b524258c685db696c278198dae1e6a10bd39fb63e893b2df031df"
G3_FINGERPRINT = "4a0f3956432030e4fa2fb8b51d15aa8fa5269f990d7795cca40d580970920a64"
DEMOGRAPHICS_SHA256 = "b9009598589846e84462636bfcaa290ed1d8666c89b1bc1ad1a9eda26b43ff7f"
SCHEMA = "g4a_seed42_v1"
SAMPLER_VERSION = "age_label_70pct_pcg64_v1"
ENV_PYTHON = Path("/database/home/gaohaojie/.conda/envs/sleepfm_env/bin/python")
PRIVATE_ROOT = Path("/database/home/gaohaojie/workspace/python-example-2026-siren-a-g4a-private/seed42_smoke_v1")
ENV_AUDIT_ROOT = Path("/database/home/gaohaojie/workspace/python-example-2026-siren-a-g4a-private/environment_sleepfm_lgbm460")

LR_PARAMS = dict(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, tol=1e-4,
                 fit_intercept=True, class_weight=None, random_state=42)
HGB_PARAMS = dict(loss="log_loss", learning_rate=0.05, max_iter=250, max_leaf_nodes=15,
                  max_depth=3, min_samples_leaf=30, l2_regularization=1.0, max_bins=255,
                  early_stopping=False, class_weight=None, random_state=42)
LGBM_PARAMS = dict(objective="lambdarank", n_estimators=250, learning_rate=0.05,
                   num_leaves=15, max_depth=3, min_child_samples=30, reg_lambda=1.0,
                   colsample_bytree=1.0, subsample=1.0, subsample_freq=0,
                   label_gain=[0, 1], lambdarank_norm=True, random_state=42,
                   n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True)
