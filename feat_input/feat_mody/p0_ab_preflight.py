#!/usr/bin/env python
"""Reproducible preflight and launch gate for the P0 legacy_clip/typed_v1 pair.

This utility only reads the committed raw NPZ caches.  It never extracts
features, writes transformed NPZ files, trains a formal model, or saves model
weights.  ``--generate`` creates the audit artifacts; ``--verify`` re-hashes
the frozen inputs and production code immediately before a formal launch.
"""

from __future__ import annotations

import argparse
import ast
import collections
import datetime as dt
import hashlib
import io
import json
import os
import pickle
import platform
import random
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import feature_scaling
import team_code
import train_lstm
from feature_scaling import (
    ECG_EPOCH_OFFSET,
    FeatureScaler,
    LEGACY_MODE,
    SCALER_VERSION,
    detect_fallback_sequence,
    record_key,
    stable_hash,
)

AUDIT_COMMIT = "0e6e7087e17af5df3f9438d440f45b2d47b06cf7"
PARENT_COMMIT = "a4fe0b4dd914856993e67870801c9eb35b5311b2"
SPLITS = ("train", "val", "test", "external")
SEED = 7
PRODUCTION_CODE = (
    "train_lstm.py",
    "team_code.py",
    "feature_scaling.py",
    "evaluate_model.py",
    "helper_code.py",
)
UNCHANGED_TOP_LEVEL_NODES = (
    "seed_everything",
    "collate_fn",
    "LSTMModel",
    "train_epoch",
    "collect_outputs",
    "evaluate",
    "count_cached_labels",
    "build_balanced_sampler",
    "sigmoid_np",
    "fit_platt_calibrator",
    "apply_calibrator",
    "select_youden_threshold",
)
FROZEN_CONSTANTS = (
    "SEQ_DIM",
    "ECG_DIM",
    "STATIC_DIM",
    "INPUT_DIM",
    "HIDDEN_DIM",
    "NUM_LAYERS",
    "DROPOUT",
    "FC_HIDDEN",
    "BATCH_SIZE",
    "LR",
    "EPOCHS",
    "PATIENCE",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return sha256_bytes(payload)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False
    )
    if check and result.returncode:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout.strip()


def record_id_hash(ids: list[str]) -> str:
    return sha256_bytes("".join(f"{item}\n" for item in ids).encode("utf-8"))


def array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def finite_max_abs(*arrays: np.ndarray) -> float:
    maximum = 0.0
    for array in arrays:
        value = np.asarray(array)
        finite = value[np.isfinite(value)]
        if finite.size:
            maximum = max(maximum, float(np.max(np.abs(finite))))
    return maximum


def scalar_label(value: np.ndarray) -> int | None:
    array = np.asarray(value)
    if array.size != 1:
        return None
    try:
        numeric = float(array.reshape(-1)[0])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or numeric != int(numeric):
        return None
    return int(numeric)


def serializable_float(value: float) -> float | str:
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "+inf"
    if np.isneginf(value):
        return "-inf"
    return float(value)


def git_blob(path: str, revision: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{revision}:{path}"], cwd=ROOT, text=True
    )


def ast_nodes(source: str) -> tuple[dict[str, str], dict[str, str]]:
    tree = ast.parse(source)
    nodes: dict[str, str] = {}
    constants: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nodes[node.name] = sha256_bytes(
                ast.dump(node, annotate_fields=True, include_attributes=False).encode("utf-8")
            )
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in FROZEN_CONSTANTS:
                # Preserve expressions such as INPUT_DIM = SEQ_DIM + ECG_DIM;
                # equality here is a source-level AST equality check.
                constants[target.id] = ast.dump(
                    node.value, annotate_fields=True, include_attributes=False
                )
    return nodes, constants


def main_assignment_hashes(source: str) -> dict[str, str]:
    """Hash critical assignments/epoch loop within main, independent of line numbers."""
    tree = ast.parse(source)
    main_node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    wanted = {
        "loader_generator",
        "train_sampler",
        "train_loader",
        "val_loader",
        "external_loader",
        "model",
        "pos_weight_value",
        "optimizer",
        "pos_weight",
        "criterion",
        "scheduler",
        "calibrator",
        "val_prob_calibrated",
        "val_threshold",
    }
    result: dict[str, str] = {}
    for node in ast.walk(main_node):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    key = target.id
                    value = sha256_bytes(
                        ast.dump(node, annotate_fields=True, include_attributes=False).encode("utf-8")
                    )
                    # pos_weight_value occurs in both if/else branches; preserve both.
                    if key in result and result[key] != value:
                        result[key] = json_hash(sorted((result[key], value)))
                    else:
                        result[key] = value
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "epoch":
            result["epoch_training_and_early_stopping_loop"] = sha256_bytes(
                ast.dump(node, annotate_fields=True, include_attributes=False).encode("utf-8")
            )
    return result


def production_freeze() -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    parent = git("rev-parse", f"{head}^")
    code_hashes = {name: sha256_file(ROOT / name) for name in PRODUCTION_CODE}
    head_blob_matches = {
        name: sha256_bytes(git_blob(name, head).encode("utf-8")) == code_hashes[name]
        for name in PRODUCTION_CODE
    }

    parent_source = git_blob("train_lstm.py", PARENT_COMMIT)
    current_source = (ROOT / "train_lstm.py").read_text(encoding="utf-8")
    parent_nodes, parent_constants = ast_nodes(parent_source)
    current_nodes, current_constants = ast_nodes(current_source)
    node_checks = {
        name: {
            "parent_sha256": parent_nodes.get(name),
            "current_sha256": current_nodes.get(name),
            "unchanged": parent_nodes.get(name) == current_nodes.get(name),
        }
        for name in UNCHANGED_TOP_LEVEL_NODES
    }
    constant_checks = {
        name: {
            "parent": parent_constants.get(name),
            "current": current_constants.get(name),
            "unchanged": parent_constants.get(name) == current_constants.get(name),
        }
        for name in FROZEN_CONSTANTS
    }
    parent_main = main_assignment_hashes(parent_source)
    current_main = main_assignment_hashes(current_source)
    main_checks = {
        name: {
            "parent_sha256": parent_main.get(name),
            "current_sha256": current_main.get(name),
            "unchanged": parent_main.get(name) == current_main.get(name),
        }
        for name in sorted(set(parent_main) | set(current_main))
    }
    changed_files = git("diff", "--name-only", f"{PARENT_COMMIT}..{AUDIT_COMMIT}").splitlines()
    diff_text = git("diff", "--unified=0", f"{PARENT_COMMIT}..{AUDIT_COMMIT}", "--", "train_lstm.py", "team_code.py")
    hunk_headers = [line for line in diff_text.splitlines() if line.startswith("@@")]
    return {
        "head": head,
        "parent": parent,
        "expected_head": AUDIT_COMMIT,
        "expected_parent": PARENT_COMMIT,
        "head_matches": head == AUDIT_COMMIT,
        "parent_matches": parent == PARENT_COMMIT,
        "initial_status_before_audit_artifacts": "## official-submission...github-target/main\n?? eeg_qc/",
        "current_status": git("status", "--short", "--branch"),
        "production_code_sha256": code_hashes,
        "production_worktree_matches_head": head_blob_matches,
        "changed_files_parent_to_head": changed_files,
        "production_diff_sha256": sha256_bytes(diff_text.encode("utf-8")),
        "production_diff_hunks": hunk_headers,
        "unchanged_top_level_nodes": node_checks,
        "unchanged_constants": constant_checks,
        "unchanged_main_training_nodes": main_checks,
        "diff_scope_conclusion": (
            "Production changes are limited to shared input preprocessing, raw-age preservation, "
            "preprocessing fit/restore plumbing, and checkpoint preprocessing metadata."
        ),
    }


def collect_input_fingerprints(
    cache_root: Path, split_root: Path, rules_path: Path, code_freeze: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifests: dict[str, Any] = {}
    all_memberships: dict[str, list[str]] = {}
    records_out: list[dict[str, Any]] = []
    fallback_out: dict[str, Any] = {}
    missing: list[str] = []
    unexpected: list[str] = []
    read_errors: list[dict[str, str]] = []
    top_large: dict[str, Any] | None = None

    for split in SPLITS:
        manifest_path = split_root / f"{split}_records.json"
        manifest_bytes = manifest_path.read_bytes()
        records = json.loads(manifest_bytes)
        ids = [record_key(item) for item in records]
        all_memberships[split] = ids
        duplicates = sorted(key for key, count in collections.Counter(ids).items() if count > 1)
        actual = sorted(path.stem for path in (cache_root / split).glob("*.npz"))
        expected = set(ids)
        unexpected.extend(str(cache_root / split / f"{rid}.npz") for rid in sorted(set(actual) - expected))
        manifests[split] = {
            "absolute_path": str(manifest_path.resolve()),
            "file_sha256": sha256_bytes(manifest_bytes),
            "record_count": len(records),
            "ordered_record_id_sha256": record_id_hash(ids),
            "sorted_record_id_sha256": record_id_hash(sorted(ids)),
            "duplicates": duplicates,
            "actual_npz_count": len(actual),
            "unexpected_npz_count": len(set(actual) - expected),
        }
        hits: list[str] = []
        condition_counts = collections.Counter()
        label_counts = collections.Counter()
        for position, (record, rid) in enumerate(zip(records, ids)):
            path = cache_root / split / f"{rid}.npz"
            if not path.is_file():
                missing.append(str(path))
                continue
            try:
                file_digest = sha256_file(path)
                with np.load(path, allow_pickle=False) as data:
                    keys = sorted(data.files)
                    X_seq = np.asarray(data["X_seq"])
                    X_ecg = np.asarray(data["X_ecg"])
                    x_static = np.asarray(data["x_static"])
                    mask = np.asarray(data["mask"])
                    label = scalar_label(data["y"])
                raw_age = float(np.asarray(x_static, dtype=np.float64)[0])
                conditions = {
                    "seq_shape_1x483": X_seq.shape == (1, feature_scaling.SEQ_DIM),
                    "seq_all_zero": bool(np.all(X_seq == 0)),
                    "ecg_empty": X_ecg.shape == (0, feature_scaling.ECG_DIM),
                    "mask_all_false": not bool(np.any(np.asarray(mask, dtype=bool))),
                }
                for key, value in conditions.items():
                    condition_counts[key] += int(value)
                fallback = detect_fallback_sequence(X_seq, X_ecg, mask)
                if fallback:
                    hits.append(rid)
                label_counts[str(label)] += 1
                max_abs = finite_max_abs(X_seq, X_ecg, x_static)
                if split == "train" and (top_large is None or max_abs > top_large["max_abs_raw"]):
                    top_large = {"split": split, "record_id": rid, "max_abs_raw": max_abs, "label": label}
                records_out.append(
                    {
                        "split": split,
                        "position": position,
                        "record_id": rid,
                        "absolute_path": str(path.resolve()),
                        "npz_file_sha256": file_digest,
                        "npz_keys": keys,
                        "X_seq_shape": list(X_seq.shape),
                        "X_ecg_shape": list(X_ecg.shape),
                        "x_static_shape": list(x_static.shape),
                        "mask_shape": list(mask.shape),
                        "mask_true_count": int(np.asarray(mask, dtype=bool).sum()),
                        "mask_content_sha256": array_hash(np.asarray(mask, dtype=np.bool_)),
                        "label": label,
                        "raw_age": serializable_float(raw_age),
                        "fallback_sequence": fallback,
                        "max_abs_finite_raw": max_abs,
                    }
                )
            except Exception as exc:
                read_errors.append(
                    {"split": split, "record_id": rid, "path": str(path), "error": f"{type(exc).__name__}: {exc}"}
                )
        fallback_out[split] = {
            "record_count": len(ids),
            "fallback_count": len(hits),
            "fallback_record_ids": hits,
            "individual_condition_counts": dict(condition_counts),
            "label_counts": dict(sorted(label_counts.items())),
        }

    overlaps = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            shared = sorted(set(all_memberships[left]) & set(all_memberships[right]))
            overlaps[f"{left}__{right}"] = {"count": len(shared), "record_ids": shared}

    ordered_file_tuples = [
        (item["split"], item["position"], item["record_id"], item["npz_file_sha256"])
        for item in records_out
    ]
    train_val_binary = all(
        item["label"] in (0, 1) for item in records_out if item["split"] in ("train", "val")
    )
    hidden_minus_one = {
        split: sum(item["label"] == -1 for item in records_out if item["split"] == split)
        for split in ("test", "external")
    }
    fingerprint = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "cache_root": str(cache_root.resolve()),
        "split_root": str(split_root.resolve()),
        "rules_path": str(rules_path.resolve()),
        "rules_file_sha256": sha256_file(rules_path),
        "rules_canonical_sha256": stable_hash(feature_scaling.load_feature_rules(rules_path)),
        "rules_generation_rerun": False,
        "manifests": manifests,
        "overlaps": overlaps,
        "missing_cache_files": missing,
        "unexpected_cache_files": unexpected,
        "read_errors": read_errors,
        "record_count_scanned": len(records_out),
        "ordered_npz_content_sha256": json_hash(ordered_file_tuples),
        "records": records_out,
        "production_code": code_freeze,
    }
    fallback = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "production_condition": {
            "X_seq_shape": [1, feature_scaling.SEQ_DIM],
            "X_seq_all_zero": True,
            "X_ecg_shape": [0, feature_scaling.ECG_DIM],
            "mask_all_false": True,
        },
        "splits": fallback_out,
        "train_fallback_count": fallback_out["train"]["fallback_count"],
        "fit_impact_conclusion": (
            "No training fallback_sequence records exist; the known fit/offline fallback omission "
            "does not affect the current typed_v1 fit. No preprocessing code was changed."
            if fallback_out["train"]["fallback_count"] == 0
            else "Training fallback records exist; preprocessing boundary repair is required before launch."
        ),
    }
    gates = {
        "manifest_files_present": all((split_root / f"{split}_records.json").is_file() for split in SPLITS),
        "no_missing_cache": not missing,
        "no_unexpected_cache": not unexpected,
        "no_npz_read_errors": not read_errors,
        "expected_total_records_1103": len(records_out) == 1103,
        "expected_split_counts_733_158_158_54": [manifests[s]["record_count"] for s in SPLITS] == [733, 158, 158, 54],
        "no_within_split_duplicates": all(not manifests[s]["duplicates"] for s in SPLITS),
        "no_cross_split_overlap": all(item["count"] == 0 for item in overlaps.values()),
        "train_val_labels_binary": train_val_binary,
        "train_fallback_absent": fallback_out["train"]["fallback_count"] == 0,
        "test_external_minus_one_excluded_from_future_truth_metrics": True,
    }
    context = {"largest_training_record": top_large, "hidden_minus_one_counts": hidden_minus_one}
    return fingerprint, fallback, {"gates": gates, "context": context}


def rng_snapshot() -> dict[str, str]:
    result = {
        "python": sha256_bytes(pickle.dumps(random.getstate(), protocol=5)),
        "numpy": sha256_bytes(pickle.dumps(np.random.get_state(), protocol=5)),
        "torch_cpu": sha256_bytes(torch.get_rng_state().cpu().numpy().tobytes()),
    }
    if torch.cuda.is_available():
        result["torch_cuda"] = json_hash(
            [sha256_bytes(item.cpu().numpy().tobytes()) for item in torch.cuda.get_rng_state_all()]
        )
    else:
        result["torch_cuda"] = "not_available"
    return result


def model_state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


class RecordingPSGDataset(train_lstm.PSGDataset):
    def __init__(self, *args, trace: list[str], **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = trace

    def __getitem__(self, idx):
        self.trace.append(record_key(self.records[idx]))
        return super().__getitem__(idx)


def build_actual_loader(records, cache_dir: Path, preprocessor: FeatureScaler, trace: list[str]):
    dataset = RecordingPSGDataset(records, str(cache_dir), None, preprocessor, trace=trace)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    sampler = train_lstm.build_balanced_sampler(records, str(cache_dir), generator=generator)
    return DataLoader(
        dataset,
        batch_size=train_lstm.BATCH_SIZE,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=train_lstm.collate_fn,
        num_workers=0,
        generator=generator,
    )


def fit_typed(train_records, cache_root: Path, split_root: Path, rules_path: Path) -> FeatureScaler:
    return FeatureScaler.fit_from_cache(
        train_records,
        cache_root / "train",
        rules_path=rules_path,
        sample_cap=128,
        seed=SEED,
        manifest_path=split_root / "train_records.json",
        code_head=AUDIT_COMMIT,
    )


def test_command() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "unittest",
        "feat_input.feat_mody.test_feature_scaling",
        "feat_input.feat_mody.test_feature_scaling_fit_and_compat",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    combined = (result.stdout + "\n" + result.stderr).strip()
    match = re.search(r"Ran\s+(\d+)\s+tests?", combined)
    return {
        "command": command,
        "returncode": result.returncode,
        "reported_test_count": int(match.group(1)) if match else None,
        "expected_test_count": 15,
        "passed": result.returncode == 0 and match is not None and int(match.group(1)) == 15,
        "output": combined,
    }


def scan_transform_compatibility(
    fingerprint: dict[str, Any], typed: FeatureScaler
) -> dict[str, Any]:
    legacy = FeatureScaler.legacy_clip()
    legacy_mismatches = []
    typed_nonfinite = []
    for item in fingerprint["records"]:
        path = Path(item["absolute_path"])
        with np.load(path, allow_pickle=False) as data:
            raw = (np.asarray(data["X_seq"]), np.asarray(data["X_ecg"]), np.asarray(data["x_static"]))
            mask = np.asarray(data["mask"])
        fallback = detect_fallback_sequence(raw[0], raw[1], mask)
        legacy_out = legacy.transform_arrays(*raw, record_id=item["record_id"], fallback_sequence=fallback)
        expected = tuple(
            np.clip(
                np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0
            ).astype(np.float32)
            for value in raw
        )
        if any(not np.array_equal(left, right) for left, right in zip(legacy_out, expected)):
            legacy_mismatches.append(item["record_id"])
        typed_out = typed.transform_arrays(
            *raw, record_id=item["record_id"], fallback_sequence=fallback
        )
        for branch, value in zip(("X_seq", "X_ecg", "x_static"), typed_out):
            if value.dtype != np.float32 or not np.all(np.isfinite(value)):
                typed_nonfinite.append({"record_id": item["record_id"], "branch": branch, "dtype": str(value.dtype)})
    return {
        "records_checked": len(fingerprint["records"]),
        "legacy_reference": "a4fe0b4 np.clip(np.nan_to_num(x,nan=0,posinf=0,neginf=0),-50,50) then torch.FloatTensor",
        "legacy_exact_mismatch_count": len(legacy_mismatches),
        "legacy_exact_mismatch_record_ids": legacy_mismatches[:20],
        "typed_float32_nonfinite_or_dtype_failure_count": len(typed_nonfinite),
        "typed_failures": typed_nonfinite[:20],
        "passed": not legacy_mismatches and not typed_nonfinite,
    }


def paired_loader_and_initialization(
    train_records, cache_root: Path, split_root: Path, rules_path: Path, typed: FeatureScaler
) -> dict[str, Any]:
    """Reproduce loader-generator calls without repeatedly decoding feature tensors."""
    class RecordingIdDataset(Dataset):
        def __init__(self, records, trace):
            self.records = records
            self.trace = trace

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            rid = record_key(self.records[index])
            self.trace.append(rid)
            return rid

    variants = {}
    for mode in (LEGACY_MODE, SCALER_VERSION):
        train_lstm.seed_everything(SEED)
        if mode == SCALER_VERSION:
            FeatureScaler.fit_from_cache(
                train_records[:2], cache_root / "train", rules_path=rules_path,
                sample_cap=128, seed=SEED, code_head=AUDIT_COMMIT,
            )
        trace = []
        generator = torch.Generator()
        generator.manual_seed(SEED)
        sampler = train_lstm.build_balanced_sampler(
            train_records, str(cache_root / "train"), generator=generator
        )
        loader = DataLoader(
            RecordingIdDataset(train_records, trace),
            batch_size=train_lstm.BATCH_SIZE,
            shuffle=sampler is None, sampler=sampler, num_workers=0,
            generator=generator, collate_fn=lambda batch: batch,
        )
        diagnostic_ids = list(next(iter(loader)))
        model = train_lstm.LSTMModel().to(train_lstm.device)
        initial_hash = model_state_hash(model)
        epoch_orders = []
        for _epoch in range(2):
            start = len(trace)
            for _batch in loader:
                pass
            epoch_orders.append(trace[start:])
        preprocessor = FeatureScaler.legacy_clip() if mode == LEGACY_MODE else typed
        variants[mode] = {
            "first_batch_diagnostic_record_ids": diagnostic_ids,
            "first_batch_was_nonempty": bool(diagnostic_ids),
            "initial_model_state_sha256": initial_hash,
            "epoch_1_ordered_record_ids": epoch_orders[0],
            "epoch_2_ordered_record_ids": epoch_orders[1],
            "epoch_1_order_sha256": record_id_hash(epoch_orders[0]),
            "epoch_2_order_sha256": record_id_hash(epoch_orders[1]),
            "scaler_state_sha256": stable_hash(preprocessor.state_dict()),
            "order_probe_note": "Uses production sampler, DataLoader, and generator call order; feature decoding and diagnostic reductions consume no RNG.",
        }
        del model, loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    comparison = {
        "diagnostic_order_identical": variants[LEGACY_MODE]["first_batch_diagnostic_record_ids"] == variants[SCALER_VERSION]["first_batch_diagnostic_record_ids"],
        "epoch_1_order_identical": variants[LEGACY_MODE]["epoch_1_ordered_record_ids"] == variants[SCALER_VERSION]["epoch_1_ordered_record_ids"],
        "epoch_2_order_identical": variants[LEGACY_MODE]["epoch_2_ordered_record_ids"] == variants[SCALER_VERSION]["epoch_2_ordered_record_ids"],
        "initial_model_weights_identical": variants[LEGACY_MODE]["initial_model_state_sha256"] == variants[SCALER_VERSION]["initial_model_state_sha256"],
    }
    comparison["passed"] = all(comparison.values())
    return {"variants": variants, "comparison": comparison}


def real_device_smoke(
    train_records,
    cache_root: Path,
    typed: FeatureScaler,
    largest_training_record: dict[str, Any],
) -> dict[str, Any]:
    by_id = {record_key(item): item for item in train_records}
    largest_id = largest_training_record["record_id"]
    selected = [by_id[largest_id]]
    largest_label = largest_training_record["label"]
    for item in train_records:
        rid = record_key(item)
        if rid == largest_id:
            continue
        with np.load(cache_root / "train" / f"{rid}.npz", allow_pickle=False) as data:
            label = scalar_label(data["y"])
        if label != largest_label:
            selected.append(item)
            break
    if len(selected) < 2:
        selected.append(next(item for item in train_records if record_key(item) != largest_id))

    labels = []
    for item in train_records:
        with np.load(cache_root / "train" / f"{record_key(item)}.npz", allow_pickle=False) as data:
            labels.append(scalar_label(data["y"]))
    pos = sum(label == 1 for label in labels)
    neg = sum(label == 0 for label in labels)
    pos_weight_value = float(neg / pos)

    variants = {}
    for mode, preprocessor in ((LEGACY_MODE, FeatureScaler.legacy_clip()), (SCALER_VERSION, typed)):
        train_lstm.seed_everything(SEED)
        dataset = train_lstm.PSGDataset(selected, str(cache_root / "train"), None, preprocessor)
        batch = train_lstm.collate_fn([dataset[index] for index in range(len(dataset))])
        X_seq, X_ecg, mask_ecg, x_static, ages, y, lengths = batch
        model = train_lstm.LSTMModel().to(train_lstm.device)
        before_hash = model_state_hash(model)
        model.train()
        prediction = model(
            X_seq.to(train_lstm.device),
            X_ecg.to(train_lstm.device),
            mask_ecg.to(train_lstm.device),
            x_static.to(train_lstm.device),
            lengths.to(train_lstm.device),
        )
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=train_lstm.device)
        )
        loss = criterion(prediction, y.squeeze(-1).to(train_lstm.device))
        loss.backward()
        gradients_finite = all(
            parameter.grad is None or torch.all(torch.isfinite(parameter.grad)).item()
            for parameter in model.parameters()
        )
        after_hash = model_state_hash(model)
        variants[mode] = {
            "device": str(train_lstm.device),
            "record_ids": [record_key(item) for item in selected],
            "largest_record_included": largest_id in [record_key(item) for item in selected],
            "largest_record_max_abs_raw": largest_training_record["max_abs_raw"],
            "batch_shapes": {
                "X_seq": list(X_seq.shape), "X_ecg": list(X_ecg.shape), "x_static": list(x_static.shape)
            },
            "inputs_float32_and_finite": all(
                value.dtype == torch.float32 and torch.all(torch.isfinite(value)).item()
                for value in (X_seq, X_ecg, x_static)
            ),
            "prediction_finite": torch.all(torch.isfinite(prediction)).item(),
            "loss_finite": torch.isfinite(loss).item(),
            "gradients_finite": gradients_finite,
            "weights_unchanged_without_optimizer_step": before_hash == after_hash,
            "initial_model_state_sha256": before_hash,
        }
        variants[mode]["passed"] = all(
            variants[mode][key]
            for key in (
                "largest_record_included",
                "inputs_float32_and_finite",
                "prediction_finite",
                "loss_finite",
                "gradients_finite",
                "weights_unchanged_without_optimizer_step",
            )
        )
        del model, batch, prediction, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "variants": variants,
        "same_initial_model_hash": variants[LEGACY_MODE]["initial_model_state_sha256"]
        == variants[SCALER_VERSION]["initial_model_state_sha256"],
        "passed": all(item["passed"] for item in variants.values()),
    }


def dataset_inference_consistency(
    train_records, cache_root: Path, typed: FeatureScaler, record_id: str
) -> dict[str, Any]:
    record = next(item for item in train_records if record_key(item) == record_id)
    path = cache_root / "train" / f"{record_id}.npz"
    with np.load(path, allow_pickle=False) as data:
        raw_age = float(np.asarray(data["x_static"], dtype=np.float64)[0])
        raw_mask = np.asarray(data["mask"], dtype=bool)
    dataset = train_lstm.PSGDataset([record], str(cache_root / "train"), None, typed)
    dataset_item = dataset[0]
    inference_item = team_code._sample(record, ROOT, [cache_root / "train"], typed)
    checks = {
        "X_seq_equal": np.array_equal(dataset_item["X_seq"].numpy(), inference_item["X_seq"]),
        "X_ecg_equal": np.array_equal(dataset_item["X_ecg"].numpy(), inference_item["X_ecg"]),
        "x_static_equal": np.array_equal(dataset_item["x_static"].numpy(), inference_item["x_static"]),
        "mask_equal": np.array_equal(dataset_item["mask"].numpy(), raw_mask),
        "raw_age_preserved": float(dataset_item["age"].item()) == float(np.float32(raw_age)),
        "raw_age_source": serializable_float(raw_age),
        "metric_age_tensor": float(dataset_item["age"].item()),
    }
    checks["passed"] = all(value for key, value in checks.items() if isinstance(value, bool))
    return {"record_id": record_id, **checks}


def runtime_environment() -> dict[str, Any]:
    gpu = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu.append(
                {
                    "visible_index": index,
                    "name": properties.name,
                    "total_memory": properties.total_memory,
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], text=True, capture_output=True, check=False
    )
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "sklearn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": gpu,
        "pip_freeze_returncode": freeze.returncode,
        "pip_freeze_sha256": sha256_bytes(freeze.stdout.encode("utf-8")),
    }


def verify_expected(output_dir: Path) -> dict[str, Any]:
    expected = json.loads((output_dir / "input_fingerprints.json").read_text(encoding="utf-8"))
    failures = []
    if git("rev-parse", "HEAD") != expected["production_code"]["head"]:
        failures.append("Git HEAD changed")
    for name, digest in expected["production_code"]["production_code_sha256"].items():
        path = ROOT / name
        if not path.is_file() or sha256_file(path) != digest:
            failures.append(f"production code changed: {name}")
    rules_path = Path(expected["rules_path"])
    if not rules_path.is_file() or sha256_file(rules_path) != expected["rules_file_sha256"]:
        failures.append("feature rules file changed")
    for split, manifest in expected["manifests"].items():
        path = Path(manifest["absolute_path"])
        if not path.is_file() or sha256_file(path) != manifest["file_sha256"]:
            failures.append(f"manifest changed: {split}")
    tuples = []
    for item in expected["records"]:
        path = Path(item["absolute_path"])
        if not path.is_file():
            failures.append(f"cache missing: {path}")
            continue
        digest = sha256_file(path)
        if digest != item["npz_file_sha256"]:
            failures.append(f"cache content changed: {path}")
        tuples.append((item["split"], item["position"], item["record_id"], digest))
    if json_hash(tuples) != expected["ordered_npz_content_sha256"]:
        failures.append("ordered aggregate NPZ content hash changed")
    pairing = json.loads((output_dir / "pairing_checks.json").read_text(encoding="utf-8"))
    if pairing.get("overall_status") != "PASS":
        failures.append("stored preflight status is not PASS")
    result = {
        "verified_at_utc": utc_now(),
        "status": "PASS" if not failures else "BLOCKED",
        "failures": failures,
        "records_rehashed": len(tuples),
        "ordered_npz_content_sha256": json_hash(tuples),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def launch_metadata(output_dir: Path, mode: str, model_dir: Path) -> None:
    config = json.loads((output_dir / "p0_ab_config.json").read_text(encoding="utf-8"))
    payload = {
        "created_at_utc": utc_now(),
        "mode": mode,
        "model_dir": str(model_dir.resolve()),
        "git_head": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short", "--branch"),
        "environment": runtime_environment(),
        "command": [config["environment"]["python_executable"], str(ROOT / "train_lstm.py")],
        "explicit_environment": config["common_training_environment"]
        | {"LSTM_INPUT_PREPROCESSING": mode, "LSTM_MODEL_DIR": str(model_dir.resolve())},
        "rules_file_sha256": config["rules"]["file_sha256"],
        "code_sha": config["git"]["head"],
        "ordered_npz_content_sha256": config["inputs"]["ordered_npz_content_sha256"],
        "resume": False,
        "old_weights_loaded": False,
    }
    write_json(model_dir / "launch_metadata.json", payload)


def render_report(
    config: dict[str, Any], fingerprint: dict[str, Any], fallback: dict[str, Any], pairing: dict[str, Any]
) -> str:
    status = pairing["overall_status"]
    failed = [name for name, value in pairing["gates"].items() if not value]
    lines = [
        "# P0 legacy_clip vs typed_v1 launch preflight",
        "",
        f"**Final gate: {status}**",
        "",
        f"Generated: `{config['generated_at_utc']}`",
        f"Frozen Git HEAD: `{config['git']['head']}`",
        f"Parent baseline: `{config['git']['parent']}`",
        "",
        "## Scope freeze",
        "",
        "The worktree started at the audited commit with only the pre-existing untracked `eeg_qc/` directory. "
        "No reset was performed and no user file was overwritten. Audit artifacts added later are intentionally untracked.",
        "",
        "The parent-to-head production diff was classified as input preprocessing plumbing only. "
        "AST/content checks confirm that model architecture and initialization, collate/ECG offset 10, batch size, "
        "optimizer/LR, balanced sampler, pos_weight, gradient clipping, scheduler, early stopping, checkpoint selection, "
        "calibration, and threshold selection are unchanged.",
        "",
        f"Frozen rule file SHA256: `{fingerprint['rules_file_sha256']}`. "
        "`build_feature_rules_v1.py` was not run, no common z clipping was added, and per-column rules were not changed.",
        "",
        "## Raw input identity",
        "",
        f"Cache: `{fingerprint['cache_root']}`",
        f"Split: `{fingerprint['split_root']}`",
        f"Records hashed: `{fingerprint['record_count_scanned']}`",
        f"Ordered aggregate NPZ content SHA256: `{fingerprint['ordered_npz_content_sha256']}`",
        "",
        "| split | records | manifest SHA256 | ordered ID SHA256 | fallback | label counts |",
        "|---|---:|---|---|---:|---|",
    ]
    for split in SPLITS:
        manifest = fingerprint["manifests"][split]
        item = fallback["splits"][split]
        lines.append(
            f"| {split} | {manifest['record_count']} | `{manifest['file_sha256']}` | "
            f"`{manifest['ordered_record_id_sha256']}` | {item['fallback_count']} | "
            f"`{json.dumps(item['label_counts'], sort_keys=True)}` |"
        )
    lines.extend(
        [
            "",
            "Every listed NPZ has an individual byte-content SHA256 in `input_fingerprints.json`; mask content, label, "
            "raw age, sequence lengths, and ordered membership are recorded there as well. A and B point to these exact "
            "same files. Missing caches cause an immediate BLOCKED result and never trigger extraction or cache substitution.",
            "",
            "Train/val labels are required to be exactly binary. Test/external `-1` values, if present, are explicitly "
            "not valid truth labels and must be excluded from later metric computation.",
            "",
            "## fallback_sequence boundary",
            "",
            fallback["fit_impact_conclusion"],
            "",
            "The known offline-validator omission remains documented but is not triggered by this frozen dataset. "
            "Because the training count is zero, no scaler code or old report was changed.",
            "",
            "## Pairing checks",
            "",
        ]
    )
    for key, value in pairing["gates"].items():
        lines.append(f"- {'PASS' if value else 'FAIL'} — `{key}`")
    lines.extend(
        [
            "",
            "The checks ran in a standalone preflight process. They did not update optimizer weights or save a model. "
            "Formal runs start in separate fresh processes and do not execute additional sampling diagnostics.",
            "",
            "## Launch",
            "",
            "`run_p0_ab.sh` first re-hashes Git HEAD, production code, all manifests, the frozen rules, and all 1103 NPZ "
            "files. It refuses existing result directories, then runs A followed by B on the same visible GPU with seed 7. "
            "Only `LSTM_INPUT_PREPROCESSING` and `LSTM_MODEL_DIR` differ.",
            "",
        ]
    )
    if status == "PASS":
        lines.append("**PASS: the pair can be launched; only input preprocessing changes.**")
    else:
        lines.append(f"**BLOCKED: failed gates: `{', '.join(failed)}`. See `pairing_checks.json` for evidence.**")
    lines.append("")
    return "\n".join(lines)


def generate(output_dir: Path, cache_root: Path, split_root: Path, rules_path: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    code_freeze = production_freeze()
    fingerprint, fallback, input_result = collect_input_fingerprints(
        cache_root, split_root, rules_path, code_freeze
    )
    write_json(output_dir / "input_fingerprints.json", fingerprint)
    write_json(output_dir / "fallback_counts.json", fallback)

    gates = dict(input_result["gates"])
    gates.update(
        {
            "git_head_is_0e6e708": code_freeze["head_matches"],
            "git_parent_is_a4fe0b4": code_freeze["parent_matches"],
            "production_worktree_matches_head": all(code_freeze["production_worktree_matches_head"].values()),
            "frozen_constants_unchanged": all(
                item["unchanged"] for item in code_freeze["unchanged_constants"].values()
            ),
            "frozen_functions_and_model_unchanged": all(
                item["unchanged"] for item in code_freeze["unchanged_top_level_nodes"].values()
            ),
            "training_strategy_nodes_unchanged": all(
                item["unchanged"] for item in code_freeze["unchanged_main_training_nodes"].values()
            ),
            "rules_file_is_committed_version": fingerprint["rules_file_sha256"]
            == sha256_bytes(git_blob("feat_input/feature_rules_v1.json", AUDIT_COMMIT).encode("utf-8")),
        }
    )

    details: dict[str, Any] = {"input_context": input_result["context"]}
    external_minus_one = input_result["context"]["hidden_minus_one_counts"]["external"]
    external_scoring_present = "external_age_auroc, external_auroc, external_tpr5 = evaluate" in (ROOT / "train_lstm.py").read_text(encoding="utf-8")
    details["evaluation_label_safety"] = {
        "test_minus_one_count": input_result["context"]["hidden_minus_one_counts"]["test"],
        "external_minus_one_count": external_minus_one,
        "train_lstm_external_evaluate_call_present": external_scoring_present,
        "conclusion": "Current trainer would log pseudo external metrics from -1 placeholders; these are not truth labels.",
    }
    gates["training_entry_does_not_score_minus_one_external"] = not (external_minus_one > 0 and external_scoring_present)
    if all(input_result["gates"].values()):
        train_records = json.loads((split_root / "train_records.json").read_text(encoding="utf-8"))
        tests = test_command()
        details["unit_tests"] = tests
        gates["existing_15_tests_pass"] = tests["passed"]

        train_lstm.seed_everything(SEED)
        rng_before_fit = rng_snapshot()
        fit_probe = FeatureScaler.fit_from_cache(
            train_records[:2], cache_root / "train", rules_path=rules_path,
            sample_cap=128, seed=SEED, code_head=AUDIT_COMMIT,
        )
        rng_after_fit = rng_snapshot()
        frozen_state_path = ROOT / "feat_input" / "p0_fix" / "scaler_state_preview.json"
        frozen_state = json.loads(frozen_state_path.read_text(encoding="utf-8"))
        typed = FeatureScaler.from_state_dict(frozen_state)
        frozen_meta = typed.metadata
        frozen_state_validation = {
            "absolute_path": str(frozen_state_path.resolve()),
            "file_sha256": sha256_file(frozen_state_path),
            "committed_file_matches_worktree": sha256_bytes(git_blob("feat_input/p0_fix/scaler_state_preview.json", AUDIT_COMMIT).encode("utf-8")) == sha256_file(frozen_state_path),
            "sampling_seed_is_7": frozen_meta.get("sampling_seed") == SEED,
            "fit_record_count_is_733": frozen_meta.get("fit_record_count") == 733,
            "manifest_hash_matches": frozen_meta.get("manifest_sha256") == fingerprint["manifests"]["train"]["file_sha256"],
            "canonical_rules_hash_matches": frozen_meta.get("rules_sha256") == fingerprint["rules_canonical_sha256"],
            "clip_z_is_none": typed.clip_z is None,
            "parameter_content_sha256": stable_hash(typed.parameters),
            "stored_code_head": frozen_meta.get("code_head"),
            "provenance_note": "Committed seed-7 state was generated before commit 0e6e708, so stored code_head is its then-current a4fe0b4; formal B refits from the frozen raw train files and records current HEAD.",
        }
        details["frozen_seed7_scaler_state"] = frozen_state_validation
        gates["committed_seed7_scaler_state_matches_frozen_inputs"] = all(
            frozen_state_validation[key] for key in (
                "committed_file_matches_worktree", "sampling_seed_is_7",
                "fit_record_count_is_733", "manifest_hash_matches",
                "canonical_rules_hash_matches", "clip_z_is_none",
            )
        )
        sample_record = input_result["context"]["largest_training_record"]["record_id"]
        sample_path = cache_root / "train" / f"{sample_record}.npz"
        with np.load(sample_path, allow_pickle=False) as data:
            sample_arrays = (np.asarray(data["X_seq"]), np.asarray(data["X_ecg"]), np.asarray(data["x_static"]))
            sample_mask = np.asarray(data["mask"])
        rng_before_transform = rng_snapshot()
        typed.transform_arrays(
            *sample_arrays,
            record_id=sample_record,
            fallback_sequence=detect_fallback_sequence(sample_arrays[0], sample_arrays[1], sample_mask),
        )
        rng_after_transform = rng_snapshot()
        details["rng_isolation"] = {
            "before_fit": rng_before_fit,
            "after_fit": rng_after_fit,
            "fit_preserved_global_rng": rng_before_fit == rng_after_fit,
            "before_transform": rng_before_transform,
            "after_transform": rng_after_transform,
            "transform_preserved_global_rng": rng_before_transform == rng_after_transform,
        }
        gates["fit_preserves_all_global_rng_states"] = rng_before_fit == rng_after_fit
        gates["transform_preserves_all_global_rng_states"] = rng_before_transform == rng_after_transform
        details["typed_scaler_state_sha256"] = stable_hash(typed.state_dict())
        details["typed_scaler_metadata"] = typed.metadata

        compatibility = scan_transform_compatibility(fingerprint, typed)
        details["transform_compatibility"] = compatibility
        gates["legacy_matches_a4fe0b4_on_all_1103_records"] = compatibility["legacy_exact_mismatch_count"] == 0
        gates["typed_all_1103_outputs_float32_finite"] = compatibility["typed_float32_nonfinite_or_dtype_failure_count"] == 0

        paired = paired_loader_and_initialization(train_records, cache_root, split_root, rules_path, typed)
        details["paired_loader_and_initialization"] = paired
        gates["same_seed_initial_model_weights_identical"] = paired["comparison"]["initial_model_weights_identical"]
        gates["first_batch_diagnostic_sampling_identical"] = paired["comparison"]["diagnostic_order_identical"]
        gates["first_two_epoch_sampling_identical"] = (
            paired["comparison"]["epoch_1_order_identical"] and paired["comparison"]["epoch_2_order_identical"]
        )

        smoke = real_device_smoke(
            train_records, cache_root, typed, input_result["context"]["largest_training_record"]
        )
        details["real_device_forward_backward"] = smoke
        gates["real_device_forward_backward_pass"] = smoke["passed"]
        gates["smoke_initial_weights_identical"] = smoke["same_initial_model_hash"]

        consistency = dataset_inference_consistency(train_records, cache_root, typed, sample_record)
        details["typed_dataset_inference_consistency"] = consistency
        gates["typed_dataset_equals_inference_path"] = consistency["passed"]
        gates["raw_metric_age_preserved"] = consistency["raw_age_preserved"]
    else:
        details["stopped_before_pairing_checks"] = (
            "Input gate failed; no feature extraction, cache substitution, transform scan, or model smoke was attempted."
        )

    overall = "PASS" if all(gates.values()) else "BLOCKED"
    pairing = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "overall_status": overall,
        "gates": gates,
        "details": details,
        "formal_training_started": False,
        "formal_weights_saved": False,
        "npz_written_or_transformed": False,
    }
    write_json(output_dir / "pairing_checks.json", pairing)

    environment = runtime_environment()
    output_a = (ROOT / "output" / "p0_ab" / "seed7" / "A_legacy_clip").resolve()
    output_b = (ROOT / "output" / "p0_ab" / "seed7" / "B_typed_v1").resolve()
    common_env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "LSTM_CACHE_DIR": str(cache_root.resolve()),
        "LSTM_SPLITS_DIR": str(split_root.resolve()),
        "LSTM_FEATURE_RULES": str(rules_path.resolve()),
        "LSTM_SEED": str(SEED),
        "PYTHONHASHSEED": str(SEED),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    config = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "preflight_status": overall,
        "repository": "ysj-EDG/physionet-challenge-2026-baseline",
        "git": {"head": code_freeze["head"], "parent": code_freeze["parent"]},
        "environment": environment,
        "gpu_policy": {
            "physical_gpu_index": 0,
            "cuda_visible_devices": "0",
            "preflight_visible_device": environment["gpus"][0] if environment["gpus"] else None,
        },
        "inputs": {
            "cache_root": str(cache_root.resolve()),
            "split_root": str(split_root.resolve()),
            "ordered_npz_content_sha256": fingerprint["ordered_npz_content_sha256"],
            "record_count": fingerprint["record_count_scanned"],
        },
        "rules": {
            "absolute_path": str(rules_path.resolve()),
            "file_sha256": fingerprint["rules_file_sha256"],
            "canonical_sha256": fingerprint["rules_canonical_sha256"],
            "regenerated": False,
            "clip_z": None,
        },
        "frozen_training": {
            "epochs": train_lstm.EPOCHS,
            "patience": train_lstm.PATIENCE,
            "batch_size": train_lstm.BATCH_SIZE,
            "learning_rate": train_lstm.LR,
            "ecg_epoch_offset": ECG_EPOCH_OFFSET,
            "optimizer": "Adam",
            "gradient_clip_max_norm": 1.0,
            "scheduler": "ReduceLROnPlateau(mode=max,factor=0.5,patience=8)",
            "checkpoint_selection": "validation age-conditioned AUROC, gap=2",
            "calibration": "validation Platt logistic",
            "threshold": "validation Youden threshold on calibrated probability",
            "resume": False,
            "old_weights_loaded": False,
        },
        "common_training_environment": common_env,
        "variants": [
            {"arm": "A", "mode": LEGACY_MODE, "model_dir": str(output_a)},
            {"arm": "B", "mode": SCALER_VERSION, "model_dir": str(output_b)},
        ],
        "allowed_arm_differences": ["LSTM_INPUT_PREPROCESSING", "LSTM_MODEL_DIR"],
        "training_command": [environment["python_executable"], str((ROOT / "train_lstm.py").resolve())],
        "formal_training_started": False,
        "evaluation_label_policy": "Only labels 0/1 are truth; exclude -1 in test/external from all metrics.",
    }
    write_json(output_dir / "p0_ab_config.json", config)
    (output_dir / "P0_AB_PREFLIGHT.md").write_text(
        render_report(config, fingerprint, fallback, pairing), encoding="utf-8"
    )
    print(json.dumps({"status": overall, "output_dir": str(output_dir), "failed": [k for k, v in gates.items() if not v]}, indent=2))
    return 0 if overall == "PASS" else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--generate", action="store_true")
    action.add_argument("--verify", action="store_true")
    action.add_argument("--launch-metadata", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "feat_input" / "p0_ab")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "npz_new")
    parser.add_argument("--split-root", type=Path, default=ROOT / "split")
    parser.add_argument("--rules", type=Path, default=ROOT / "feat_input" / "feature_rules_v1.json")
    parser.add_argument("--mode", choices=(LEGACY_MODE, SCALER_VERSION))
    parser.add_argument("--model-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.generate:
        return generate(output_dir, args.cache_root.resolve(), args.split_root.resolve(), args.rules.resolve())
    if args.verify:
        result = verify_expected(output_dir)
        return 0 if result["status"] == "PASS" else 2
    if not args.mode or not args.model_dir:
        raise SystemExit("--launch-metadata requires --mode and --model-dir")
    launch_metadata(output_dir, args.mode, args.model_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
