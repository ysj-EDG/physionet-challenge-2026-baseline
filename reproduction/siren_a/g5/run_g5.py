"""Execute the fixed 225-model Branch A rank ensemble and official LOSO scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import numpy as np

from reproduction.siren_a.pipeline.label_join import load_metadata_index

from .core import (
    HOLDOUTS,
    HOLDOUT_TO_TRAIN,
    PROTOCOLS,
    aggregate_protocol,
    load_eval_targets,
    load_predictions,
    mean_ranks,
    score_site,
    sha256_file,
)


REPO = Path("/database/home/gaohaojie/workspace/python-example-2026-siren-a")
G4B_ROOT = Path("/database/home/gaohaojie/workspace/python-example-2026-siren-a-g4b-private/seeds43_56_v1")
G4A_ROOT = Path("/database/home/gaohaojie/workspace/python-example-2026-siren-a-g4a-private/seed42_smoke_v1")
G2_ROOT = Path("/database/home/gaohaojie/workspace/python-example-2026-siren-a-g2b1-private/night_cache_v1")
DEMOGRAPHICS = Path("/database2/physionet2026_kaggle/data2/demographics.csv")
PRIVATE_ROOT = Path("/database/home/gaohaojie/workspace/python-example-2026-siren-a-g5-private/branch_a_rankmean_v1")
PUBLIC_CSV = REPO / "reproduction/siren_a/audit/g5_loso_scores.csv"
PUBLIC_REPORT = REPO / "reproduction/siren_a/docs/G5_LOSO_REPORT.md"
EXPECTED_DEMOGRAPHICS_SHA = "b9009598589846e84462636bfcaa290ed1d8666c89b1bc1ad1a9eda26b43ff7f"
EXPECTED_COUNTS = {"I0002": 319, "I0006": 1142, "S0001": 5139}
SCHEMA = "g5_branch_a_rankmean_v1"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=True).encode("utf-8")


def _identity_sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise ValueError(f"existing output differs: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                     suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_bytes(path, _canonical(value) + b"\n")


def _same_array(left: np.ndarray, right: np.ndarray) -> bool:
    return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(
        left, right, equal_nan=True)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"invalid existing NPZ: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays) or any(
                    not _same_array(np.asarray(archive[key]), np.asarray(value))
                    for key, value in arrays.items()):
                raise ValueError(f"existing NPZ differs: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                     suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez(temporary, **arrays)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        with np.load(temporary, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays):
                raise ValueError("temporary NPZ key mismatch")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_identity_sites(g2_root: Path) -> dict[str, np.ndarray]:
    inventory = json.loads((g2_root / "inventory.json").read_text(encoding="utf-8"))
    private = json.loads((g2_root / "private_index.json").read_text(encoding="utf-8"))
    rows = inventory.get("rows") if isinstance(inventory, dict) else inventory
    if not isinstance(rows, list) or not isinstance(private, list) or len(rows) != 6600 or len(private) != 6600:
        raise ValueError("G2 inventory/private index identity mismatch")
    identity_fields = ("cache_name", "relative_path", "site")
    if any(tuple(left.get(key) for key in identity_fields)
           != tuple(right.get(key) for key in identity_fields)
           for left, right in zip(rows, private)):
        raise ValueError("G2 inventory/private index identity mismatch")
    sites = []
    for position, row in enumerate(rows):
        if row.get("cache_name") != f"cache_{position:06d}.npz":
            raise ValueError("anonymous cache order mismatch")
        site = str(row.get("site"))
        relative = Path(str(row.get("relative_path")))
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2 or relative.parts[0] != site:
            raise ValueError("unsafe or mismatched G2 identity path")
        sites.append(site)
    result = {site: np.flatnonzero(np.asarray(sites) == site).astype(np.int64) for site in EXPECTED_COUNTS}
    if {site: len(indices) for site, indices in result.items()} != EXPECTED_COUNTS:
        raise ValueError("G2 site count mismatch")
    return result


def _load_official(path: Path):
    spec = importlib.util.spec_from_file_location("siren_g5_official_evaluate_model", path)
    if spec is None or spec.loader is None:
        raise ValueError("official scoring module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("compute_auroc_age", "compute_auroc", "compute_auprc"):
        if not callable(getattr(module, name, None)):
            raise ValueError(f"official scoring function missing: {name}")
    return module


def _verify_g5_ready(g5_ready: Path) -> None:
    complete = json.loads((g5_ready / "complete.json").read_text(encoding="utf-8"))
    if complete.get("schema") != "g4b_g5_manifest_v1":
        raise ValueError("G5-ready completion schema mismatch")
    for name in ("predictions_manifest.json", "models_manifest.json", "coverage_audit.json"):
        if sha256_file(g5_ready / name) != complete.get("files", {}).get(name):
            raise ValueError(f"G5-ready completion hash mismatch: {name}")


def _code_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {name: sha256_file(directory / name) for name in ("core.py", "run_g5.py")}


def _run_config(g5_ready: Path) -> dict[str, object]:
    official = REPO / "evaluate_model.py"
    if sha256_file(DEMOGRAPHICS) != EXPECTED_DEMOGRAPHICS_SHA:
        raise ValueError("demographics SHA256 drift")
    return {
        "schema": SCHEMA,
        "algorithm": "per_model_average_tie_percentile_then_equal_mean",
        "rank_formula": "(L + 0.5*E) / N",
        "models_per_group": 225,
        "model_weight": "1/225",
        "official_gap": 2,
        "predictions_manifest_sha256": sha256_file(g5_ready / "predictions_manifest.json"),
        "models_manifest_sha256": sha256_file(g5_ready / "models_manifest.json"),
        "g5_ready_complete_sha256": sha256_file(g5_ready / "complete.json"),
        "demographics_sha256": sha256_file(DEMOGRAPHICS),
        "official_scoring_path": str(official.resolve()),
        "official_scoring_sha256": sha256_file(official),
        "code_sha256": _code_hashes(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def _write_csv(rows: list[dict[str, object]]) -> None:
    fields = ["holdout_site", "train_site", "protocol", "ensemble_method", "n_records",
              "n_positive", "n_negative", "ac_pair_count", "n_models", "ac_auroc",
              "auroc", "auprc", "status"]
    PUBLIC_CSV.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=PUBLIC_CSV.parent,
                                     prefix=PUBLIC_CSV.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in rows])
        handle.flush()
        os.fsync(handle.fileno())
    try:
        _atomic_bytes(PUBLIC_CSV, temporary.read_bytes())
    finally:
        temporary.unlink(missing_ok=True)


def _fmt(value: float) -> str:
    return "NaN" if not np.isfinite(value) else f"{value:.6f}"


def _write_report(rows: list[dict[str, object]], summaries: dict[str, dict[str, float]]) -> None:
    PUBLIC_REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# G5 Branch A Rank Ensemble and LOSO Scoring Report", "",
        "`G5_STATUS = SUBMITTED_FOR_REVIEW`", "",
        "All results use the fixed equal-weight rank mean over 5 views × 3 model families × 15 seeds (225 models). No model, seed, or weight was selected from holdout performance.", "",
        "| Holdout | Train site | Protocol | N | Positive | Negative | AC pairs | Models | AC-AUROC | AUROC | AUPRC |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| {holdout_site} | {train_site} | {protocol} | {n_records} | {n_positive} | {n_negative} | {ac_pair_count} | {n_models} | {ac} | {auc} | {ap} |".format(
            **row, ac=_fmt(float(row["ac_auroc"])), auc=_fmt(float(row["auroc"])),
            ap=_fmt(float(row["auprc"]))))
    lines += ["", "| Protocol | I0002 AC | I0006 AC | S0001 AC | Macro AC | Worst-site AC | Small-site mean AC |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for protocol in PROTOCOLS:
        site_rows = {row["holdout_site"]: row for row in rows if row["protocol"] == protocol}
        summary = summaries[protocol]
        lines.append(f"| {protocol} | {_fmt(site_rows['I0002']['ac_auroc'])} | {_fmt(site_rows['I0006']['ac_auroc'])} | {_fmt(site_rows['S0001']['ac_auroc'])} | {_fmt(summary['macro_ac'])} | {_fmt(summary['worst_site_ac'])} | {_fmt(summary['small_site_mean_ac'])} |")
    lines += [
        "", "## Interpretation and scope", "",
        "This is the first fixed SIREN-inspired Branch A baseline score and does not include Branch B. Each source score vector was ranked over its complete holdout cohort with average ties, then the 225 rank vectors were averaged with weight 1/225. Raw Rank256 inputs, models, and predictions were not recomputed.", "",
        "The inductive-feature comparison means that its feature reference distribution came from the training site. Its final rank ensemble remains cohort-based and is not an end-to-end strictly inductive deployment protocol.", "",
        "No holdout result was used for model selection, seed selection, sign reversal, calibration, thresholding, or weight optimization. Repeated future tuning against these LOSO results would introduce model-selection bias and must be disclosed separately.", "",
        "`G5_PREDICTION_INPUTS = VERIFIED`", "", "`G5_FIXED_RANK_ENSEMBLE = COMPLETE`", "",
        "`G5_OFFICIAL_LOSO_SCORES = COMPLETE`", "", "`G4_MODELS_REFIT = 0`", "",
        "`G4_MODEL_REPLAY = 0`", "", "`G1_G2_G3_G4_ARTIFACTS = UNCHANGED`", "",
        "`HYPERPARAMETER_SEARCH = NOT_STARTED`", "", "`ADDITIONAL_ABLATIONS = NOT_STARTED`", "",
    ]
    _atomic_bytes(PUBLIC_REPORT, ("\n".join(lines)).encode("utf-8"))


def execute() -> dict[str, object]:
    started = time.perf_counter()
    g5_ready = G4B_ROOT / "g5_ready"
    _verify_g5_ready(g5_ready)
    config = _run_config(g5_ready)
    config_sha = _identity_sha(config)
    PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
    _atomic_json(PRIVATE_ROOT / "run_config.json", {**config, "identity_sha256": config_sha})
    site_indices = _load_identity_sites(G2_ROOT)
    load_started = time.perf_counter()
    groups, input_summary = load_predictions(
        g5_ready / "predictions_manifest.json", g5_ready / "models_manifest.json",
        site_indices, (G4A_ROOT, G4B_ROOT))
    prediction_seconds = time.perf_counter() - load_started
    if len(groups) != 6 or any(group.scores.shape[1] != 225 for group in groups.values()):
        raise ValueError("six 225-model groups were not loaded")
    _atomic_json(PRIVATE_ROOT / "input_summary.json", input_summary)

    rank_seconds = 0.0
    for holdout in HOLDOUTS:
        for protocol in PROTOCOLS:
            group = groups[(holdout, protocol)]
            rank_started = time.perf_counter()
            ranks, ensemble = mean_ranks(group.scores)
            rank_seconds += time.perf_counter() - rank_started
            _atomic_npz(PRIVATE_ROOT / "groups" / f"{holdout}_{protocol}" / "fusion.npz",
                        model_keys=np.asarray(group.model_keys, dtype="U32"),
                        test_global_indices=group.test_indices.astype(np.int64, copy=False),
                        model_ranks=ranks, ensemble_scores=ensemble)

    join_started = time.perf_counter()
    metadata = load_metadata_index(DEMOGRAPHICS, G2_ROOT, expected_counts=EXPECTED_COUNTS)
    label_seconds = time.perf_counter() - join_started
    if metadata.demographics_sha256 != EXPECTED_DEMOGRAPHICS_SHA or metadata.audit.get("matched") != 6600:
        raise ValueError("evaluation identity join mismatch")
    official = _load_official(REPO / "evaluate_model.py")
    rows = []
    score_seconds = 0.0
    for holdout in HOLDOUTS:
        shared_targets = None
        for protocol in PROTOCOLS:
            group_path = PRIVATE_ROOT / "groups" / f"{holdout}_{protocol}"
            with np.load(group_path / "fusion.npz", allow_pickle=False) as archive:
                model_keys = np.asarray(archive["model_keys"])
                indices = np.asarray(archive["test_global_indices"])
                ranks = np.asarray(archive["model_ranks"])
                ensemble = np.asarray(archive["ensemble_scores"])
            if not np.all(metadata.sites[indices] == holdout):
                raise ValueError("evaluation indices cross site boundaries")
            labels, ages = load_eval_targets(metadata, indices)
            if shared_targets is not None and (not np.array_equal(labels, shared_targets[0]) or not np.array_equal(ages, shared_targets[1])):
                raise ValueError("protocol evaluation targets differ")
            shared_targets = labels.copy(), ages.copy()
            score_started = time.perf_counter()
            metrics = score_site(labels, ensemble, ages, official)
            score_seconds += time.perf_counter() - score_started
            row = {"holdout_site": holdout, "train_site": HOLDOUT_TO_TRAIN[holdout],
                   "protocol": protocol, "ensemble_method": "equal_mean_of_225_cohort_average_tie_ranks",
                   "n_models": 225, **metrics}
            rows.append(row)
            _atomic_npz(group_path / "result.npz", model_keys=model_keys,
                        test_global_indices=indices, model_ranks=ranks,
                        ensemble_scores=ensemble, labels=labels, ages=ages)
            _atomic_json(group_path / "metrics.json", row)
            print(json.dumps({key: row[key] for key in ("holdout_site", "protocol", "n_records",
                                                        "ac_pair_count", "ac_auroc", "auroc", "auprc")}))

    summaries = {protocol: aggregate_protocol([row for row in rows if row["protocol"] == protocol])
                 for protocol in PROTOCOLS}
    scores_payload = {"schema": SCHEMA, "groups": rows, "protocol_summaries": summaries,
                      "timing_seconds": {"prediction_load": prediction_seconds,
                                         "rank_ensemble": rank_seconds,
                                         "label_join": label_seconds,
                                         "official_scoring": score_seconds,
                                         "total": time.perf_counter() - started}}
    _atomic_json(PRIVATE_ROOT / "scores.json", scores_payload)
    _write_csv(rows)
    _write_report(rows, summaries)
    files = {}
    for path in sorted(PRIVATE_ROOT.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlink forbidden in G5 output")
        if path.is_file() and path.name != "complete.json":
            files[path.relative_to(PRIVATE_ROOT).as_posix()] = sha256_file(path)
    _atomic_json(PRIVATE_ROOT / "complete.json", {
        "schema": SCHEMA, "identity_sha256": config_sha, "files": files,
        "groups": 6, "models_per_group": 225, "status": "PASS"})
    return scores_payload


def verify_only() -> dict[str, object]:
    complete = json.loads((PRIVATE_ROOT / "complete.json").read_text(encoding="utf-8"))
    config = json.loads((PRIVATE_ROOT / "run_config.json").read_text(encoding="utf-8"))
    expected_config = _run_config(G4B_ROOT / "g5_ready")
    if complete.get("schema") != SCHEMA or config.get("identity_sha256") != _identity_sha(expected_config):
        raise ValueError("G5 completion identity mismatch")
    for relative, expected in complete.get("files", {}).items():
        path = PRIVATE_ROOT / relative
        if path.is_symlink() or sha256_file(path) != expected:
            raise ValueError(f"G5 completed file mismatch: {relative}")
    scores = json.loads((PRIVATE_ROOT / "scores.json").read_text(encoding="utf-8"))
    if len(scores.get("groups", [])) != 6:
        raise ValueError("G5 score group count mismatch")
    _write_csv(scores["groups"])
    _write_report(scores["groups"], scores["protocol_summaries"])
    return scores


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fixed Branch A G5 LOSO scoring")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    payload = execute() if args.build else verify_only()
    print(json.dumps({"status": "PASS", "groups": len(payload["groups"]),
                      "protocol_summaries": payload["protocol_summaries"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
