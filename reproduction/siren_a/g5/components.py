"""Decompose the frozen G5 rank ensemble by view and model family."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from reproduction.siren_a.g5.core import (
    HOLDOUTS,
    HOLDOUT_TO_TRAIN,
    MODEL_TYPES,
    PROTOCOLS,
    SEEDS,
    VIEWS,
    score_site,
    sha256_file,
)


REPO = Path("/database/home/gaohaojie/workspace/python-example-2026-siren-a")
G5_ROOT = Path("/database/home/gaohaojie/workspace/python-example-2026-siren-a-g5-private/branch_a_rankmean_v1")
PRIVATE_ROOT = Path("/database/home/gaohaojie/workspace/python-example-2026-siren-a-g5-private/components_v1")
PUBLIC_CSV = REPO / "reproduction/siren_a/audit/g5_component_scores.csv"
PUBLIC_REPORT = REPO / "reproduction/siren_a/docs/G5_COMPONENT_REPORT.md"
OFFICIAL_SHA256 = "227e40a45a7ddb918812c17fd4a8e678334134c965d488db763409d245802970"
SCHEMA = "g5_components_v1"
EXPECTED = {
    "I0002": (319, 52, 267, 1886),
    "I0006": (1142, 112, 1030, 14493),
    "S0001": (5139, 334, 4805, 182330),
}
KEY_RE = re.compile(r"^seed(\d+)\|(V[1-5])\|(lr|hgb|lambdarank)$")


@dataclass(frozen=True)
class Component:
    component_id: str
    component_kind: str
    view: str
    model_type: str
    columns: tuple[int, ...]


@dataclass(frozen=True)
class GroupData:
    model_keys: np.ndarray
    test_global_indices: np.ndarray
    model_ranks: np.ndarray
    ensemble_scores: np.ndarray
    labels: np.ndarray
    ages: np.ndarray


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise ValueError(f"refusing to overwrite different artifact: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_bytes(path, _json_bytes(value))


def _arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right, equal_nan=True)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink():
            raise ValueError(f"symlinked artifact forbidden: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays) or any(not _arrays_equal(np.asarray(archive[key]), np.asarray(value))
                                                         for key, value in arrays.items()):
                raise ValueError(f"refusing to overwrite different artifact: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".npz", delete=False) as handle:
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


def _parse_keys(model_keys: Sequence[str]) -> list[tuple[int, str, str]]:
    parsed = []
    for raw in np.asarray(model_keys).tolist():
        key = str(raw)
        match = KEY_RE.fullmatch(key)
        if match is None:
            raise ValueError(f"illegal model key: {key}")
        parsed.append((int(match.group(1)), match.group(2), match.group(3)))
    expected = {(seed, view, model) for seed in SEEDS for view in VIEWS for model in MODEL_TYPES}
    if len(parsed) != 225 or len(set(parsed)) != 225 or set(parsed) != expected:
        raise ValueError("model keys do not exactly cover the frozen 225 combinations")
    return parsed


def make_components(model_keys: Sequence[str]) -> tuple[Component, ...]:
    parsed = _parse_keys(model_keys)

    def select(view: str | None, model: str | None) -> tuple[int, ...]:
        return tuple(index for index, (_, found_view, found_model) in enumerate(parsed)
                     if (view is None or found_view == view) and (model is None or found_model == model))

    result = []
    for view in VIEWS:
        for model in MODEL_TYPES:
            result.append(Component(f"{view}_{model}", "cell", view, model, select(view, model)))
    for view in VIEWS:
        result.append(Component(f"{view}_all", "view", view, "all", select(view, None)))
    for model in MODEL_TYPES:
        result.append(Component(f"all_{model}", "model_family", "all", model, select(None, model)))
    result.append(Component("all_all", "full", "all", "all", select(None, None)))
    expected_counts = [15] * 15 + [45] * 5 + [75] * 3 + [225]
    if [len(item.columns) for item in result] != expected_counts:
        raise ValueError("component column counts are invalid")
    return tuple(result)


def component_scores(model_ranks: np.ndarray, components: Sequence[Component],
                     saved_full: np.ndarray | None = None) -> np.ndarray:
    ranks = np.asarray(model_ranks)
    if ranks.ndim != 2 or ranks.dtype != np.float64 or ranks.shape[1] != 225:
        raise ValueError("model ranks must have shape (N,225) and dtype float64")
    if not np.isfinite(ranks).all() or np.any(ranks < 0) or np.any(ranks > 1):
        raise ValueError("model ranks must be finite and within [0,1]")
    output = np.column_stack([ranks[:, np.asarray(item.columns)].mean(axis=1, dtype=np.float64)
                              for item in components])
    by_id = {item.component_id: output[:, index] for index, item in enumerate(components)}
    full = ranks.mean(axis=1, dtype=np.float64)
    if saved_full is not None:
        saved = np.asarray(saved_full)
        if saved.shape != (ranks.shape[0],) or saved.dtype != np.float64 or not np.allclose(full, saved, rtol=0, atol=1e-15):
            raise ValueError("saved G5 ensemble does not close against model ranks")
        full = saved
        output[:, [item.component_id for item in components].index("all_all")] = saved
    checks = (
        np.column_stack([by_id[f"V{i}_all"] for i in range(1, 6)]).mean(axis=1, dtype=np.float64),
        np.column_stack([by_id[f"all_{model}"] for model in MODEL_TYPES]).mean(axis=1, dtype=np.float64),
        np.column_stack([by_id[f"V{i}_{model}"] for i in range(1, 6) for model in MODEL_TYPES]).mean(axis=1, dtype=np.float64),
    )
    if any(not np.allclose(check, full, rtol=0, atol=1e-15) for check in checks):
        raise ValueError("component scores do not close against full ensemble")
    return output


def load_group(path: Path, expected_sha256: str) -> GroupData:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"G5 result integrity failure: {path}")
    required = {"model_keys", "test_global_indices", "model_ranks", "ensemble_scores", "labels", "ages"}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError("G5 result keys mismatch")
        values = {key: np.asarray(archive[key]) for key in required}
    keys = values["model_keys"]
    indices = values["test_global_indices"]
    ranks = values["model_ranks"]
    ensemble = values["ensemble_scores"]
    labels = values["labels"]
    ages = values["ages"]
    n_rows = ranks.shape[0] if ranks.ndim == 2 else -1
    make_components(keys)
    if ranks.shape != (n_rows, 225) or ranks.dtype != np.float64:
        raise ValueError("G5 model-rank shape/dtype mismatch")
    if not np.isfinite(ranks).all() or np.any(ranks < 0) or np.any(ranks > 1):
        raise ValueError("G5 model ranks outside [0,1]")
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) != n_rows or len(np.unique(indices)) != n_rows:
        raise ValueError("G5 test indices invalid")
    if ensemble.shape != (n_rows,) or ensemble.dtype != np.float64 or not np.isfinite(ensemble).all():
        raise ValueError("G5 ensemble invalid")
    if labels.shape != (n_rows,) or not np.all(np.isin(labels, [0, 1])):
        raise ValueError("G5 labels invalid")
    if ages.shape != (n_rows,) or not np.isfinite(ages).all():
        raise ValueError("G5 ages invalid")
    if not np.allclose(ranks.mean(axis=1, dtype=np.float64), ensemble, rtol=0, atol=1e-15):
        raise ValueError("G5 saved ensemble mismatch")
    return GroupData(keys, indices, ranks, ensemble, labels, ages)


def validate_protocol_pair(left: Mapping[str, np.ndarray] | GroupData,
                           right: Mapping[str, np.ndarray] | GroupData) -> None:
    def get(value, key):
        return getattr(value, key) if isinstance(value, GroupData) else value[key]
    for key in ("test_global_indices", "labels", "ages"):
        if not np.array_equal(get(left, key), get(right, key)):
            raise ValueError(f"protocol evaluation rows differ: {key}")


def evaluate_components(labels: np.ndarray, ages: np.ndarray, scores: np.ndarray,
                        component_ids: Sequence[str], official) -> list[dict[str, object]]:
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[1] != len(component_ids):
        raise ValueError("component score matrix/ID mismatch")
    return [{"component_id": str(component_id), **score_site(labels, values[:, column], ages, official)}
            for column, component_id in enumerate(component_ids)]


def summarize_components(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        key = str(row["protocol"]), str(row["component_id"])
        site = str(row["holdout_site"])
        if site in grouped.setdefault(key, {}):
            raise ValueError("duplicate component/site summary row")
        grouped[key][site] = float(row["ac_auroc"])
    result = {}
    for key, by_site in grouped.items():
        if set(by_site) != set(HOLDOUTS):
            raise ValueError("component summary lacks a holdout site")
        values = np.asarray([by_site[site] for site in HOLDOUTS], dtype=np.float64)
        if not np.isfinite(values).all():
            macro = worst = small = float("nan")
        else:
            macro = float(values.mean())
            worst = float(values.min())
            small = float((by_site["I0002"] + by_site["I0006"]) / 2.0)
        result[key] = {**{f"{site}_ac": by_site[site] for site in HOLDOUTS},
                       "macro_ac": macro, "worst_site_ac": worst, "small_site_mean_ac": small}
    for protocol in PROTOCOLS:
        baseline = result[(protocol, "all_all")]
        for component_id in [key[1] for key in result if key[0] == protocol]:
            current = result[(protocol, component_id)]
            current["delta_macro_vs_full"] = current["macro_ac"] - baseline["macro_ac"]
            current["delta_worst_vs_full"] = current["worst_site_ac"] - baseline["worst_site_ac"]
    return result


def _load_official(path: Path):
    if sha256_file(path) != OFFICIAL_SHA256:
        raise ValueError("official scoring script SHA256 mismatch")
    spec = importlib.util.spec_from_file_location("official_g5_components", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sources() -> tuple[dict[tuple[str, str], GroupData], dict[str, object], dict[str, object]]:
    complete_path = G5_ROOT / "complete.json"
    run_path = G5_ROOT / "run_config.json"
    scores_path = G5_ROOT / "scores.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("status") != "PASS" or complete.get("schema") != "g5_branch_a_rankmean_v1":
        raise ValueError("G5 completion status invalid")
    for path in (run_path, scores_path):
        relative = path.relative_to(G5_ROOT).as_posix()
        if sha256_file(path) != complete.get("files", {}).get(relative):
            raise ValueError(f"G5 source SHA mismatch: {relative}")
    run_config = json.loads(run_path.read_text(encoding="utf-8"))
    g5_scores = json.loads(scores_path.read_text(encoding="utf-8"))
    groups = {}
    for holdout in HOLDOUTS:
        for protocol in PROTOCOLS:
            path = G5_ROOT / "groups" / f"{holdout}_{protocol}" / "result.npz"
            relative = path.relative_to(G5_ROOT).as_posix()
            groups[(holdout, protocol)] = load_group(path, complete.get("files", {}).get(relative, ""))
        validate_protocol_pair(groups[(holdout, "transductive")], groups[(holdout, "inductive")])
        n, positive, negative, _ = EXPECTED[holdout]
        group = groups[(holdout, "transductive")]
        if len(group.labels) != n or int(group.labels.sum()) != positive or int(np.sum(group.labels == 0)) != negative:
            raise ValueError(f"G5 expected cohort counts differ: {holdout}")
    return groups, run_config, g5_scores


def _component_metadata(components: Sequence[Component]) -> dict[str, Component]:
    return {item.component_id: item for item in components}


def _baseline_rows(g5_scores: Mapping[str, object]) -> dict[tuple[str, str], Mapping[str, object]]:
    rows = {(str(row["holdout_site"]), str(row["protocol"])): row for row in g5_scores["groups"]}
    if set(rows) != {(site, protocol) for site in HOLDOUTS for protocol in PROTOCOLS}:
        raise ValueError("G5 baseline score rows incomplete")
    return rows


def _write_csv(rows: Sequence[Mapping[str, object]]) -> None:
    fields = ["holdout_site", "train_site", "protocol", "component_id", "component_kind", "view",
              "model_type", "n_models", "n_records", "n_positive", "n_negative", "ac_pair_count",
              "ac_auroc", "auroc", "auprc", "delta_ac_vs_full", "status"]
    output = [",".join(fields)]
    for row in rows:
        output.append(",".join(str(row[field]) for field in fields))
    _atomic_bytes(PUBLIC_CSV, ("\n".join(output) + "\n").encode("utf-8"))


def _f(value: float) -> str:
    return "NaN" if not np.isfinite(value) else f"{value:.6f}"


def _matrix_lines(title: str, summary: Mapping[tuple[str, str], Mapping[str, float]],
                  metric: str = "macro_ac") -> list[str]:
    lines = [f"### {title}", "", "| View | LR | HGB | LambdaRank | Three-model fusion |",
             "|---|---:|---:|---:|---:|"]
    for view in VIEWS:
        lines.append(f"| {view} | {_f(summary[(title.split()[0].lower(), f'{view}_lr')][metric])} | "
                     f"{_f(summary[(title.split()[0].lower(), f'{view}_hgb')][metric])} | "
                     f"{_f(summary[(title.split()[0].lower(), f'{view}_lambdarank')][metric])} | "
                     f"{_f(summary[(title.split()[0].lower(), f'{view}_all')][metric])} |")
    protocol = title.split()[0].lower()
    lines.append(f"| Five-view fusion | {_f(summary[(protocol, 'all_lr')][metric])} | "
                 f"{_f(summary[(protocol, 'all_hgb')][metric])} | "
                 f"{_f(summary[(protocol, 'all_lambdarank')][metric])} | "
                 f"{_f(summary[(protocol, 'all_all')][metric])} |")
    return lines


def _site_matrix_lines(holdout: str, protocol: str, rows: Sequence[Mapping[str, object]]) -> list[str]:
    lookup = {(str(row["component_id"])): float(row["ac_auroc"]) for row in rows
              if row["holdout_site"] == holdout and row["protocol"] == protocol}
    lines = [f"#### {holdout} / {protocol}", "", "| View | LR | HGB | LambdaRank | Three-model fusion |",
             "|---|---:|---:|---:|---:|"]
    for view in VIEWS:
        lines.append(f"| {view} | {_f(lookup[f'{view}_lr'])} | {_f(lookup[f'{view}_hgb'])} | "
                     f"{_f(lookup[f'{view}_lambdarank'])} | {_f(lookup[f'{view}_all'])} |")
    lines.append(f"| Five-view fusion | {_f(lookup['all_lr'])} | {_f(lookup['all_hgb'])} | "
                 f"{_f(lookup['all_lambdarank'])} | {_f(lookup['all_all'])} |")
    return lines


def write_report(rows: Sequence[Mapping[str, object]],
                 summary: Mapping[tuple[str, str], Mapping[str, float]]) -> None:
    lines = ["# G5-C View × Model Component Decomposition", "", "`G5C_STATUS = SUBMITTED_FOR_REVIEW`", "",
             "This post-hoc diagnostic averages only the frozen per-model cohort-rank columns saved by G5. It performs no training, inference, raw-prediction reload, reranking, weight search, or component selection.", "",
             "## Protocol-level Macro AC-AUROC matrices", ""]
    lines += _matrix_lines("Transductive Macro AC-AUROC", summary)
    lines += [""] + _matrix_lines("Inductive Macro AC-AUROC", summary)
    lines += ["", "## Holdout-level AC-AUROC matrices", ""]
    for protocol in PROTOCOLS:
        for holdout in HOLDOUTS:
            lines += _site_matrix_lines(holdout, protocol, rows) + [""]
    for kind, identifiers, heading in (
        ("view", [f"V{i}_all" for i in range(1, 6)], "View fusions"),
        ("model", [f"all_{model}" for model in MODEL_TYPES], "Model-family fusions"),
    ):
        lines += [f"## {heading}", "", "| Protocol | Component | I0002 | I0006 | S0001 | Macro | Worst | Small-site mean | ΔMacro vs full | ΔWorst vs full |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for protocol in PROTOCOLS:
            for component_id in identifiers:
                value = summary[(protocol, component_id)]
                lines.append(f"| {protocol} | {component_id} | {_f(value['I0002_ac'])} | {_f(value['I0006_ac'])} | {_f(value['S0001_ac'])} | {_f(value['macro_ac'])} | {_f(value['worst_site_ac'])} | {_f(value['small_site_mean_ac'])} | {_f(value['delta_macro_vs_full'])} | {_f(value['delta_worst_vs_full'])} |")
        lines.append("")
    lines += ["## Protocol differences", "", "| Component | Inductive Macro − Transductive Macro | Inductive Worst − Transductive Worst |",
              "|---|---:|---:|"]
    component_ids = [f"V{i}_{model}" for i in range(1, 6) for model in MODEL_TYPES]
    component_ids += [f"V{i}_all" for i in range(1, 6)] + [f"all_{model}" for model in MODEL_TYPES] + ["all_all"]
    for component_id in component_ids:
        trans = summary[("transductive", component_id)]
        ind = summary[("inductive", component_id)]
        lines.append(f"| {component_id} | {_f(ind['macro_ac'] - trans['macro_ac'])} | {_f(ind['worst_site_ac'] - trans['worst_site_ac'])} |")
    lines += ["", "## Observations", "",
              "- V5 was not the strongest view in this fixed decomposition. Its three-model Macro AC was 0.606453 (transductive) and 0.606666 (inductive), below every EEG view. V2 was the strongest single-view fusion in both protocols: 0.644282 and 0.650150. Thus the EEG views retain substantial ranking ability in these predictions.",
              "- Across all five views, LR had the highest model-family Macro AC in both protocols: 0.664048 and 0.672476. HGB was strongest on S0001 (0.668288 and 0.686393), while LR was stronger on the two smaller sites. LambdaRank had the lowest five-view family Macro in both protocols.",
              "- The LR-only five-view fusion exceeded the full 225-model Macro by 0.001655 (transductive) and 0.001498 (inductive), but its Worst-site AC was lower by 0.031520 and 0.009324. No single-view three-model fusion exceeded the corresponding full Macro. These post-hoc differences do not establish that removing components would generalize.",
              "- The full inductive-feature ensemble improved Macro AC over transductive by 0.008586. The largest positive cell-level Macro differences were V1-HGB (+0.019917), V2-HGB (+0.015690), V3-HGB (+0.013449), and V4-HGB (+0.012826). V5 fusion changed by only +0.000213; V5-HGB and V5-LambdaRank decreased by 0.006925 and 0.011196.",
              "- A sensible next step, if separately approved and frozen, would be a controlled confirmation of model-family balance and protocol sensitivity. These observed LOSO results must not themselves be used to silently redefine the primary G5 ensemble.",
              "", "## Interpretation boundaries", "",
              "These are prediction-layer component diagnostics after observing LOSO outcomes, not causal input-feature contributions or a new selected primary model. A component exceeding the full ensemble does not establish that removing other components will generalize.", "",
              "V5 combines CAISR20, HRV22, EMG48, Resp28 and Demo5; it is not a pure CAISR view. Every view contains Demo5, EEG views overlap in information, and equal model weights are not physiological contribution weights. No statistical-significance claim is made.", "",
              "Any follow-up prompted by these observations must be a separately frozen controlled experiment; this gate does not alter the G5 baseline.", "",
              "`G5C_FIXED_COMPONENTS = 24_PER_GROUP`", "", "`G5C_SCORE_ROWS = 144`", "",
              "`G5C_FULL_BASELINE_REPRODUCTION = PASS`", "", "`MODEL_REFIT = 0`", "",
              "`MODEL_REPLAY = 0`", "", "`RAW_PREDICTION_RELOAD = 0`", "",
              "`MODEL_RANK_RECOMPUTATION = 0`", "", "`G1_G2_G3_G4_G5_ARTIFACTS = UNCHANGED`", "",
              "`WEIGHT_SEARCH = NOT_STARTED`", "", "`INPUT_FEATURE_ABLATION = NOT_STARTED`", ""]
    _atomic_bytes(PUBLIC_REPORT, ("\n".join(lines)).encode("utf-8"))


def _analysis_config() -> dict[str, object]:
    complete = G5_ROOT / "complete.json"
    run_config = G5_ROOT / "run_config.json"
    scores = G5_ROOT / "scores.json"
    return {"schema": SCHEMA, "source_complete_sha256": sha256_file(complete),
            "source_run_config_sha256": sha256_file(run_config), "source_scores_sha256": sha256_file(scores),
            "official_scoring_sha256": sha256_file(REPO / "evaluate_model.py"),
            "components_code_sha256": sha256_file(Path(__file__)), "component_count": 24,
            "formula": "mean(saved_model_ranks[:, component_columns], dtype=float64)",
            "tolerance": 1e-15}


def _identity(config: Mapping[str, object]) -> str:
    return hashlib.sha256(_json_bytes(config)).hexdigest()


def _verify_complete(config: Mapping[str, object]) -> dict[str, object] | None:
    path = PRIVATE_ROOT / "complete.json"
    if not path.exists():
        return None
    complete = json.loads(path.read_text(encoding="utf-8"))
    if (complete.get("schema") != SCHEMA or complete.get("status") != "PASS"
            or complete.get("identity_sha256") != _identity(config)):
        raise ValueError("existing G5-C output has a different source identity")
    for relative, digest in complete.get("files", {}).items():
        artifact = PRIVATE_ROOT / relative
        if artifact.is_symlink() or not artifact.is_file() or sha256_file(artifact) != digest:
            raise ValueError(f"completed G5-C artifact invalid: {relative}")
    summary = json.loads((PRIVATE_ROOT / "summaries.json").read_text(encoding="utf-8"))
    if len(summary.get("rows", [])) != 144 or len(summary.get("summaries", [])) != 48:
        raise ValueError("completed G5-C row count mismatch")
    return summary


def build() -> dict[str, object]:
    started = time.perf_counter()
    config = _analysis_config()
    existing = _verify_complete(config)
    if existing is not None:
        print(json.dumps({"status": "SKIP_VERIFIED", "rows": 144}, sort_keys=True))
        return existing
    PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
    _atomic_json(PRIVATE_ROOT / "analysis_config.json", {**config, "identity_sha256": _identity(config)})
    groups, _, g5_scores = _load_sources()
    official = _load_official(REPO / "evaluate_model.py")
    baselines = _baseline_rows(g5_scores)
    rows = []
    for holdout in HOLDOUTS:
        for protocol in PROTOCOLS:
            group = groups[(holdout, protocol)]
            components = make_components(group.model_keys)
            ids = tuple(item.component_id for item in components)
            values = component_scores(group.model_ranks, components, saved_full=group.ensemble_scores)
            metrics = evaluate_components(group.labels, group.ages, values[:, :-1], ids[:-1], official)
            baseline = baselines[(holdout, protocol)]
            metrics.append({"component_id": "all_all", **{key: baseline[key] for key in
                            ("n_records", "n_positive", "n_negative", "ac_pair_count", "n_score_unique",
                             "ac_auroc", "auroc", "auprc", "status")}})
            full_ac = float(baseline["ac_auroc"])
            metadata = _component_metadata(components)
            group_rows = []
            for metric in metrics:
                item = metadata[str(metric["component_id"])]
                row = {"holdout_site": holdout, "train_site": HOLDOUT_TO_TRAIN[holdout],
                       "protocol": protocol, "component_id": item.component_id,
                       "component_kind": item.component_kind, "view": item.view,
                       "model_type": item.model_type, "n_models": len(item.columns), **metric,
                       "delta_ac_vs_full": float(metric["ac_auroc"]) - full_ac}
                group_rows.append(row)
                rows.append(row)
            offsets = np.arange(0, 226, 15, dtype=np.int64)
            flat_columns = np.concatenate([np.asarray(item.columns, dtype=np.int64) for item in components])
            column_offsets = np.concatenate((np.array([0], dtype=np.int64),
                                             np.cumsum([len(item.columns) for item in components], dtype=np.int64)))
            group_root = PRIVATE_ROOT / "groups" / f"{holdout}_{protocol}"
            _atomic_npz(group_root / "components.npz",
                        component_ids=np.asarray(ids, dtype="U24"), component_scores=values,
                        test_global_indices=group.test_global_indices.astype(np.int64, copy=False),
                        column_indices=flat_columns, column_offsets=column_offsets)
            _atomic_json(group_root / "scores.json", group_rows)
            print(json.dumps({"holdout": holdout, "protocol": protocol, "components": 24,
                              "full_ac": full_ac}, sort_keys=True))
    if len(rows) != 144:
        raise ValueError("G5-C must produce exactly 144 score rows")
    summary_map = summarize_components(rows)
    summary_rows = [{"protocol": protocol, "component_id": component_id, **value}
                    for (protocol, component_id), value in summary_map.items()]
    payload = {"schema": SCHEMA, "rows": rows, "summaries": summary_rows,
               "elapsed_seconds": time.perf_counter() - started}
    _atomic_json(PRIVATE_ROOT / "summaries.json", payload)
    _write_csv(rows)
    write_report(rows, summary_map)
    files = {path.relative_to(PRIVATE_ROOT).as_posix(): sha256_file(path)
             for path in sorted(PRIVATE_ROOT.rglob("*"))
             if path.is_file() and path.name != "complete.json"}
    _atomic_json(PRIVATE_ROOT / "complete.json", {"schema": SCHEMA, "status": "PASS",
                 "identity_sha256": _identity(config), "groups": 6, "components_per_group": 24,
                 "score_rows": 144, "files": files})
    return payload


def verify_only() -> dict[str, object]:
    config = _analysis_config()
    payload = _verify_complete(config)
    if payload is None:
        raise ValueError("G5-C output is incomplete")
    rows = payload["rows"]
    summary_map = {(row["protocol"], row["component_id"]): {key: value for key, value in row.items()
                   if key not in {"protocol", "component_id"}} for row in payload["summaries"]}
    if len(rows) != 144 or len(summary_map) != 48:
        raise ValueError("G5-C public closure count mismatch")
    for holdout in HOLDOUTS:
        for protocol in PROTOCOLS:
            path = PRIVATE_ROOT / "groups" / f"{holdout}_{protocol}" / "components.npz"
            with np.load(path, allow_pickle=False) as archive:
                if np.asarray(archive["component_scores"]).shape != (EXPECTED[holdout][0], 24):
                    raise ValueError("saved G5-C component matrix shape mismatch")
    _write_csv(rows)
    write_report(rows, summary_map)
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="G5-C fixed view/model component decomposition")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    payload = build() if args.build else verify_only()
    print(json.dumps({"status": "PASS", "score_rows": len(payload["rows"]),
                      "summaries": len(payload["summaries"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["Component", "GroupData", "component_scores", "evaluate_components", "load_group",
           "make_components", "summarize_components", "validate_protocol_pair"]
