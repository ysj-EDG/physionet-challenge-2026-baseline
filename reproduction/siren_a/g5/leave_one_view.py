"""Fixed prediction-layer leave-one-view-out analysis of frozen G5 ranks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from reproduction.siren_a.g5.components import (
    _atomic_bytes,
    _atomic_json,
    _atomic_npz,
    load_group,
)
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
PRIVATE_ROOT = Path("/database/home/gaohaojie/workspace/python-example-2026-siren-a-g5-private/leave_one_view_v1")
PUBLIC_CSV = REPO / "reproduction/siren_a/audit/g5d_leave_one_view_scores.csv"
PUBLIC_REPORT = REPO / "reproduction/siren_a/docs/G5D_LEAVE_ONE_VIEW_REPORT.md"
OFFICIAL_SHA256 = "227e40a45a7ddb918812c17fd4a8e678334134c965d488db763409d245802970"
SCHEMA = "g5d_leave_one_view_v1"
EXPECTED = {
    "I0002": (319, 52, 267, 1886),
    "I0006": (1142, 112, 1030, 14493),
    "S0001": (5139, 334, 4805, 182330),
}
KEY_RE = re.compile(r"^seed(\d+)\|(V[1-5])\|(lr|hgb|lambdarank)$")
CONFIGURATIONS = ("full", "drop_V1", "drop_V2", "drop_V3", "drop_V4", "drop_V5")
CLOSURE_ATOL = 2e-15


@dataclass(frozen=True)
class ViewRemoval:
    name: str
    dropped_view: str
    removed: tuple[int, ...]
    remaining: tuple[int, ...]


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _identity(config: Mapping[str, object]) -> str:
    return hashlib.sha256(_json_bytes(config)).hexdigest()


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


def select_remaining_columns(model_keys: Sequence[str]) -> dict[str, ViewRemoval]:
    parsed = _parse_keys(model_keys)
    result = {}
    all_columns = set(range(225))
    for view in VIEWS:
        removed = tuple(index for index, (_, found_view, _) in enumerate(parsed) if found_view == view)
        remaining = tuple(index for index in range(225) if index not in set(removed))
        if len(removed) != 45 or len(remaining) != 180 or set(removed) & set(remaining):
            raise ValueError(f"invalid fixed view partition: {view}")
        if set(removed) | set(remaining) != all_columns:
            raise ValueError(f"incomplete fixed view partition: {view}")
        result[f"drop_{view}"] = ViewRemoval(f"drop_{view}", view, removed, remaining)
    return result


def leave_one_view_scores(model_ranks: np.ndarray, model_keys: Sequence[str],
                          saved_full: np.ndarray) -> tuple[tuple[str, ...], np.ndarray]:
    ranks = np.asarray(model_ranks)
    full = np.asarray(saved_full)
    if ranks.ndim != 2 or ranks.shape[1] != 225 or ranks.dtype != np.float64:
        raise ValueError("model_ranks must have shape (N,225) and dtype float64")
    if not np.isfinite(ranks).all() or np.any(ranks < 0) or np.any(ranks > 1):
        raise ValueError("model ranks must be finite and within [0,1]")
    recomputed_full = ranks.mean(axis=1, dtype=np.float64)
    if full.shape != (ranks.shape[0],) or full.dtype != np.float64 or not np.isfinite(full).all():
        raise ValueError("saved full ensemble is invalid")
    if not np.allclose(recomputed_full, full, rtol=0, atol=1e-15):
        raise ValueError("saved full ensemble does not close against 225 ranks")
    selections = select_remaining_columns(model_keys)
    columns = [full]
    for name in CONFIGURATIONS[1:]:
        selection = selections[name]
        direct = ranks[:, np.asarray(selection.remaining)].mean(axis=1, dtype=np.float64)
        dropped_view = ranks[:, np.asarray(selection.removed)].mean(axis=1, dtype=np.float64)
        equivalent = (5.0 * full - dropped_view) / 4.0
        if not np.allclose(direct, equivalent, rtol=0, atol=CLOSURE_ATOL):
            raise ValueError(f"leave-one-view arithmetic closure failed: {name}")
        columns.append(direct)
    result = np.column_stack(columns)
    if not np.isfinite(result).all():
        raise ValueError("leave-one-view scores contain non-finite values")
    return CONFIGURATIONS, result


def validate_protocol_pair(left, right) -> None:
    def get(value, key):
        return getattr(value, key) if hasattr(value, key) else value[key]
    for key in ("test_global_indices", "labels", "ages"):
        if not np.array_equal(get(left, key), get(right, key)):
            raise ValueError(f"protocol evaluation rows differ: {key}")


def evaluate(labels: np.ndarray, ages: np.ndarray, scores: np.ndarray, configuration_ids: Sequence[str],
             official, baseline: Mapping[str, object]) -> list[dict[str, object]]:
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[1] != len(configuration_ids):
        raise ValueError("score matrix and configuration IDs differ")
    if tuple(configuration_ids)[0] != "full":
        raise ValueError("full baseline must be the first configuration")
    count_keys = ("n_records", "n_positive", "n_negative", "ac_pair_count", "ac_auroc", "auroc", "auprc", "status")
    full_row = {key: baseline[key] for key in count_keys}
    rows = [{"configuration": "full", "dropped_view": "none", "n_models": 225,
             **full_row, "delta_ac_vs_full": 0.0}]
    full_ac = float(baseline["ac_auroc"])
    for column, configuration in enumerate(configuration_ids[1:], start=1):
        metrics = score_site(labels, values[:, column], ages, official)
        rows.append({"configuration": str(configuration), "dropped_view": str(configuration).removeprefix("drop_"),
                     "n_models": 180, **metrics,
                     "delta_ac_vs_full": float(metrics["ac_auroc"]) - full_ac})
    return rows


def summarize(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        key = str(row["protocol"]), str(row["configuration"])
        site = str(row["holdout_site"])
        if site in grouped.setdefault(key, {}):
            raise ValueError("duplicate site/configuration row")
        grouped[key][site] = float(row["ac_auroc"])
    result = {}
    for key, by_site in grouped.items():
        if set(by_site) != set(HOLDOUTS):
            raise ValueError("summary requires all three holdouts")
        values = np.asarray([by_site[site] for site in HOLDOUTS], dtype=np.float64)
        if not np.isfinite(values).all():
            macro = worst = small = float("nan")
        else:
            macro = float(values.mean())
            worst = float(values.min())
            small = float((by_site["I0002"] + by_site["I0006"]) / 2.0)
        result[key] = {**{f"{site}_ac": by_site[site] for site in HOLDOUTS}, "macro_ac": macro,
                       "worst_site_ac": worst, "small_site_mean_ac": small}
    for protocol in PROTOCOLS:
        baseline = result[(protocol, "full")]
        for configuration in (item[1] for item in result if item[0] == protocol):
            current = result[(protocol, configuration)]
            current["delta_macro_vs_full"] = current["macro_ac"] - baseline["macro_ac"]
            current["delta_worst_vs_full"] = current["worst_site_ac"] - baseline["worst_site_ac"]
    return result


def _load_official(path: Path):
    if sha256_file(path) != OFFICIAL_SHA256:
        raise ValueError("official scoring script SHA256 mismatch")
    spec = importlib.util.spec_from_file_location("official_g5d", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_inputs():
    complete_path = G5_ROOT / "complete.json"
    run_path = G5_ROOT / "run_config.json"
    scores_path = G5_ROOT / "scores.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("schema") != "g5_branch_a_rankmean_v1" or complete.get("status") != "PASS":
        raise ValueError("G5 completion status invalid")
    for path in (run_path, scores_path):
        relative = path.relative_to(G5_ROOT).as_posix()
        if sha256_file(path) != complete.get("files", {}).get(relative):
            raise ValueError(f"G5 source SHA mismatch: {relative}")
    run_config = json.loads(run_path.read_text(encoding="utf-8"))
    source_scores = json.loads(scores_path.read_text(encoding="utf-8"))
    baseline = {(str(row["holdout_site"]), str(row["protocol"])): row for row in source_scores["groups"]}
    if set(baseline) != {(site, protocol) for site in HOLDOUTS for protocol in PROTOCOLS}:
        raise ValueError("G5 baseline is incomplete")
    groups = {}
    for holdout in HOLDOUTS:
        for protocol in PROTOCOLS:
            path = G5_ROOT / "groups" / f"{holdout}_{protocol}" / "result.npz"
            relative = path.relative_to(G5_ROOT).as_posix()
            groups[(holdout, protocol)] = load_group(path, complete.get("files", {}).get(relative, ""))
        validate_protocol_pair(groups[(holdout, "transductive")], groups[(holdout, "inductive")])
        n, positive, negative, pairs = EXPECTED[holdout]
        group = groups[(holdout, "transductive")]
        if (len(group.labels), int(group.labels.sum()), int(np.sum(group.labels == 0))) != (n, positive, negative):
            raise ValueError(f"expected cohort counts differ: {holdout}")
        if int(baseline[(holdout, "transductive")]["ac_pair_count"]) != pairs:
            raise ValueError(f"expected AC pair count differs: {holdout}")
    return groups, baseline, run_config, source_scores


def _analysis_config() -> dict[str, object]:
    source_results = {
        f"{holdout}_{protocol}": sha256_file(G5_ROOT / "groups" / f"{holdout}_{protocol}" / "result.npz")
        for holdout in HOLDOUTS for protocol in PROTOCOLS
    }
    return {"schema": SCHEMA, "source_complete_sha256": sha256_file(G5_ROOT / "complete.json"),
            "source_run_config_sha256": sha256_file(G5_ROOT / "run_config.json"),
            "source_scores_sha256": sha256_file(G5_ROOT / "scores.json"),
            "source_result_sha256": source_results,
            "official_scoring_sha256": sha256_file(REPO / "evaluate_model.py"),
            "code_sha256": sha256_file(Path(__file__)), "configurations": list(CONFIGURATIONS),
            "formula": "mean(saved_model_ranks[:, remaining_180_columns], dtype=float64)",
            "official_gap": 2, "closure_atol": CLOSURE_ATOL}


def _verify_complete(config: Mapping[str, object]):
    complete_path = PRIVATE_ROOT / "complete.json"
    if not complete_path.exists():
        return None
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if (complete.get("schema") != SCHEMA or complete.get("status") != "PASS"
            or complete.get("identity_sha256") != _identity(config)):
        raise ValueError("existing G5-D output has a different source identity")
    for relative, digest in complete.get("files", {}).items():
        path = PRIVATE_ROOT / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"completed G5-D artifact invalid: {relative}")
    payload = json.loads((PRIVATE_ROOT / "scores.json").read_text(encoding="utf-8"))
    if len(payload.get("rows", [])) != 36 or len(payload.get("summaries", [])) != 12:
        raise ValueError("completed G5-D row count mismatch")
    return payload


def _write_csv(rows: Sequence[Mapping[str, object]]) -> None:
    fields = ["holdout_site", "train_site", "protocol", "configuration", "dropped_view", "n_models",
              "n_records", "n_positive", "n_negative", "ac_pair_count", "ac_auroc", "auroc", "auprc",
              "delta_ac_vs_full", "status"]
    lines = [",".join(fields)] + [",".join(str(row[field]) for field in fields) for row in rows]
    _atomic_bytes(PUBLIC_CSV, ("\n".join(lines) + "\n").encode("utf-8"))


def _f(value: float) -> str:
    return "NaN" if not np.isfinite(value) else f"{value:.6f}"


def write_report(summary: Mapping[tuple[str, str], Mapping[str, float]]) -> None:
    lines = ["# G5-D Fixed Leave-One-View-Out Analysis", "", "`G5D_STATUS = SUBMITTED_FOR_REVIEW`", "",
             "This prediction-layer diagnostic directly averages the frozen G5 per-model cohort ranks after removing all 45 columns trained on one view. It performs no training, model loading, inference, raw-prediction reload, reranking, weight search or input-feature change.", ""]
    for protocol, title in (("transductive", "Transductive"), ("inductive", "Inductive-feature")):
        lines += [f"## {title}", "", "| Configuration | Models | I0002 AC | I0006 AC | S0001 AC | Macro AC | Worst-site AC | Small-site mean AC | ΔMacro vs full | ΔWorst vs full |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for configuration in CONFIGURATIONS:
            item = summary[(protocol, configuration)]
            models = 225 if configuration == "full" else 180
            lines.append(f"| {configuration} | {models} | {_f(item['I0002_ac'])} | {_f(item['I0006_ac'])} | {_f(item['S0001_ac'])} | {_f(item['macro_ac'])} | {_f(item['worst_site_ac'])} | {_f(item['small_site_mean_ac'])} | {_f(item['delta_macro_vs_full'])} | {_f(item['delta_worst_vs_full'])} |")
        lines.append("")
    lines += ["## Interpretation", ""]
    for protocol in PROTOCOLS:
        values = [(configuration, summary[(protocol, configuration)]["delta_macro_vs_full"],
                   summary[(protocol, configuration)]["delta_worst_vs_full"]) for configuration in CONFIGURATIONS[1:]]
        best_macro = max(values, key=lambda item: item[1])
        worst_macro = min(values, key=lambda item: item[1])
        lines.append(f"- In {protocol}, the largest Macro change was {best_macro[0]} ({best_macro[1]:+.6f}); the most negative was {worst_macro[0]} ({worst_macro[1]:+.6f}). Macro and Worst-site changes should be considered separately because a removal can redistribute performance across sites.")
    lines += ["", "A positive delta only means the fixed 180-rank ensemble scored higher on these already observed LOSO cohorts; it does not prove that the removed view lacks physiological value or that removal will generalize. A negative delta indicates marginal complementarity within this fixed ensemble, not causal attribution.", "",
              "`drop_V5` removes the complete V5 model block (CAISR20, HRV22, EMG48, Resp28 and Demo5), not CAISR alone. Every remaining view still contains Demo5, and the retained views each change weight from 20% to 25%. The EEG views overlap in information, so ensemble weight is not physiological contribution.", "",
              "The inductive-feature route still uses test-cohort output ranking and is not end-to-end inductive deployment. This is post-hoc exploratory diagnosis after observing LOSO outcomes; no significance claim, best-subset selection or mainline replacement is made. The five-view primary baseline remains unchanged.", "",
              "All 36 formal site/protocol/configuration rows, including AUROC and AUPRC, are in `reproduction/siren_a/audit/g5d_leave_one_view_scores.csv`.", "",
              "`G5D_FIXED_REMOVALS = 5_PER_GROUP`", "`G5D_NEW_SCORE_ROWS = 30`",
              "`G5D_BASELINE_ROWS_REUSED = 6`", "`G5D_TOTAL_SCORE_ROWS = 36`",
              "`MODEL_REFIT = 0`", "`MODEL_LOAD = 0`", "`MODEL_REPLAY = 0`",
              "`RAW_PREDICTION_RELOAD = 0`", "`MODEL_RANK_RECOMPUTATION = 0`",
              "`G1_G2_G3_G4_G5_G5C_ARTIFACTS = UNCHANGED`", "`FIVE_VIEW_MAINLINE = UNCHANGED`",
              "`NEW_FEATURE_DESIGN = NOT_STARTED`", "`FEATURE_EXTRACTION_CHANGE = NOT_STARTED`",
              "`HYPERPARAMETER_SEARCH = NOT_STARTED`", ""]
    _atomic_bytes(PUBLIC_REPORT, "\n".join(lines).encode("utf-8"))


def _publish_from_payload(payload: Mapping[str, object]) -> None:
    summary = {(str(row["protocol"]), str(row["configuration"])):
               {key: value for key, value in row.items() if key not in {"protocol", "configuration"}}
               for row in payload["summaries"]}
    if len(payload["rows"]) != 36 or len(summary) != 12:
        raise ValueError("G5-D public payload is incomplete")
    _write_csv(payload["rows"])
    write_report(summary)


def build():
    started = time.perf_counter()
    config = _analysis_config()
    existing = _verify_complete(config)
    if existing is not None:
        _publish_from_payload(existing)
        print(json.dumps({"status": "SKIP_VERIFIED", "rows": 36}, sort_keys=True))
        return existing
    PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
    _atomic_json(PRIVATE_ROOT / "config.json", {**config, "identity_sha256": _identity(config)})
    groups, baselines, _, _ = load_inputs()
    official = _load_official(REPO / "evaluate_model.py")
    rows = []
    for holdout in HOLDOUTS:
        for protocol in PROTOCOLS:
            group = groups[(holdout, protocol)]
            names, scores = leave_one_view_scores(group.model_ranks, group.model_keys, group.ensemble_scores)
            metrics = evaluate(group.labels, group.ages, scores, names, official, baselines[(holdout, protocol)])
            group_rows = []
            for row in metrics:
                enriched = {"holdout_site": holdout, "train_site": HOLDOUT_TO_TRAIN[holdout],
                            "protocol": protocol, **row}
                rows.append(enriched)
                group_rows.append(enriched)
            selections = select_remaining_columns(group.model_keys)
            removed_keys = np.asarray([[str(group.model_keys[index]) for index in selections[name].removed]
                                       for name in CONFIGURATIONS[1:]], dtype="U32")
            remaining_keys = np.asarray([[str(group.model_keys[index]) for index in selections[name].remaining]
                                         for name in CONFIGURATIONS[1:]], dtype="U32")
            root = PRIVATE_ROOT / "groups" / f"{holdout}_{protocol}"
            _atomic_npz(root / "scores.npz", configuration_ids=np.asarray(names, dtype="U16"),
                        scores=scores, test_global_indices=group.test_global_indices.astype(np.int64, copy=False),
                        removed_model_keys=removed_keys, remaining_model_keys=remaining_keys)
            _atomic_json(root / "metrics.json", group_rows)
            print(json.dumps({"holdout": holdout, "protocol": protocol,
                              "full_ac": metrics[0]["ac_auroc"],
                              "drop_ac": {item["configuration"]: item["ac_auroc"] for item in metrics[1:]}}, sort_keys=True))
    if len(rows) != 36:
        raise ValueError("G5-D must produce 36 total rows")
    summary_map = summarize(rows)
    summary_rows = [{"protocol": protocol, "configuration": configuration, **value}
                    for (protocol, configuration), value in summary_map.items()]
    payload = {"schema": SCHEMA, "rows": rows, "summaries": summary_rows,
               "elapsed_seconds": time.perf_counter() - started}
    _atomic_json(PRIVATE_ROOT / "scores.json", payload)
    _publish_from_payload(payload)
    files = {path.relative_to(PRIVATE_ROOT).as_posix(): sha256_file(path)
             for path in sorted(PRIVATE_ROOT.rglob("*")) if path.is_file() and path.name != "complete.json"}
    _atomic_json(PRIVATE_ROOT / "complete.json", {"schema": SCHEMA, "status": "PASS",
                 "identity_sha256": _identity(config), "groups": 6, "new_score_rows": 30,
                 "baseline_rows_reused": 6, "total_score_rows": 36, "files": files})
    return payload


def verify_only():
    config = _analysis_config()
    payload = _verify_complete(config)
    if payload is None:
        raise ValueError("G5-D output is incomplete")
    for holdout in HOLDOUTS:
        for protocol in PROTOCOLS:
            path = PRIVATE_ROOT / "groups" / f"{holdout}_{protocol}" / "scores.npz"
            with np.load(path, allow_pickle=False) as archive:
                if np.asarray(archive["scores"]).shape != (EXPECTED[holdout][0], 6):
                    raise ValueError("saved G5-D score matrix shape mismatch")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.build == args.verify_only:
        parser.error("choose exactly one of --build or --verify-only")
    payload = build() if args.build else verify_only()
    print(json.dumps({"status": "PASS", "rows": len(payload["rows"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
