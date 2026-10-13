"""Bounded G2-B0 Pilot helpers; deliberately exposes no full-cohort execution mode."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import platform
from pathlib import Path

import numpy as np

import g2a_features
import g2a_io
import g2b0_batch_safety
from g2a_features import compute_night_features, entropy_source_valid
from g2a_io import load_manifest, load_npz_record, read_cache, sha256_file, write_cache_atomic
from g2b0_batch_safety import Action, CACHE_SCHEMA_VERSION, assert_valid_cache_arrays, decide_resume_action

SEED = 20261013
GOLDEN_RECORDS = 18
MAX_NEW_PILOT = 12
MAX_TOTAL_RECORDS = 30
MAX_DISTINCT_READS = 80
SITES = ("I0002", "I0006", "S0001")


class ReadBudget:
    def __init__(self, loader, maximum=MAX_DISTINCT_READS):
        self.loader = loader
        self.maximum = maximum
        self.paths = set()
        self.read_calls = 0

    @property
    def distinct_count(self): return len(self.paths)

    def load(self, path):
        path = Path(path)
        key = str(path.resolve(strict=False))
        if key not in self.paths and len(self.paths) >= self.maximum:
            raise RuntimeError(f"Distinct NPZ read budget {self.maximum} exceeded")
        self.paths.add(key); self.read_calls += 1
        return self.loader(path)


def deterministic_candidates(paths, golden_sources, seed=SEED, limit=MAX_DISTINCT_READS-GOLDEN_RECORDS):
    golden = {str(Path(path).resolve(strict=False)) for path in golden_sources}
    grouped = {site: [] for site in SITES}
    for path in sorted(map(Path, paths)):
        if str(path.resolve(strict=False)) not in golden and path.parent.name in grouped:
            grouped[path.parent.name].append(path)
    rng = np.random.default_rng(seed)
    for site in SITES:
        order = rng.permutation(len(grouped[site]))
        grouped[site] = [grouped[site][int(index)] for index in order]
    selected = []; offset = 0
    while len(selected) < limit:
        added = False
        for site in SITES:
            if offset < len(grouped[site]) and len(selected) < limit:
                selected.append(grouped[site][offset]); added = True
        if not added: break
        offset += 1
    return selected


def _probability_lengths_match(record):
    total = len(record["stage_valid"]); lengths = record["annotation_signal_lengths"]
    aliases = (("caisr_prob_n1",), ("caisr_prob_n2",), ("caisr_prob_n3",),
               ("caisr_prob_r", "caisr_prob_rem"), ("caisr_prob_w", "caisr_prob_wake"))
    return all(any(lengths.get(name) == total for name in names) for names in aliases)


def boundary_flags(record):
    hrv_values = np.asarray(record["X_ecg"], dtype=float)[:, :11]
    hrv_valid = (np.asarray(record["hrv_success"], dtype=bool)[:, None]
                 & np.asarray(record["hrv_feature_valid"], dtype=bool)
                 & np.isfinite(hrv_values))
    stage_valid = np.asarray(record["stage_valid"], dtype=bool)
    stage_code = np.asarray(record["stage_code_aligned"], dtype=float)
    eeg_epoch_lead = (stage_valid[:, None] & np.isin(stage_code, (1, 2, 3, 4, 5))[:, None]
                      & np.asarray(record["eeg_channel_available"], dtype=bool)[None, :]
                      & (np.asarray(record["eeg_clean_subsegment_count"]) > 0))
    eeg_values = np.asarray(record["X_seq"], dtype=float)[:, :54].reshape(len(stage_valid), 6, 9)
    has_eeg_output = bool(np.any(np.sum(eeg_epoch_lead[:, :, None] & np.isfinite(eeg_values), axis=0) >= 5))
    probabilities = np.asarray(record["stage_probability"], dtype=float)
    non9_oor = np.isfinite(probabilities) & (probabilities != 9) & ((probabilities < 0) | (probabilities > 1))
    stage_valid = np.asarray(record["stage_valid"], dtype=bool)
    incomplete = (not np.asarray(record["stage_probability_channels_present"], dtype=bool).all()
                  or not _probability_lengths_match(record))
    return {
        "no_hrv": not bool(hrv_valid.any()) and has_eeg_output,
        "annotation_missing": record["annotation_status"] == "missing" and has_eeg_output,
        "race_unavailable": bool(np.asarray(record["x_static"])[7] == 1),
        "probability_incomplete": bool(incomplete) and record["annotation_status"] == "available",
        "non9_probability_out_of_range": bool(non9_oor.any()),
        "probability_anomaly_stage_valid": bool(np.any(non9_oor[stage_valid])),
        "probability_anomaly_stage_invalid": bool(np.any(non9_oor[~stage_valid])),
    }


def select_targeted_records(candidate_paths, budget, maximum=MAX_NEW_PILOT):
    categories = ("no_hrv", "annotation_missing", "race_unavailable", "probability_incomplete",
                  "non9_probability_out_of_range", "probability_anomaly_stage_valid",
                  "probability_anomaly_stage_invalid")
    coverage = {category: False for category in categories}; selected = []
    for path in map(Path, candidate_paths):
        record = budget.load(path); flags = boundary_flags(record)
        newly_covered = [category for category in categories if flags[category] and not coverage[category]]
        if newly_covered and len(selected) < maximum:
            selected.append((path, record, flags))
            for category in categories:
                coverage[category] |= flags[category]
    return selected, coverage


def probability_qc(record):
    values = np.asarray(record["stage_probability"], dtype=float)
    stage_valid = np.asarray(record["stage_valid"], dtype=bool)
    finite = np.isfinite(values); nine = finite & (values == 9)
    non9 = finite & ~nine; below = non9 & (values < 0); above = non9 & (values > 1)
    between = non9 & (values > 1) & (values < 9); over9 = non9 & (values > 9)
    abnormal = nine | below | above | ~finite
    non9_oor = below | above
    return {
        "value_9": int(nine.sum()),
        "non9_lt_0": int(below.sum()),
        "non9_gt_1": int(above.sum()),
        "non9_between_1_9": int(between.sum()),
        "non9_gt_9": int(over9.sum()),
        "nonfinite": int((~finite).sum()),
        "value_9_stage_valid": int(nine[stage_valid].sum()),
        "value_9_stage_invalid": int(nine[~stage_valid].sum()),
        "non9_oor_stage_valid": int(non9_oor[stage_valid].sum()),
        "non9_oor_stage_invalid": int(non9_oor[~stage_valid].sum()),
        "value_9_by_channel": [int(nine[:, index].sum()) for index in range(5)],
        "non9_oor_by_channel": [int(non9_oor[:, index].sum()) for index in range(5)],
        "anomalous_rows": int(abnormal.any(axis=1).sum()),
        "source_lengths_match": bool(_probability_lengths_match(record)),
        "entropy_source_valid": bool(entropy_source_valid(record, float(record["x_static"][87]))),
    }


def assert_boundary_effects(category, result):
    ids = result.unique_ids; valid = result.unique_feature_valid_mask
    positions = {uid: index for index, uid in enumerate(ids)}
    if category == "no_hrv":
        target = [index for index, uid in enumerate(ids) if uid.startswith("hrv_")]
        unaffected = [index for index, uid in enumerate(ids) if uid.startswith("eeg_")]
        ok = len(target) == 22 and not valid[target].any() and valid[unaffected].any()
    elif category == "annotation_missing":
        target = [index for index, uid in enumerate(ids) if uid.startswith("caisr_")]
        unaffected = [index for index, uid in enumerate(ids) if uid.startswith("eeg_")]
        ok = len(target) == 20 and not valid[target].any() and valid[unaffected].any()
    elif category == "race_unavailable":
        target = [positions[name] for name in ("demo_race_white", "demo_race_black", "demo_race_asian")]
        ok = not valid[target].any()
    elif category == "probability_incomplete":
        entropy = positions["caisr_sleep_mean_stage_entropy"]
        other = [index for index, uid in enumerate(ids) if uid.startswith("caisr_") and index != entropy]
        ok = not valid[entropy] and len(other) == 19 and valid[other].any()
    else:
        raise ValueError(f"Unknown boundary category: {category}")
    if not ok: raise ValueError(f"Boundary effect mismatch: {category}")
    return True


def build_pilot_payload(result, provenance):
    payload = {
        "unique_feature_ids": np.asarray(result.unique_ids),
        "unique_values": np.asarray(result.unique_feature_values, dtype=np.float64),
        "unique_valid_mask": np.asarray(result.unique_feature_valid_mask, dtype=bool),
        "unique_support_count": np.asarray(result.unique_feature_support_count, dtype=np.int64),
    }
    for view in result.view_values:
        payload[f"{view}_values"] = np.asarray(result.view_values[view], dtype=np.float64)
        payload[f"{view}_valid_mask"] = np.asarray(result.view_valid_masks[view], dtype=bool)
        payload[f"{view}_support_count"] = np.asarray(result.view_support_counts[view], dtype=np.int64)
    payload.update({key: np.asarray(value) for key, value in provenance.items()})
    return payload


def build_inventory(root, expected_counts=None):
    root = Path(root).resolve(); rows = []; counts = {site: 0 for site in SITES}; keys = set()
    entries = sorted(path for site in SITES for path in (root / site).iterdir())
    for path in entries:
        if path.is_symlink(): raise ValueError("Inventory contains symlink")
        if not path.is_file() or path.suffix != ".npz": raise ValueError("Inventory contains unexpected entry")
        resolved = path.resolve()
        try: relative = resolved.relative_to(root)
        except ValueError as exc: raise ValueError("Inventory path escapes input root") from exc
        if len(relative.parts) != 2 or relative.parts[0] not in counts: raise ValueError("Inventory layout")
        key = f"{relative.parts[0]}/{path.stem}"
        if key in keys: raise ValueError("Duplicate record key")
        keys.add(key); counts[relative.parts[0]] += 1
        stat = path.stat()
        rows.append({"relative_path": relative.as_posix(), "site": relative.parts[0],
                     "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    if expected_counts is not None and counts != expected_counts:
        raise ValueError(f"Inventory counts mismatch: {counts}")
    return rows, {"total": len(rows), "site_counts": counts, "duplicates": 0, "symlinks": 0}


def validate_public_report_text(text):
    forbidden = ("/database/", ".npz", "BidsFolder", "SessionID", "BDSPPatientID")
    found = [token for token in forbidden if token in text]
    if found: raise ValueError("Public report contains private token")
    return True


def write_json_no_clobber(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", prefix=path.name + ".",
                                         dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name); json.dump(data, handle, indent=2, sort_keys=True)
            handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    finally:
        if temporary is not None and temporary.exists(): temporary.unlink()


def _assert_result_matches_cache(result, cache):
    pairs = (
        (result.unique_feature_values, cache["unique_values"]),
        (result.unique_feature_valid_mask, cache["unique_valid_mask"]),
        (result.unique_feature_support_count, cache["unique_support_count"]),
    )
    for current, stored in pairs:
        if current.dtype != stored.dtype or current.shape != stored.shape or not np.array_equal(current, stored, equal_nan=True):
            raise ValueError("Golden unique array mismatch")
    for view in result.view_values:
        for current, key in ((result.view_values[view], f"{view}_values"),
                             (result.view_valid_masks[view], f"{view}_valid_mask"),
                             (result.view_support_counts[view], f"{view}_support_count")):
            stored = cache[key]
            if current.dtype != stored.dtype or current.shape != stored.shape or not np.array_equal(current, stored, equal_nan=True):
                raise ValueError(f"Golden view mismatch: {key}")


def _aggregate_probability_qc(rows):
    aggregate = {"records": len(rows)}
    scalar_keys = ("value_9", "non9_lt_0", "non9_gt_1", "non9_between_1_9", "non9_gt_9",
                   "nonfinite", "value_9_stage_valid", "value_9_stage_invalid",
                   "non9_oor_stage_valid", "non9_oor_stage_invalid", "anomalous_rows")
    for key in scalar_keys: aggregate[key] = sum(row[key] for row in rows)
    for key in ("value_9_by_channel", "non9_oor_by_channel"):
        aggregate[key] = [sum(row[key][index] for row in rows) for index in range(5)]
    aggregate["records_with_non9_oor"] = sum((row["non9_lt_0"] + row["non9_gt_1"]) > 0 for row in rows)
    aggregate["source_length_match_records"] = sum(row["source_lengths_match"] for row in rows)
    aggregate["entropy_source_valid_records"] = sum(row["entropy_source_valid"] for row in rows)
    return aggregate


def _execution_provenance(source_path, manifest_sha, spec_sha, seed):
    return {
        "source_npz_sha256": sha256_file(source_path),
        "manifest_sha256": manifest_sha,
        "spec_sha256": spec_sha,
        "g2a_features_sha256": sha256_file(Path(g2a_features.__file__)),
        "g2a_io_sha256": sha256_file(Path(g2a_io.__file__)),
        "g2b0_batch_safety_sha256": sha256_file(Path(g2b0_batch_safety.__file__)),
        "g2b0_pilot_sha256": sha256_file(Path(__file__)),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "sample_seed": seed,
    }


def execute_bounded_pilot(input_root, golden_root, output_root, manifest_path, spec_path,
                          candidate_paths=None, expected_counts=None, seed=SEED):
    input_root = Path(input_root).resolve(); golden_root = Path(golden_root).resolve()
    output_root = Path(output_root).resolve(); manifest_path = Path(manifest_path); spec_path = Path(spec_path)
    if output_root.exists(): raise FileExistsError(output_root)
    if input_root == output_root or input_root in output_root.parents or output_root in input_root.parents:
        raise ValueError("Input and output roots must be separated")
    manifest = load_manifest(manifest_path); spec_sha = sha256_file(spec_path)
    if spec_sha != "f9f6ae6b7fd3126569914dad9c643c17ee9fa56354c460c694f46060cb174d9a":
        raise ValueError("Frozen Spec SHA256 mismatch")
    inventory_rows, inventory_summary = build_inventory(input_root, expected_counts)
    inventory_paths = [input_root / row["relative_path"] for row in inventory_rows]
    inventory_set = {str(path.resolve()) for path in inventory_paths}
    mapping = json.loads((golden_root / "private_sample_map.json").read_text())
    if len(mapping) != GOLDEN_RECORDS: raise ValueError("Golden mapping must contain 18 records")
    golden_sources = {str(Path(item["source"]).resolve()) for item in mapping}
    if not golden_sources.issubset(inventory_set): raise ValueError("Golden source outside inventory")
    budget = ReadBudget(load_npz_record, MAX_DISTINCT_READS)
    golden_hashes_before = {}; source_hashes_before = {}; i0006_qc = []; golden_exact = 0
    for item in mapping:
        source = Path(item["source"]).resolve(); cache_path = golden_root / item["cache"]
        golden_hashes_before[item["cache"]] = sha256_file(cache_path)
        source_hashes_before[str(source)] = sha256_file(source)
        cache = read_cache(cache_path)
        if str(cache["source_npz_sha256"].item()) != source_hashes_before[str(source)]:
            raise ValueError("Golden source SHA256 mismatch")
        if str(cache["manifest_sha256"].item()) != manifest.sha256 or str(cache["spec_sha256"].item()) != spec_sha:
            raise ValueError("Golden frozen provenance mismatch")
        record = budget.load(source); result = compute_night_features(record, manifest)
        _assert_result_matches_cache(result, cache); golden_exact += 1
        if item["site"] == "I0006": i0006_qc.append(probability_qc(record))
    if golden_exact != GOLDEN_RECORDS: raise ValueError("Golden replay incomplete")
    if candidate_paths is None:
        candidates = deterministic_candidates(inventory_paths, golden_sources, seed)
    else:
        candidates = [Path(path).resolve() for path in candidate_paths]
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) > MAX_DISTINCT_READS - GOLDEN_RECORDS: raise ValueError("Candidate budget exceeds 62")
    if any(str(path.resolve()) not in inventory_set for path in candidates): raise ValueError("Candidate outside inventory")
    if any(str(path.resolve()) in golden_sources for path in candidates): raise ValueError("Candidate overlaps Golden")
    selected, coverage = select_targeted_records(candidates, budget, MAX_NEW_PILOT)
    if len(selected) > MAX_NEW_PILOT or len(selected) + GOLDEN_RECORDS > MAX_TOTAL_RECORDS:
        raise ValueError("Pilot output cap exceeded")
    output_root.parent.mkdir(parents=True, exist_ok=True); output_root.mkdir()
    code_fingerprints = _execution_provenance(next(iter(map(Path, golden_sources))), manifest.sha256, spec_sha, seed)
    code_fingerprints.pop("source_npz_sha256")
    write_json_no_clobber(output_root / "run_identity.json", code_fingerprints)
    write_json_no_clobber(output_root / "inventory.json", {"summary": inventory_summary, "files": inventory_rows})
    write_json_no_clobber(output_root / "golden_sha256_before.json", golden_hashes_before)
    pilot_mapping = []; pilot_qc = []; resume_skips = 0; replay_exact = 0; source_verified = 0
    for number, (source, record, flags) in enumerate(selected):
        result = compute_night_features(record, manifest)
        for category in ("no_hrv", "annotation_missing", "race_unavailable", "probability_incomplete"):
            if flags[category]: assert_boundary_effects(category, result)
        provenance = _execution_provenance(source, manifest.sha256, spec_sha, seed)
        payload = build_pilot_payload(result, provenance)
        cache_name = f"pilot_{number:03d}.npz"; cache_path = output_root / cache_name
        validator = lambda arrays, p=provenance: assert_valid_cache_arrays(arrays, p, manifest)
        write_cache_atomic(cache_path, payload, payload_validator=validator)
        if decide_resume_action(cache_path, provenance, manifest) != Action.SKIP_VERIFIED:
            raise ValueError("Pilot Resume validation failed")
        resume_skips += 1
        replay = compute_night_features(budget.load(source), manifest)
        _assert_result_matches_cache(replay, read_cache(cache_path)); replay_exact += 1
        if sha256_file(source) != provenance["source_npz_sha256"]: raise ValueError("Pilot source changed")
        source_verified += 1
        pilot_mapping.append({"cache": cache_name, "source": str(source), "site": source.parent.name, "flags": flags})
        pilot_qc.append({"cache": cache_name, "site": source.parent.name, "qc": probability_qc(record)})
    golden_hashes_after = {item["cache"]: sha256_file(golden_root / item["cache"]) for item in mapping}
    if golden_hashes_after != golden_hashes_before: raise ValueError("Golden cache changed")
    for source, digest in source_hashes_before.items():
        if sha256_file(Path(source)) != digest: raise ValueError("Golden source changed")
    site_counts = {site: sum(item["site"] == site for item in pilot_mapping) for site in SITES}
    metadata = {
        "seed": seed, "golden_exact": golden_exact, "pilot_records": len(selected),
        "pilot_site_counts": site_counts, "coverage": coverage,
        "distinct_npz_reads": budget.distinct_count, "npz_read_calls": budget.read_calls,
        "inventory": inventory_summary, "resume_skips": resume_skips,
        "pilot_replay_exact": replay_exact, "pilot_source_sha_verified": source_verified,
        "i0006_golden_probability_qc": _aggregate_probability_qc(i0006_qc),
        "golden_unchanged": True, "source_unchanged": True,
        "manifest_sha256": manifest.sha256, "spec_sha256": spec_sha,
    }
    write_json_no_clobber(output_root / "pilot_mapping.json", pilot_mapping)
    write_json_no_clobber(output_root / "probability_qc_details.json", {"i0006_golden": i0006_qc, "pilot": pilot_qc})
    write_json_no_clobber(output_root / "run_metadata.json", metadata)
    write_json_no_clobber(output_root / "golden_sha256_after.json", golden_hashes_after)
    return metadata


def build_parser():
    parser = argparse.ArgumentParser(description="Bounded G2-B0 targeted Pilot only")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--golden-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--candidate-list", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main():
    args = build_parser().parse_args()
    candidates = None
    if args.candidate_list is not None:
        candidate_data = json.loads(args.candidate_list.read_text(encoding="utf-8"))
        if not isinstance(candidate_data, list) or not all(isinstance(item, str) for item in candidate_data):
            raise ValueError("Candidate list must be a JSON list of paths")
        candidates = [Path(item) for item in candidate_data]
    metadata = execute_bounded_pilot(
        args.input_root, args.golden_root, args.output_root, args.manifest, args.spec,
        candidate_paths=candidates, seed=args.seed,
        expected_counts={"I0002": 319, "I0006": 1142, "S0001": 5139},
    )
    public_summary = {
        "pilot_records": metadata["pilot_records"],
        "pilot_site_counts": metadata["pilot_site_counts"],
        "coverage": metadata["coverage"],
        "distinct_npz_reads": metadata["distinct_npz_reads"],
    }
    print(json.dumps(public_summary, sort_keys=True))


if __name__ == "__main__": main()
