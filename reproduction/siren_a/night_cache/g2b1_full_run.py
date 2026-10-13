"""Deterministic, resumable G2-B1 full-cohort cache runner."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import g2a_features
import g2a_io
import g2b0_batch_safety
import g2b1_cache_contract
from g2a_features import compute_night_features
from g2a_io import load_manifest, load_npz_record, read_cache, sha256_file, write_cache_atomic
from g2b1_cache_contract import (
    Action, CACHE_SCHEMA_VERSION, assert_valid_cache_arrays, decide_resume_action,
    validate_existing_cache,
)

SITES = ("I0002", "I0006", "S0001")
EXPECTED_COUNTS = {"I0002": 319, "I0006": 1142, "S0001": 5139}
EXPECTED_MANIFEST_SHA256 = "7aef4e0462936ad0fbb3ee504a71062b0f50d82de5c54e1d70df2d13691e455a"
EXPECTED_SPEC_SHA256 = "f9f6ae6b7fd3126569914dad9c643c17ee9fa56354c460c694f46060cb174d9a"
EXPECTED_AUDIT = (70, 6, 76)


def cache_name(index):
    return f"cache_{int(index):06d}.npz"


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _json_fingerprint(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _resolved(path):
    return Path(path).resolve(strict=False)


def _overlaps(left, right):
    left, right = _resolved(left), _resolved(right)
    return left == right or left in right.parents or right in left.parents


def build_inventory(root, expected_counts=EXPECTED_COUNTS):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("Input NPZ root is unavailable")
    rows, keys = [], set()
    counts = {site: 0 for site in SITES}
    for site in SITES:
        directory = root / site
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"Invalid site directory: {site}")
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if path.is_symlink():
                raise ValueError("Inventory contains symlink")
            if not path.is_file() or path.suffix != ".npz":
                raise ValueError("Inventory contains unexpected entry")
            relative = path.resolve().relative_to(root)
            if len(relative.parts) != 2 or relative.parts[0] != site:
                raise ValueError("Inventory path escapes expected layout")
            key = f"{site}/{path.stem}"
            if key in keys:
                raise ValueError("Duplicate record key")
            keys.add(key)
            stat = path.stat()
            rows.append({"relative_path": relative.as_posix(), "site": site,
                         "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
            counts[site] += 1
    if counts != expected_counts:
        raise ValueError(f"Inventory counts mismatch: {counts}")
    for index, row in enumerate(rows):
        row["cache_name"] = cache_name(index)
    summary = {"total": len(rows), "site_counts": counts, "duplicates": 0,
               "symlinks": 0, "fingerprint": _json_fingerprint(rows)}
    return rows, summary


def _audit_key(row):
    if row.get("record_key"):
        return str(row["record_key"])
    bids, session = str(row.get("BidsFolder", "")), str(row.get("SessionID", ""))
    if not bids or not session:
        raise ValueError("CAISR audit row lacks an identity key")
    return f"{bids}_ses-{session}"


def load_caisr_audit(audit_root, inventory_rows, expected_facts=EXPECTED_AUDIT):
    summary_path = Path(audit_root) / "summary.json"
    if not summary_path.is_file():
        raise ValueError("CAISR private audit summary is missing")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("CAISR audit rows missing")
    expected = {row["relative_path"] for row in inventory_rows}
    by_relative = {}
    for row in rows:
        site = str(row.get("site", row.get("SiteID", "")))
        relative = f"{site}/{_audit_key(row)}.npz"
        if relative in by_relative:
            raise ValueError("Duplicate CAISR audit identity")
        if relative not in expected:
            raise ValueError("Unknown CAISR audit identity")
        by_relative[relative] = row
    if set(by_relative) != expected:
        raise ValueError("CAISR audit does not close over inventory")
    missing = sum(row.get("file_status") == "FILE_MISSING" for row in rows)
    readable_lt20 = sum(row.get("file_status") == "FILE_PRESENT_STAGE_READABLE"
                        and int(row.get("aligned_valid_count", -1)) < 20 for row in rows)
    source_lt20 = sum(int(row.get("npz_stage_valid_count", -1)) < 20 for row in rows)
    facts = (missing, readable_lt20, source_lt20)
    if facts != tuple(expected_facts):
        raise ValueError(f"CAISR audit facts mismatch: {facts}")
    return {"by_relative_path": by_relative, "annotation_missing": missing,
            "readable_lt20": readable_lt20, "stage_source_lt20": source_lt20,
            "summary_sha256": sha256_file(summary_path)}


def _code_hashes():
    return {
        "g2a_features_sha256": sha256_file(Path(g2a_features.__file__)),
        "g2a_io_sha256": sha256_file(Path(g2a_io.__file__)),
        "g2b0_batch_safety_sha256": sha256_file(Path(g2b0_batch_safety.__file__)),
        "g2b1_cache_contract_sha256": sha256_file(Path(g2b1_cache_contract.__file__)),
        "g2b1_full_run_sha256": sha256_file(Path(__file__)),
    }


def preflight(input_root, output_root, manifest_path, spec_path, audit_root,
              expected_counts=EXPECTED_COUNTS, expected_audit=EXPECTED_AUDIT,
              estimated_cache_bytes=None, disk_free_bytes=None):
    input_root, output_root, audit_root = map(Path, (input_root, output_root, audit_root))
    if _overlaps(input_root, output_root) or _overlaps(audit_root, output_root):
        raise ValueError("Input/audit and output roots must be separated")
    manifest = load_manifest(manifest_path, EXPECTED_MANIFEST_SHA256)
    spec_sha = sha256_file(Path(spec_path))
    if spec_sha != EXPECTED_SPEC_SHA256:
        raise ValueError("Frozen Spec SHA256 mismatch")
    rows, inventory = build_inventory(input_root, expected_counts)
    caisr = load_caisr_audit(audit_root, rows, expected_audit)
    estimate = int(estimated_cache_bytes if estimated_cache_bytes is not None else inventory["total"] * 25000)
    existing = output_root.parent
    while not existing.exists():
        existing = existing.parent
    free = int(disk_free_bytes if disk_free_bytes is not None else shutil.disk_usage(existing).free)
    required = max(estimate * 3, 100 * 1024 * 1024)
    if free < required:
        raise ValueError(f"Insufficient disk safety margin: free={free}, required={required}")
    code = _code_hashes()
    run_identity = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "input_root": str(input_root.resolve()),
        "inventory_fingerprint": inventory["fingerprint"],
        "manifest_sha256": manifest.sha256,
        "spec_sha256": spec_sha,
        "caisr_audit_summary_sha256": caisr["summary_sha256"],
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        **code,
    }
    return {"input_root": input_root.resolve(), "output_root": output_root.resolve(strict=False),
            "manifest": manifest, "manifest_path": Path(manifest_path),
            "spec_path": Path(spec_path), "inventory_rows": rows, "inventory": inventory,
            "caisr_audit": caisr["by_relative_path"],
            "caisr": {key: caisr[key] for key in ("annotation_missing", "readable_lt20", "stage_source_lt20")},
            "run_identity": run_identity, "estimated_cache_bytes": estimate,
            "disk_free_bytes": free, "disk_required_bytes": required}


def _atomic_json_no_clobber(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None; linked = False
    try:
        with tempfile.NamedTemporaryFile("wb", prefix=path.name + ".", suffix=".tmp",
                                         dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name); handle.write(_json_bytes(payload)); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path); linked = True
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("JSON publication round-trip failed")
        temporary.unlink(); temporary = None
    except Exception:
        if linked and path.exists(): path.unlink()
        raise
    finally:
        if temporary is not None and temporary.exists(): temporary.unlink()


def _write_or_match_json(path, payload):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"Run identity mismatch: {path.name}")
    else:
        _atomic_json_no_clobber(path, payload)


def _cache_provenance(source_sha, run_identity):
    keys = ("manifest_sha256", "spec_sha256", "g2a_features_sha256", "g2a_io_sha256",
            "g2b0_batch_safety_sha256", "g2b1_cache_contract_sha256",
            "g2b1_full_run_sha256", "python_version", "numpy_version", "cache_schema_version")
    return {"source_npz_sha256": source_sha, **{key: run_identity[key] for key in keys}}


def _cache_payload(result, provenance):
    payload = {"unique_feature_ids": np.asarray(result.unique_ids),
               "unique_values": np.asarray(result.unique_feature_values, dtype=np.float64),
               "unique_valid_mask": np.asarray(result.unique_feature_valid_mask, dtype=bool),
               "unique_support_count": np.asarray(result.unique_feature_support_count, dtype=np.int64)}
    for view in result.view_values:
        payload[f"{view}_values"] = np.asarray(result.view_values[view], dtype=np.float64)
        payload[f"{view}_valid_mask"] = np.asarray(result.view_valid_masks[view], dtype=bool)
        payload[f"{view}_support_count"] = np.asarray(result.view_support_counts[view], dtype=np.int64)
    payload.update({key: np.asarray(value) for key, value in provenance.items()})
    return payload


def _initialise_run(preflight_result, output_root, mode):
    output_root = Path(output_root)
    if mode == "build":
        if output_root.exists(): raise FileExistsError("G2-B1 target already exists; use resume after review")
        output_root.mkdir(parents=True); (output_root / "cache").mkdir(); (output_root / "checkpoints").mkdir()
    elif mode == "resume":
        if not output_root.is_dir(): raise FileNotFoundError("Resume target does not exist")
    else: raise ValueError("mode must be build or resume")
    identity = preflight_result["run_identity"]
    inventory_doc = {"rows": preflight_result["inventory_rows"], "summary": preflight_result["inventory"]}
    private_index = [{"cache_name": row["cache_name"], "relative_path": row["relative_path"], "site": row["site"]}
                     for row in preflight_result["inventory_rows"]]
    _write_or_match_json(output_root / "run_identity.json", identity)
    _write_or_match_json(output_root / "inventory.json", inventory_doc)
    _write_or_match_json(output_root / "private_index.json", private_index)
    (output_root / "cache").mkdir(exist_ok=True); (output_root / "checkpoints").mkdir(exist_ok=True)


def _check_audit_semantics(row, audit, record, result, manifest):
    stage_count = int(np.asarray(record["stage_valid"], bool).sum())
    if stage_count != int(audit["npz_stage_valid_count"]):
        raise ValueError("CAISR audit/NPZ stage count mismatch")
    family = {}
    for item in manifest.rows: family.setdefault(item["unique_feature_id"], item["family"])
    if audit["file_status"] == "FILE_MISSING":
        if record["annotation_status"] != "missing" or stage_count != 0:
            raise ValueError("CAISR missing-source semantics mismatch")
        forbidden = {"EEG54_stage_spectral", "spatial_asymmetry_candidate",
                     "spatial_ap_gradient_candidate", "CAISR20"}
        positions = [i for i, uid in enumerate(manifest.unique_ids) if family[uid] in forbidden]
        if result.unique_feature_valid_mask[positions].any():
            raise ValueError("Stage-dependent feature valid for missing annotation")
    elif record["annotation_status"] != "available":
        raise ValueError("Readable CAISR source not marked available")


def build_or_resume(preflight_result, output_root, mode="build", stop_after=None):
    output_root = Path(output_root); _initialise_run(preflight_result, output_root, mode)
    manifest = preflight_result["manifest"]; started = time.perf_counter()
    computed = skipped = processed = 0; source_verified = 0
    for index, row in enumerate(preflight_result["inventory_rows"]):
        source = preflight_result["input_root"] / row["relative_path"]
        source_sha = sha256_file(source); source_verified += 1
        provenance = _cache_provenance(source_sha, preflight_result["run_identity"])
        target = output_root / "cache" / row["cache_name"]
        action = decide_resume_action(target, provenance, manifest)
        if action == Action.STOP_INVALID:
            details = validate_existing_cache(target, provenance, manifest)
            raise ValueError("STOP_INVALID: " + "; ".join(details.errors))
        if action == Action.SKIP_VERIFIED:
            skipped += 1; processed += 1
        else:
            record = load_npz_record(source)
            result = compute_night_features(record, manifest)
            _check_audit_semantics(row, preflight_result["caisr_audit"][row["relative_path"]],
                                   record, result, manifest)
            payload = _cache_payload(result, provenance)
            write_cache_atomic(target, payload,
                               lambda arrays, p=provenance: assert_valid_cache_arrays(arrays, p, manifest))
            if decide_resume_action(target, provenance, manifest) != Action.SKIP_VERIFIED:
                raise ValueError("Published cache failed Resume validation")
            computed += 1; processed += 1
            if stop_after is not None and computed >= stop_after:
                break
        if processed and processed % 100 == 0:
            checkpoint = {"processed": processed, "computed": computed, "skipped": skipped,
                          "inventory_fingerprint": preflight_result["inventory"]["fingerprint"]}
            path = output_root / "checkpoints" / f"checkpoint_{processed:06d}.json"
            if not path.exists(): _atomic_json_no_clobber(path, checkpoint)
    complete = processed == len(preflight_result["inventory_rows"])
    summary = {"computed": computed, "skipped": skipped, "processed": processed,
               "source_sha_verified": source_verified, "complete": complete,
               "elapsed_seconds": time.perf_counter() - started}
    if complete:
        completion = {"complete": True, "records": len(preflight_result["inventory_rows"]),
                      "inventory_fingerprint": preflight_result["inventory"]["fingerprint"]}
        _write_or_match_json(output_root / "build_summary.json", completion)
    return summary


def _validate_run_files(preflight_result, output_root):
    output_root = Path(output_root)
    expected = {
        "run_identity.json": preflight_result["run_identity"],
        "inventory.json": {"rows": preflight_result["inventory_rows"], "summary": preflight_result["inventory"]},
        "private_index.json": [{"cache_name": row["cache_name"], "relative_path": row["relative_path"], "site": row["site"]}
                               for row in preflight_result["inventory_rows"]],
    }
    for name, payload in expected.items():
        path = output_root / name
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"Run identity mismatch: {name}")


def verify_full_cache(preflight_result, output_root):
    output_root = Path(output_root); _validate_run_files(preflight_result, output_root)
    expected_names = {row["cache_name"] for row in preflight_result["inventory_rows"]}
    actual = {path.name for path in (output_root / "cache").iterdir() if path.is_file()}
    if actual != expected_names: raise ValueError("Cache file set mismatch")
    verified = missing_sources = short_sources = 0
    for row in preflight_result["inventory_rows"]:
        source = preflight_result["input_root"] / row["relative_path"]
        provenance = _cache_provenance(sha256_file(source), preflight_result["run_identity"])
        result = validate_existing_cache(output_root / "cache" / row["cache_name"], provenance,
                                         preflight_result["manifest"])
        if not result.valid: raise ValueError("STOP_INVALID: " + "; ".join(result.errors))
        audit = preflight_result["caisr_audit"][row["relative_path"]]
        missing_sources += int(audit["file_status"] == "FILE_MISSING")
        short_sources += int(audit["file_status"] == "FILE_PRESENT_STAGE_READABLE"
                             and int(audit["aligned_valid_count"]) < 20)
        verified += 1
    if (missing_sources, short_sources, missing_sources + short_sources) != EXPECTED_AUDIT and verified == 6600:
        raise ValueError("CAISR 70/6/76 closure failed")
    return {"verified": verified, "unique_positions": verified * 807,
            "view_positions": verified * 827, "annotation_missing": missing_sources,
            "readable_stage_lt20": short_sources, "stage_source_lt20": missing_sources + short_sources}


def _percentiles(values):
    if not values: return [0.0] * 5
    return [float(x) for x in np.percentile(np.asarray(values), [5, 25, 50, 75, 95], method="linear")]


def aggregate_coverage(preflight_result, output_root):
    manifest = preflight_result["manifest"]
    first = {}
    for row in manifest.rows: first.setdefault(row["unique_feature_id"], row)
    stats = defaultdict(lambda: {"records": 0, "valid": 0, "missing": 0, "zeros": 0, "support": []})
    family_records = defaultdict(list)
    entropy_uid = "caisr_sleep_mean_stage_entropy"
    entropy_index = manifest.unique_ids.index(entropy_uid)
    entropy = Counter()
    for row in preflight_result["inventory_rows"]:
        arrays = read_cache(Path(output_root) / "cache" / row["cache_name"])
        values, masks, support = arrays["unique_values"], arrays["unique_valid_mask"], arrays["unique_support_count"]
        for scope in (row["site"], "Overall"):
            by_family = defaultdict(list)
            for index, uid in enumerate(manifest.unique_ids):
                item = stats[(scope, uid)]; item["records"] += 1; item["valid"] += int(masks[index])
                item["missing"] += int(not masks[index]); item["zeros"] += int(masks[index] and values[index] == 0)
                item["support"].append(int(support[index])); by_family[first[uid]["family"]].append(bool(masks[index]))
            for family, flags in by_family.items(): family_records[(scope, family)].append(float(np.mean(flags)))
        entropy[row["site"]] += int(masks[entropy_index]); entropy["Overall"] += int(masks[entropy_index])
    public = []
    for scope in (*SITES, "Overall"):
        for uid in manifest.unique_ids:
            item = stats[(scope, uid)]; p05, p25, median, p75, p95 = _percentiles(item["support"])
            public.append({"site": scope, "unique_feature_id": uid, "family": first[uid]["family"],
                           "records": item["records"], "valid_count": item["valid"],
                           "missing_count": item["missing"], "valid_rate": item["valid"] / item["records"],
                           "support_p05": p05, "support_p25": p25, "support_median": median,
                           "support_p75": p75, "support_p95": p95,
                           "lawful_zero_count": item["zeros"],
                           "lawful_zero_rate": item["zeros"] / item["valid"] if item["valid"] else 0.0})
    family_summary = []
    for (scope, family), fractions in sorted(family_records.items()):
        family_uids = [uid for uid in manifest.unique_ids if first[uid]["family"] == family]
        valid = sum(stats[(scope, uid)]["valid"] for uid in family_uids)
        total = sum(stats[(scope, uid)]["records"] for uid in family_uids)
        family_summary.append({"site": scope, "family": family, "columns": len(family_uids),
                               "valid": valid, "missing": total - valid,
                               "valid_rate": valid / total, "record_valid_fraction": _percentiles(fractions),
                               "fully_missing_columns": sum(stats[(scope, uid)]["valid"] == 0 for uid in family_uids)})
    return {"public_rows": public, "family_summary": family_summary,
            "records": len(preflight_result["inventory_rows"]), "entropy_valid": dict(entropy)}


def validate_public_coverage_rows(rows):
    allowed = {"site", "unique_feature_id", "family", "records", "valid_count", "missing_count",
               "valid_rate", "support_p05", "support_p25", "support_median", "support_p75", "support_p95",
               "lawful_zero_count", "lawful_zero_rate"}
    for row in rows:
        if set(row) != allowed: raise ValueError("Public coverage schema mismatch")
        text = json.dumps(row)
        if any(token in text for token in ("/database/", ".npz", "BidsFolder", "SessionID", "record_id")):
            raise ValueError("Private identity in public coverage")
    return True


def write_coverage_csv(path, rows):
    validate_public_coverage_rows(rows); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def verify_reference_caches(root, mapping_filename, manifest):
    root = Path(root); mapping = json.loads((root / mapping_filename).read_text(encoding="utf-8"))
    before = {item["cache"]: sha256_file(root / item["cache"]) for item in mapping}; matched = 0
    for item in mapping:
        result = compute_night_features(load_npz_record(Path(item["source"])), manifest)
        cached = read_cache(root / item["cache"])
        checks = [np.array_equal(result.unique_feature_values, cached["unique_values"], equal_nan=True),
                  np.array_equal(result.unique_feature_valid_mask, cached["unique_valid_mask"]),
                  np.array_equal(result.unique_feature_support_count, cached["unique_support_count"])]
        for view in manifest.view_sizes:
            checks.extend((np.array_equal(result.view_values[view], cached[f"{view}_values"], equal_nan=True),
                           np.array_equal(result.view_valid_masks[view], cached[f"{view}_valid_mask"]),
                           np.array_equal(result.view_support_counts[view], cached[f"{view}_support_count"])))
        if not all(checks): raise ValueError(f"Reference cache mismatch: {item['cache']}")
        matched += 1
    after = {item["cache"]: sha256_file(root / item["cache"]) for item in mapping}
    if before != after: raise ValueError("Reference cache was modified")
    return {"matched": matched, "file_sha256": before}


def build_parser():
    parser = argparse.ArgumentParser(description="Verified G2-B1 full-cache runner")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--build", action="store_true")
    modes.add_argument("--verify-only", action="store_true")
    modes.add_argument("--resume", action="store_true")
    parser.add_argument("--input-root", type=Path, required=True); parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True); parser.add_argument("--golden-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True); parser.add_argument("--coverage-csv", type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    pilot_sizes = [path.stat().st_size for path in args.pilot_root.glob("*.npz")]
    if not pilot_sizes: raise ValueError("Accepted Pilot cache unavailable")
    estimate = int(np.median(pilot_sizes) * 6600)
    pf = preflight(args.input_root, args.output_root, args.manifest, args.spec, args.audit_root,
                   estimated_cache_bytes=estimate)
    references = {"golden": verify_reference_caches(args.golden_root, "private_sample_map.json", pf["manifest"]),
                  "pilot": verify_reference_caches(args.pilot_root, "pilot_mapping.json", pf["manifest"])}
    if references["golden"]["matched"] != 18 or references["pilot"]["matched"] != 2:
        raise ValueError("Golden/Pilot count mismatch")
    if args.preflight_only:
        result = {"preflight": pf["inventory"], "caisr": pf["caisr"], "estimate": estimate,
                  "disk_free": pf["disk_free_bytes"], "references": {k: v["matched"] for k, v in references.items()}}
    elif args.build:
        result = build_or_resume(pf, args.output_root, "build")
    elif args.resume:
        result = build_or_resume(pf, args.output_root, "resume")
    else:
        result = verify_full_cache(pf, args.output_root)
        coverage = aggregate_coverage(pf, args.output_root)
        if args.coverage_csv: write_coverage_csv(args.coverage_csv, coverage["public_rows"])
        result["coverage"] = {"rows": len(coverage["public_rows"]), "entropy_valid": coverage["entropy_valid"],
                              "family_summary": coverage["family_summary"]}
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__": main()
