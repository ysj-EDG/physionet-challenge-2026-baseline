"""Read-only CAISR stage EDF census and frozen-NPZ metadata cross-check."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

FILE_MISSING = "FILE_MISSING"
FILE_PRESENT_UNREADABLE = "FILE_PRESENT_UNREADABLE"
FILE_PRESENT_NO_STAGE = "FILE_PRESENT_NO_STAGE"
FILE_PRESENT_STAGE_READABLE = "FILE_PRESENT_STAGE_READABLE"
FILE_STATES = (
    FILE_MISSING, FILE_PRESENT_UNREADABLE,
    FILE_PRESENT_NO_STAGE, FILE_PRESENT_STAGE_READABLE,
)
SITES = ("I0002", "I0006", "S0001")
EXPECTED_COUNTS = {"I0002": 319, "I0006": 1142, "S0001": 5139}
VALID_CODES = np.asarray((1.0, 2.0, 3.0, 4.0, 5.0))
STAGE_NAMES = {1: "N3", 2: "N2", 3: "N1", 4: "REM", 5: "Wake"}
EXPECTED_STAGE_FREQUENCY = 1.0 / 30.0
EXPECTED_EXTRACTION_VERSION = "timegrid_v2.1.0"
EXPECTED_VALIDITY_VERSION = "validity_v0.2"
AUDIT_SCHEMA_VERSION = "g2b0r_caisr6600_v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_scalar(value):
    value = np.asarray(value)
    if value.shape != ():
        raise ValueError("Expected scalar metadata")
    item = value.item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return str(item)


def load_identities(path, expected_total=6600, expected_counts=EXPECTED_COUNTS):
    required = ("SiteID", "BidsFolder", "SessionID")
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(required).issubset(reader.fieldnames):
            raise ValueError("Demographics identity columns missing")
        for source in reader:
            row = {key: source[key].strip() for key in required}
            if not all(row.values()):
                raise ValueError("Blank identity field")
            rows.append(row)
    keys = [(row["BidsFolder"], row["SessionID"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate BidsFolder/SessionID identity")
    if len(rows) != expected_total:
        raise ValueError(f"Identity count mismatch: {len(rows)}")
    counts = Counter(row["SiteID"] for row in rows)
    if dict(counts) != dict(expected_counts):
        raise ValueError(f"Site counts mismatch: {dict(counts)}")
    return rows


def _contained(path, root):
    resolved = Path(path).resolve(strict=False)
    root = Path(root).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Constructed path escapes approved root") from exc
    return resolved


def exact_paths(identity, data_root, npz_root):
    if identity["SiteID"] not in SITES:
        raise ValueError("Unexpected site")
    for key in ("BidsFolder", "SessionID"):
        value = identity[key]
        if not value or Path(value).name != value or value in (".", ".."):
            raise ValueError(f"Unsafe identity component: {key}")
    record_id = f'{identity["BidsFolder"]}_ses-{identity["SessionID"]}'
    edf = Path(data_root) / "algorithmic_annotations" / identity["SiteID"] / f"{record_id}_caisr_annotations.edf"
    npz = Path(npz_root) / identity["SiteID"] / f"{record_id}.npz"
    return _contained(edf, Path(data_root) / "algorithmic_annotations"), _contained(npz, npz_root)


def _valid_mask(values):
    values = np.asarray(values, dtype=np.float64)
    return np.isfinite(values) & np.isin(values, VALID_CODES)


def _valid_bin(count):
    if count == 0: return "0"
    if count < 5: return "1-4"
    if count < 20: return "5-19"
    return ">=20"


def stage_summary(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = _valid_mask(values)
    finite = np.isfinite(values)
    valid_count = int(valid.sum())
    return {
        "length": int(values.size),
        "valid_count": valid_count,
        "invalid_count": int((~valid).sum()),
        "unknown_count": int((finite & ~np.isin(values, VALID_CODES)).sum()),
        "valid_bin": _valid_bin(valid_count),
        "stage_counts": {STAGE_NAMES[code]: int((values == code).sum()) for code in STAGE_NAMES},
    }


def align_stage(raw_stage, length):
    if int(length) < 0:
        raise ValueError("Negative target length")
    raw = np.asarray(raw_stage, dtype=np.float64).reshape(-1)
    aligned = np.full(int(length), np.nan, dtype=np.float64)
    copied = min(len(raw), len(aligned))
    aligned[:copied] = raw[:copied]
    return aligned


def audit_edf(path, reader=None):
    path = Path(path)
    base = {
        "file_status": FILE_MISSING, "read_error_type": None,
        "raw_stage": None, "stage_sampling_frequency": None,
        "sampling_frequency_warning": False,
    }
    if not path.is_file():
        return base
    if reader is None:
        import edfio
        reader = edfio.read_edf
    try:
        edf = reader(path, lazy_load_data=True)
        signals = [signal for signal in edf.signals if signal.label.lower().strip() == "stage_caisr"]
        if not signals:
            base["file_status"] = FILE_PRESENT_NO_STAGE
            return base
        if len(signals) != 1:
            raise ValueError("Multiple stage_caisr channels")
        signal = signals[0]
        raw = np.asarray(signal.data, dtype=np.float64).reshape(-1).copy()
        frequency = float(signal.sampling_frequency)
        summary = stage_summary(raw)
        base.update({
            "file_status": FILE_PRESENT_STAGE_READABLE,
            "raw_stage": raw,
            "stage_sampling_frequency": frequency,
            "sampling_frequency_warning": not np.isclose(
                frequency, EXPECTED_STAGE_FREQUENCY, rtol=0.0, atol=1e-12),
            "raw_stage_length": summary["length"],
            "raw_valid_count": summary["valid_count"],
            "raw_invalid_count": summary["invalid_count"],
            "raw_unknown_count": summary["unknown_count"],
            "raw_valid_bin": summary["valid_bin"],
            "stage_counts": summary["stage_counts"],
        })
        return base
    except Exception as exc:
        base["file_status"] = FILE_PRESENT_UNREADABLE
        base["read_error_type"] = type(exc).__name__
        return base


def compare_npz_stage(expected_annotation, expected_stage, expected_valid,
                      npz_annotation, npz_stage, npz_valid):
    expected_stage = np.asarray(expected_stage, dtype=np.float64)
    npz_stage = np.asarray(npz_stage, dtype=np.float64)
    expected_valid = np.asarray(expected_valid, dtype=bool)
    npz_valid = np.asarray(npz_valid, dtype=bool)
    stage_equal = (expected_stage.shape == npz_stage.shape
                   and np.array_equal(expected_stage, npz_stage, equal_nan=True))
    valid_equal = (expected_valid.shape == npz_valid.shape
                   and np.array_equal(expected_valid, npz_valid))
    return {
        "annotation_status_match": expected_annotation == npz_annotation,
        "stage_code_match": bool(stage_equal),
        "stage_valid_match": bool(valid_equal),
        "stage_valid_count_match": bool(valid_equal and int(expected_valid.sum()) == int(npz_valid.sum())),
    }


def read_npz_stage_metadata(path):
    allowed = (
        "extraction_version", "validity_schema_version", "annotation_status",
        "stage_code_aligned", "stage_valid",
    )
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in allowed if key not in archive.files]
        if missing:
            raise ValueError("NPZ metadata missing: " + ",".join(missing))
        extraction = _decode_scalar(archive["extraction_version"])
        validity = _decode_scalar(archive["validity_schema_version"])
        annotation = _decode_scalar(archive["annotation_status"])
        stage = np.asarray(archive["stage_code_aligned"], dtype=np.float64).copy()
        stage_valid = np.asarray(archive["stage_valid"], dtype=bool).copy()
    if extraction != EXPECTED_EXTRACTION_VERSION or validity != EXPECTED_VALIDITY_VERSION:
        raise ValueError(f"Frozen NPZ version mismatch: {extraction}/{validity}")
    if stage.ndim != 1 or stage_valid.shape != stage.shape:
        raise ValueError("NPZ stage metadata shape mismatch")
    return {
        "extraction_version": extraction, "validity_schema_version": validity,
        "annotation_status": annotation, "stage_code_aligned": stage,
        "stage_valid": stage_valid, "T": int(stage.size),
    }


def audit_record(identity, data_root, npz_root, reader=None):
    edf_path, npz_path = exact_paths(identity, data_root, npz_root)
    if not npz_path.is_file():
        raise ValueError("Mapped frozen NPZ missing")
    npz = read_npz_stage_metadata(npz_path)
    edf = audit_edf(edf_path, reader=reader)
    row = {
        "site": identity["SiteID"], "BidsFolder": identity["BidsFolder"],
        "SessionID": identity["SessionID"], "file_status": edf["file_status"],
        "edf_size": edf_path.stat().st_size if edf_path.is_file() else None,
        "edf_mtime_ns": edf_path.stat().st_mtime_ns if edf_path.is_file() else None,
        "npz_size": npz_path.stat().st_size,
        "npz_mtime_ns": npz_path.stat().st_mtime_ns,
        "T": npz["T"], "npz_annotation_status": npz["annotation_status"],
        "npz_stage_valid_count": int(npz["stage_valid"].sum()),
        "read_error_type": edf.get("read_error_type"),
        "stage_sampling_frequency": edf.get("stage_sampling_frequency"),
        "sampling_frequency_warning": edf.get("sampling_frequency_warning", False),
        "raw_stage_length": None, "raw_valid_count": None,
        "raw_invalid_count": None, "raw_unknown_count": None,
        "raw_valid_bin": None, "stage_counts": None,
        "aligned_valid_count": None, "aligned_valid_bin": None,
        "truncated_valid_count": None, "padding_without_stage": None,
        "raw_ge20_aligned_lt20": False, "raw_aligned_count_differs": False,
        "comparison_determined": False, "annotation_status_match": None,
        "stage_code_match": None, "stage_valid_match": None,
        "stage_valid_count_match": None,
    }
    status = edf["file_status"]
    if status == FILE_PRESENT_STAGE_READABLE:
        raw = edf.pop("raw_stage")
        aligned = align_stage(raw, npz["T"])
        expected_annotation = "available"
        row.update({key: edf[key] for key in (
            "raw_stage_length", "raw_valid_count", "raw_invalid_count",
            "raw_unknown_count", "raw_valid_bin", "stage_counts")})
    elif status == FILE_MISSING:
        raw = None; aligned = np.full(npz["T"], np.nan); expected_annotation = "missing"
    elif status == FILE_PRESENT_NO_STAGE:
        raw = None; aligned = np.full(npz["T"], np.nan); expected_annotation = "available"
    else:
        raw = None; aligned = None; expected_annotation = None
    if aligned is not None:
        aligned_summary = stage_summary(aligned)
        comparison = compare_npz_stage(
            expected_annotation, aligned, _valid_mask(aligned),
            npz["annotation_status"], npz["stage_code_aligned"], npz["stage_valid"])
        row.update(comparison)
        row["comparison_determined"] = True
        row["aligned_valid_count"] = aligned_summary["valid_count"]
        row["aligned_valid_bin"] = aligned_summary["valid_bin"]
        row["padding_without_stage"] = max(npz["T"] - (len(raw) if raw is not None else 0), 0)
        if raw is not None:
            row["truncated_valid_count"] = row["raw_valid_count"] - row["aligned_valid_count"]
            row["raw_ge20_aligned_lt20"] = row["raw_valid_count"] >= 20 and row["aligned_valid_count"] < 20
            row["raw_aligned_count_differs"] = row["raw_valid_count"] != row["aligned_valid_count"]
    return row


def _distribution(counts):
    counts = list(counts)
    result = {key: 0 for key in ("0", "1-4", "5-19", ">=20")}
    for count in counts: result[_valid_bin(int(count))] += 1
    result["lt20"] = result["0"] + result["1-4"] + result["5-19"]
    if counts:
        quantiles = np.percentile(np.asarray(counts), (5, 25, 50, 75, 95), method="linear")
        result["percentiles"] = {name: float(value) for name, value in zip(("P05","P25","Median","P75","P95"), quantiles)}
    else:
        result["percentiles"] = {name: None for name in ("P05","P25","Median","P75","P95")}
    return result


def aggregate_rows(rows, expected_counts=EXPECTED_COUNTS):
    groups = {site: [row for row in rows if row["site"] == site] for site in SITES}
    groups["Overall"] = list(rows)
    output = {}
    for site, selected in groups.items():
        expected = sum(expected_counts.values()) if site == "Overall" else expected_counts[site]
        if len(selected) != expected:
            raise ValueError(f"Aggregate count mismatch for {site}: {len(selected)}")
        states = Counter(row["file_status"] for row in selected)
        state_counts = {state: int(states.get(state, 0)) for state in FILE_STATES}
        if sum(state_counts.values()) != expected:
            raise ValueError(f"File-state closure failed for {site}")
        readable = [row for row in selected if row["file_status"] == FILE_PRESENT_STAGE_READABLE]
        determined = [row for row in selected if row.get("comparison_determined")]
        output[site] = {
            "total": expected,
            "file_status": state_counts,
            "file_present": expected - state_counts[FILE_MISSING],
            "raw": _distribution(row["raw_valid_count"] for row in readable),
            "aligned": _distribution(row["aligned_valid_count"] for row in readable),
            "raw_ge20_aligned_lt20": sum(row.get("raw_ge20_aligned_lt20", False) for row in readable),
            "raw_aligned_count_differs": sum(row.get("raw_aligned_count_differs", False) for row in readable),
            "stage_length_differs_from_T": sum(row.get("raw_stage_length") != row.get("T") for row in readable),
            "sampling_frequency_warning": sum(row.get("sampling_frequency_warning", False) for row in readable),
            "comparison_determined": len(determined),
            "annotation_status_match": sum(row.get("annotation_status_match") is True for row in determined),
            "stage_code_match": sum(row.get("stage_code_match") is True for row in determined),
            "stage_valid_match": sum(row.get("stage_valid_match") is True for row in determined),
            "stage_valid_count_match": sum(row.get("stage_valid_count_match") is True for row in determined),
            "unknown_stage_records": sum((row.get("raw_unknown_count") or 0) > 0 for row in readable),
        }
    return output


def _json_default(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    raise TypeError(type(value).__name__)


def publish_chunk(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), default=_json_default)
            handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if temporary is not None and temporary.exists(): temporary.unlink()


def read_verified_chunk(path, run_identity):
    try:
        with Path(path).open(encoding="utf-8") as handle: payload = json.load(handle)
    except Exception as exc:
        raise ValueError(f"Invalid checkpoint: {type(exc).__name__}") from exc
    if payload.get("run_identity") != run_identity or not isinstance(payload.get("rows"), list):
        raise ValueError("Checkpoint identity or rows mismatch")
    return payload


def _run_identity(code_path, demographics, data_root, npz_root):
    source = {
        "schema": AUDIT_SCHEMA_VERSION,
        "code_sha256": sha256_file(code_path),
        "demographics_sha256": sha256_file(demographics),
        "data_root": str(Path(data_root).resolve()),
        "npz_root": str(Path(npz_root).resolve()),
        "extraction_version": EXPECTED_EXTRACTION_VERSION,
        "validity_schema_version": EXPECTED_VALIDITY_VERSION,
    }
    encoded = json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
    source["run_identity"] = hashlib.sha256(encoded).hexdigest()
    return source


def run_full_audit(data_root, demographics, npz_root, output_root, chunk_size=100):
    data_root = Path(data_root).resolve(); demographics = Path(demographics).resolve()
    npz_root = Path(npz_root).resolve(); output_root = Path(output_root).resolve()
    for required in (data_root, demographics, data_root / "algorithmic_annotations", npz_root):
        if not required.exists(): raise ValueError(f"Required input missing: {required}")
    for site in SITES:
        if not (data_root / "algorithmic_annotations" / site).is_dir():
            raise ValueError(f"Annotation site directory missing: {site}")
    if output_root == data_root or output_root == npz_root or output_root.is_relative_to(data_root) or output_root.is_relative_to(npz_root):
        raise ValueError("Private output overlaps source data")
    identities = sorted(load_identities(demographics), key=lambda row: (row["SiteID"], row["BidsFolder"], row["SessionID"]))
    for identity in identities:
        _edf, npz = exact_paths(identity, data_root, npz_root)
        if not npz.is_file(): raise ValueError("Demographics/NPZ one-to-one mapping failed")
    provenance = _run_identity(Path(__file__), demographics, data_root, npz_root)
    if output_root.exists():
        identity_file = output_root / "run_identity.json"
        if not identity_file.is_file(): raise ValueError("Existing private root lacks run identity")
        existing = read_verified_chunk(identity_file, provenance["run_identity"])
        stored_provenance = {key: existing.get(key) for key in provenance}
        if stored_provenance != provenance: raise ValueError("Existing private root provenance mismatch")
    else:
        output_root.mkdir(parents=True)
        publish_chunk(output_root / "run_identity.json", {**provenance, "rows": []})
        read_verified_chunk(output_root / "run_identity.json", provenance["run_identity"])
    chunks = output_root / "chunks"; chunks.mkdir(exist_ok=True)
    rows = []
    for start in range(0, len(identities), int(chunk_size)):
        chunk_path = chunks / f"chunk_{start // int(chunk_size):04d}.json"
        if chunk_path.exists():
            payload = read_verified_chunk(chunk_path, provenance["run_identity"])
        else:
            chunk_rows = [audit_record(identity, data_root, npz_root) for identity in identities[start:start + int(chunk_size)]]
            payload = {"run_identity": provenance["run_identity"], "start": start, "rows": chunk_rows}
            publish_chunk(chunk_path, payload)
            payload = read_verified_chunk(chunk_path, provenance["run_identity"])
        if payload.get("start") != start or len(payload["rows"]) != len(identities[start:start + int(chunk_size)]):
            raise ValueError("Checkpoint range mismatch")
        rows.extend(payload["rows"])
        print(f"AUDIT_PROGRESS {len(rows)}/{len(identities)}", flush=True)
    summary = aggregate_rows(rows)
    final = output_root / "summary.json"
    payload = {"run_identity": provenance["run_identity"], "rows": rows, "aggregate": summary}
    if final.exists():
        verified = read_verified_chunk(final, provenance["run_identity"])
        if verified != payload: raise ValueError("Existing final result mismatch")
    else:
        publish_chunk(final, payload)
    return summary, provenance, rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--demographics", required=True)
    parser.add_argument("--npz-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--chunk-size", type=int, default=100)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary, provenance, _rows = run_full_audit(
        args.data_root, args.demographics, args.npz_root, args.output_root, args.chunk_size)
    print(json.dumps({"run_identity": provenance["run_identity"], "aggregate": summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
