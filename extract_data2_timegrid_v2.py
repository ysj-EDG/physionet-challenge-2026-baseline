#!/usr/bin/env python
"""Restartable, atomic data2 cache builder for timegrid_v2."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DEFAULT = Path("/database2/physionet2026_kaggle/data2")
OUT_DEFAULT = ROOT / "npz_data2"
VERSION = "timegrid_v2.0.0"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance():
    files = [
        ROOT / "channel_table.csv",
        ROOT / "per_epoch_features" / "per_epoch_extractor.py",
        ROOT / "per_epoch_features" / "timegrid_v2_extractor.py",
        ROOT / "per_epoch_features" / "feature_extractor_algorithmic.py",
        ROOT / "per_epoch_features" / "eeg_sleep_features.py",
    ]
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip()
    except Exception:
        git_sha = "unknown"
    try:
        import scipy, edfio
        dependencies = {"python": platform.python_version(),
                        "numpy": np.__version__, "scipy": scipy.__version__,
                        "pandas": pd.__version__, "edfio": edfio.__version__}
    except Exception as exc:
        dependencies = {"error": repr(exc)}
    return {"git_sha": git_sha,
            "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in files},
            "dependencies": dependencies}


def scalar_text(value):
    return np.asarray("" if value is None else str(value))


def cache_valid(path, expected_record):
    try:
        with np.load(path, allow_pickle=False) as z:
            return (str(z["extraction_version"].item()) == VERSION
                    and str(z["record_id"].item()) == expected_record
                    and z["X_seq"].ndim == 2 and z["X_seq"].shape[1] == 483
                    and z["X_ecg"].ndim == 2 and z["X_ecg"].shape[1] == 12
                    and z["x_static"].shape == (196,)
                    and z["mask"].shape == (len(z["X_seq"]),)
                    and int(z["y"].item()) in (0, 1)
                    and np.isfinite(z["X_seq"]).all()
                    and np.isfinite(z["X_ecg"]).all()
                    and np.isfinite(z["x_static"]).all())
    except Exception:
        return False


def npz_payload(core, metadata, prov):
    x_seq, x_ecg, x_static, y, mask = core
    payload = {
        "X_seq": x_seq, "X_ecg": x_ecg, "x_static": x_static,
        "y": np.asarray(y, dtype=np.int8), "mask": mask,
        "extraction_version": scalar_text(VERSION),
        "code_git_sha": scalar_text(prov["git_sha"]),
        "source_sha256_json": scalar_text(json.dumps(prov["source_sha256"], sort_keys=True)),
        "dependencies_json": scalar_text(json.dumps(prov["dependencies"], sort_keys=True)),
    }
    dict_keys = {"annotation_sampling_frequencies", "annotation_signal_lengths",
                 "physiological_signal_lengths"}
    for key, value in metadata.items():
        if key in dict_keys:
            payload[key + "_json"] = scalar_text(json.dumps(value, sort_keys=True))
        elif isinstance(value, (str, type(None))):
            payload[key] = scalar_text(value)
        else:
            payload[key] = np.asarray(value)
    return payload


def worker(row, data_root, output_root, prov):
    started = time.time()
    record_id = f"{row['BidsFolder']}_ses-{row['SessionID']}"
    target = Path(output_root) / str(row["SiteID"]) / f"{record_id}.npz"
    if target.exists() and cache_valid(target, record_id):
        return {"record_id": record_id, "site_id": str(row["SiteID"]),
                "status": "existing", "path": str(target),
                "elapsed_sec": time.time() - started}
    try:
        from per_epoch_features.timegrid_v2_extractor import TimegridV2Extractor
        extractor = TimegridV2Extractor()
        result = extractor.extract_all(row, data_root, return_metadata=True)
        core, metadata = result[:5], result[5]
        if core[0] is None:
            raise RuntimeError(metadata.get("physiological_status", "extraction_failed"))
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = npz_payload(core, metadata, prov)
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npz.tmp", prefix=record_id + ".",
            dir=target.parent, delete=False,
        ) as tmp:
            temp_path = Path(tmp.name)
            np.savez_compressed(tmp, **payload)
            tmp.flush()
            os.fsync(tmp.fileno())
        if not cache_valid(temp_path, record_id):
            raise RuntimeError("atomic cache validation failed")
        os.replace(temp_path, target)
        x_seq, x_ecg, _, y, _ = core
        valid = np.asarray(metadata["stage_valid"], bool)
        available = np.asarray(metadata["eeg_channel_available"], bool)
        clean = np.asarray(metadata["eeg_clean_subsegment_count"], float)
        total = np.asarray(metadata["eeg_total_subsegment_count"], float)
        ratio = float(clean.sum() / total.sum()) if total.sum() else np.nan
        return {"record_id": record_id, "site_id": str(row["SiteID"]),
                "label": int(y), "status": "success", "path": str(target),
                "n_seq": int(len(x_seq)), "n_ecg": int(len(x_ecg)),
                "stage_valid": int(valid.sum()), "stage_unknown": int((~valid).sum()),
                "eeg_channels": int(available.sum()), "eeg_clean_ratio": ratio,
                "annotation_status": metadata["annotation_status"],
                "elapsed_sec": time.time() - started}
    except Exception as exc:
        try:
            if "temp_path" in locals() and temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
        return {"record_id": record_id, "site_id": str(row.get("SiteID", "")),
                "status": "failed", "path": str(target),
                "error": repr(exc), "traceback": traceback.format_exc(),
                "elapsed_sec": time.time() - started}


def parse_label(value):
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}: return 1
    if text in {"false", "0", "no"}: return 0
    raise ValueError(f"non-binary official label: {value!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DATA_DEFAULT)
    ap.add_argument("--output-root", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--ids", type=Path)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--manifest", type=Path)
    args = ap.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest or args.output_root / "manifest.jsonl"
    frame = pd.read_csv(args.data_root / "demographics.csv", dtype={
        "BidsFolder": "string", "SessionID": "string", "BDSPPatientID": "string"})
    frame["_label"] = frame["Cognitive_Impairment"].map(parse_label)
    if args.ids:
        wanted = {x.strip() for x in args.ids.read_text().splitlines() if x.strip()}
        keys = frame["BidsFolder"].astype(str) + "_ses-" + frame["SessionID"].astype(str)
        frame = frame.loc[keys.isin(wanted)]
        missing = wanted - set(keys[keys.isin(wanted)])
        if missing: raise SystemExit(f"IDs absent from demographics: {sorted(missing)}")
    if args.limit is not None:
        frame = frame.iloc[:args.limit]
    rows = frame.drop(columns=["_label"]).to_dict("records")
    prov = provenance()
    launch = {"event": "launch", "version": VERSION, "time": time.time(),
              "data_root": str(args.data_root), "output_root": str(args.output_root),
              "workers": args.workers, "records": len(rows), **prov}
    with manifest.open("a", encoding="utf-8") as log:
        log.write(json.dumps(launch, sort_keys=True) + "\n"); log.flush()
        counts = {"success": 0, "existing": 0, "failed": 0}
        if args.workers == 1:
            iterator = (worker(r, str(args.data_root), str(args.output_root), prov) for r in rows)
            for result in iterator:
                counts[result["status"]] += 1
                log.write(json.dumps(result, sort_keys=True, allow_nan=True) + "\n"); log.flush()
                print(json.dumps(result, sort_keys=True, allow_nan=True), flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(worker, r, str(args.data_root), str(args.output_root), prov)
                           for r in rows]
                for future in as_completed(futures):
                    result = future.result()
                    counts[result["status"]] += 1
                    log.write(json.dumps(result, sort_keys=True, allow_nan=True) + "\n"); log.flush()
                    print(json.dumps(result, sort_keys=True, allow_nan=True), flush=True)
        summary = {"event": "summary", "time": time.time(), **counts}
        log.write(json.dumps(summary, sort_keys=True) + "\n")
    (args.output_root / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if counts["failed"]: raise SystemExit(1)

if __name__ == "__main__":
    main()
