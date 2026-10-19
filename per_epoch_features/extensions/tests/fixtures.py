from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def source_payload(
    *,
    site_id: str = "SITE_A",
    record_id: str = "record-001",
    epochs: int = 4,
) -> dict[str, Any]:
    return {
        "site_id": np.asarray(site_id),
        "record_id": np.asarray(record_id),
        "extraction_version": np.asarray("timegrid_v2.1.0"),
        "validity_schema_version": np.asarray("validity_v0.2"),
        "X_seq": np.arange(epochs * 483, dtype=np.float32).reshape(epochs, 483),
        "stage_valid": np.asarray([True, False, True, True][:epochs], dtype=bool),
        "stage_code_aligned": np.asarray([2.0, 2.0, 4.0, 5.0][:epochs], dtype=np.float32),
        "resp_feature_valid": np.ones((epochs, 14), dtype=bool),
        "epoch_start_sec": np.arange(epochs, dtype=np.float32) * 30.0,
        "epoch_duration_sec": np.full(epochs, 30.0, dtype=np.float32),
        # Object dtype makes accidental label access fail under allow_pickle=False.
        "y": np.asarray({"forbidden": "label"}, dtype=object),
    }


def write_source_npz(
    root: Path,
    *,
    site_id: str = "SITE_A",
    record_id: str = "record-001",
    payload: dict[str, Any] | None = None,
) -> Path:
    path = Path(root) / site_id / f"{record_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **(payload or source_payload(site_id=site_id, record_id=record_id)))
    return path
