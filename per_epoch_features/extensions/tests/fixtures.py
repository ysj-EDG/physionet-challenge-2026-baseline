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
    stage_pattern = np.asarray([2.0, 2.0, 4.0, 5.0, 1.0, 3.0], dtype=np.float32)
    stages = np.resize(stage_pattern, epochs)
    return {
        "site_id": np.asarray(site_id),
        "record_id": np.asarray(record_id),
        "extraction_version": np.asarray("timegrid_v2.1.0"),
        "validity_schema_version": np.asarray("validity_v0.2"),
        "X_seq": np.arange(epochs * 483, dtype=np.float32).reshape(epochs, 483),
        "stage_valid": np.ones(epochs, dtype=bool),
        "stage_code_aligned": stages,
        "resp_feature_valid": np.ones((epochs, 14), dtype=bool),
        "epoch_start_sec": np.arange(epochs, dtype=np.float32) * 30.0,
        "epoch_duration_sec": np.full(epochs, 30.0, dtype=np.float32),
        "eeg_channel_available": np.ones(6, dtype=bool),
        "eeg_clean_subsegment_count": np.ones((epochs, 6), dtype=np.int32),
        "eeg_common_clean_subsegment_count": np.ones(epochs, dtype=np.int32),
        "emg_channel_available": np.ones(3, dtype=bool),
        "emg_preprocessing_success": np.ones(3, dtype=bool),
        "emg_epoch_success": np.ones((epochs, 3), dtype=bool),
        "x_static": np.zeros(196, dtype=np.float32),
        "annotation_status": np.asarray("available"),
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


def write_recipe_tree(root: Path) -> Path:
    code_root = Path(root)
    package = code_root / "pkg"
    package.mkdir(parents=True)
    (package / "family.py").write_text(
        "from pkg.common import SCALE, aggregate\n"
        "def extract_family(values):\n"
        "    return aggregate(values) * SCALE\n"
        "def unrelated(values):\n"
        "    return values\n",
        encoding="utf-8",
    )
    (package / "source.py").write_text(
        "def materialize(source):\n"
        "    return source\n",
        encoding="utf-8",
    )
    (package / "common.py").write_text(
        "SCALE = 2.0\n"
        "def stage_valid(values):\n"
        "    return values > 0\n"
        "def adjacency(starts):\n"
        "    return starts[1:] - starts[:-1] == 30\n"
        "def aggregate(values):\n"
        "    return sum(values) / len(values)\n"
        "def hidden(values):\n"
        "    return values\n",
        encoding="utf-8",
    )
    (package / "other_family.py").write_text(
        "def extract_family(values):\n"
        "    return -values\n",
        encoding="utf-8",
    )
    (code_root / "cache.py").write_text("CACHE_SCHEMA = 1\n", encoding="utf-8")
    (code_root / "record_key.py").write_text(
        "def record_key(value):\n    return value\n", encoding="utf-8"
    )
    (code_root / "README.md").write_text("synthetic recipe tree\n", encoding="utf-8")
    (code_root / "test_family.py").write_text("TEST_ONLY = True\n", encoding="utf-8")
    return code_root
