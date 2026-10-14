"""Exact, label-safe metadata joins for G3-B.

The public metadata object deliberately stores raw fields rather than a global
label or age vector.  Targets are parsed only for explicitly selected training
rows, preventing holdout outcomes from leaking into the model-input API.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


REQUIRED_COLUMNS = (
    "SiteID",
    "BidsFolder",
    "SessionID",
    "Cognitive_Impairment",
    "Age",
)
DEFAULT_SITE_COUNTS = {"I0002": 319, "I0006": 1142, "S0001": 5139}
MIN_AGE = 0.0
MAX_AGE = 130.0


@dataclass(frozen=True)
class MetadataRecord:
    site: str
    bids_folder: str
    session_id: str
    label_text: str
    age_text: str
    patient_id: str
    csv_row_index: int

    @property
    def record_stem(self) -> str:
        return f"{self.bids_folder}_ses-{self.session_id}"

    @property
    def composite_key(self) -> Tuple[str, str, str]:
        return self.site, self.bids_folder, self.session_id


@dataclass(frozen=True)
class MetadataIndex:
    records: Tuple[MetadataRecord, ...]
    cache_names: Tuple[str, ...]
    relative_paths: Tuple[str, ...]
    sites: np.ndarray
    demographics_sha256: str
    join_fingerprint: str
    audit: Mapping[str, object]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_label(raw: object) -> int:
    """Parse only the Challenge's explicitly accepted Boolean spellings."""

    token = str(raw).strip().lower()
    if token in {"true", "1", "t", "yes"}:
        return 1
    if token in {"false", "0", "f", "no"}:
        return 0
    raise ValueError(f"illegal Cognitive_Impairment value: {raw!r}")


def parse_age(raw: object) -> float:
    token = str(raw).strip()
    if not token:
        raise ValueError("missing Age")
    try:
        value = float(token)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"illegal Age value: {raw!r}") from exc
    if not math.isfinite(value) or not MIN_AGE <= value <= MAX_AGE:
        raise ValueError(f"Age outside finite [{MIN_AGE}, {MAX_AGE}] contract: {raw!r}")
    return value


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _inventory_rows(payload: object) -> Sequence[Mapping[str, object]]:
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return payload["rows"]
    if isinstance(payload, list):
        return payload
    raise ValueError("inventory.json must contain a rows list")


def _normalize_source_row(row: Mapping[str, object]) -> Tuple[str, str, str]:
    try:
        cache_name = str(row["cache_name"])
        relative_path = str(row["relative_path"])
        site = str(row["site"])
    except KeyError as exc:
        raise ValueError(f"source row missing field: {exc.args[0]}") from exc
    return cache_name, relative_path, site


def load_metadata_index(
    csv_path: Path,
    g2_root: Path,
    *,
    expected_counts: Optional[Mapping[str, int]] = None,
) -> MetadataIndex:
    """Bind demographics rows to the frozen anonymous cache order exactly."""

    csv_path = Path(csv_path)
    g2_root = Path(g2_root)
    expected = dict(DEFAULT_SITE_COUNTS if expected_counts is None else expected_counts)
    if not csv_path.is_file():
        raise ValueError(f"demographics CSV unavailable: {csv_path}")
    inventory_path = g2_root / "inventory.json"
    private_index_path = g2_root / "private_index.json"
    if not inventory_path.is_file() or not private_index_path.is_file():
        raise ValueError("G2-B1 inventory/private index unavailable")

    inventory = list(_inventory_rows(_load_json(inventory_path)))
    private_index = _load_json(private_index_path)
    if not isinstance(private_index, list) or len(inventory) != len(private_index):
        raise ValueError("inventory/private index length mismatch")

    source_rows = []
    cache_names = set()
    relative_paths = set()
    for position, (inv, idx) in enumerate(zip(inventory, private_index)):
        inv_row = _normalize_source_row(inv)
        idx_row = _normalize_source_row(idx)
        if inv_row != idx_row:
            raise ValueError(f"inventory/private index mismatch at position {position}")
        cache_name, relative_path, site = inv_row
        if cache_name != f"cache_{position:06d}.npz":
            raise ValueError(f"anonymous cache order mismatch at position {position}")
        if cache_name in cache_names or relative_path in relative_paths:
            raise ValueError("duplicate cache_name or relative_path")
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts or len(path.parts) != 2:
            raise ValueError(f"unsafe relative path at position {position}")
        if path.parts[0] != site or path.suffix != ".npz":
            raise ValueError(f"relative path/site mismatch at position {position}")
        cache_names.add(cache_name)
        relative_paths.add(relative_path)
        source_rows.append(inv_row)

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = set(REQUIRED_COLUMNS) - fields
        if missing:
            raise ValueError(f"demographics missing required columns: {sorted(missing)}")
        has_patient_id = "BDSPPatientID" in fields
        csv_records = []
        for row_number, row in enumerate(reader):
            site = str(row["SiteID"]).strip()
            bids = str(row["BidsFolder"]).strip()
            session = str(row["SessionID"]).strip()
            if not site or not bids or not session:
                raise ValueError(f"blank identity field in demographics row {row_number}")
            csv_records.append(
                MetadataRecord(
                    site=site,
                    bids_folder=bids,
                    session_id=session,
                    label_text=str(row["Cognitive_Impairment"]),
                    age_text=str(row["Age"]),
                    patient_id=str(row.get("BDSPPatientID", "")).strip() if has_patient_id else "",
                    csv_row_index=row_number,
                )
            )

    by_key: Dict[Tuple[str, str, str], MetadataRecord] = {}
    by_source: Dict[Tuple[str, str], MetadataRecord] = {}
    for record in csv_records:
        if record.composite_key in by_key:
            raise ValueError(f"duplicate demographics composite key: {record.composite_key}")
        by_key[record.composite_key] = record
        source_key = (record.site, record.record_stem)
        if source_key in by_source:
            raise ValueError(f"duplicate demographics source identity: {source_key}")
        by_source[source_key] = record

    ordered = []
    actual_counts: Dict[str, int] = {}
    for position, (_, relative_path, site) in enumerate(source_rows):
        stem = Path(relative_path).stem
        record = by_source.get((site, stem))
        if record is None:
            raise ValueError(f"source identity does not map exactly once at position {position}")
        if record.site != site or record.record_stem != stem:
            raise ValueError(f"source identity mismatch at position {position}")
        ordered.append(record)
        actual_counts[site] = actual_counts.get(site, 0) + 1

    if len(ordered) != len(csv_records):
        raise ValueError("demographics/source mapping is not bidirectionally complete")
    if actual_counts != expected:
        raise ValueError(f"site counts mismatch: {actual_counts} != {expected}")
    if len({record.composite_key for record in ordered}) != len(ordered):
        raise ValueError("ordered metadata contains duplicate records")

    canonical_join = [
        {
            "cache_name": source_rows[i][0],
            "relative_path": source_rows[i][1],
            "site": record.site,
            "bids_folder": record.bids_folder,
            "session_id": record.session_id,
        }
        for i, record in enumerate(ordered)
    ]
    demographics_sha = _sha256(csv_path)
    audit = {
        "matched": len(ordered),
        "metadata_rows": len(csv_records),
        "cache_rows": len(source_rows),
        "duplicate_count": 0,
        "unmatched_count": 0,
        "site_mismatch_count": 0,
        "site_counts": actual_counts,
        "patient_id_available": has_patient_id,
    }
    return MetadataIndex(
        records=tuple(ordered),
        cache_names=tuple(row[0] for row in source_rows),
        relative_paths=tuple(row[1] for row in source_rows),
        sites=np.asarray([record.site for record in ordered], dtype="U5"),
        demographics_sha256=demographics_sha,
        join_fingerprint=_canonical_sha256(canonical_join),
        audit=audit,
    )


def check_identity_join(metadata: MetadataIndex) -> Dict[str, object]:
    result = dict(metadata.audit)
    if result["matched"] != len(metadata.records):
        raise ValueError("metadata audit count is inconsistent")
    return result


def _checked_indices(indices: Iterable[int], size: int) -> np.ndarray:
    values = np.asarray(indices)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError("indices must be a one-dimensional integer array")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0) or np.any(values >= size) or len(np.unique(values)) != len(values):
        raise ValueError("indices are duplicated or out of range")
    return values


def load_train_targets(metadata: MetadataIndex, train_indices: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Parse labels and ages only for selected training rows."""

    indices = _checked_indices(train_indices, len(metadata.records))
    labels = np.empty(len(indices), dtype=np.int8)
    ages = np.empty(len(indices), dtype=np.float64)
    for out_index, source_index in enumerate(indices):
        record = metadata.records[int(source_index)]
        labels[out_index] = parse_label(record.label_text)
        ages[out_index] = parse_age(record.age_text)
    return labels, ages


def make_age_groups(ages: np.ndarray) -> np.ndarray:
    values = np.asarray(ages)
    if values.ndim != 1:
        raise ValueError("ages must be one-dimensional")
    checked = np.asarray([parse_age(value) for value in values], dtype=np.float64)
    return np.floor(checked / 5.0).astype(np.int64)


def audit_age_groups(groups: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    groups = np.asarray(groups)
    labels = np.asarray(labels)
    if groups.ndim != 1 or labels.ndim != 1 or groups.shape != labels.shape:
        raise ValueError("groups and labels must be aligned one-dimensional arrays")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("labels must be binary")
    rows = []
    for group in np.unique(groups):
        selected = labels[groups == group]
        positives = int(np.sum(selected == 1))
        negatives = int(np.sum(selected == 0))
        rows.append({
            "group": int(group),
            "count": int(len(selected)),
            "positive": positives,
            "negative": negatives,
            "has_both_classes": bool(positives and negatives),
        })
    return {
        "group_count": len(rows),
        "single_class_groups": sum(not row["has_both_classes"] for row in rows),
        "groups_smaller_than_two": sum(row["count"] < 2 for row in rows),
        "groups_with_both_classes": sum(row["has_both_classes"] for row in rows),
        "groups": rows,
    }


def check_patient_overlap(metadata: MetadataIndex, train_indices: Iterable[int], test_indices: Iterable[int]) -> int:
    train = _checked_indices(train_indices, len(metadata.records))
    test = _checked_indices(test_indices, len(metadata.records))
    if set(train.tolist()) & set(test.tolist()):
        raise ValueError("train and test record indices overlap")
    if not metadata.audit.get("patient_id_available"):
        return 0

    def patient_keys(indices: np.ndarray) -> set:
        keys = set()
        for index in indices:
            record = metadata.records[int(index)]
            if not record.patient_id:
                raise ValueError("blank BDSPPatientID prevents patient overlap audit")
            keys.add((record.site, record.patient_id))
        return keys

    overlap = patient_keys(train) & patient_keys(test)
    if overlap:
        raise ValueError(f"patient overlap detected across train/test: {len(overlap)}")
    return 0
