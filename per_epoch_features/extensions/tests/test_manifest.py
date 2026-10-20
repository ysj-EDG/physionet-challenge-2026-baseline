from __future__ import annotations

import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from extensions.manifest import (
    MANIFEST_COLUMNS,
    build_manifest_rows,
    frozen_feature_ids,
    write_manifest,
)
from extensions.registry import list_families


class ManifestTests(unittest.TestCase):
    def test_family_dimensions_global_uniqueness_and_frozen_nonoverlap(self) -> None:
        specs = list_families()
        self.assertEqual(
            [(spec.family_id, len(spec.feature_definitions)) for spec in specs],
            [
                ("eeg_tail", 288),
                ("eeg_temporal", 96),
                ("eeg_coherence_stage", 90),
                ("caisr_stage", 35),
                ("emg_stage", 72),
                ("resp_stage", 56),
            ],
        )
        feature_ids = [
            item.feature_id for spec in specs for item in spec.feature_definitions
        ]
        self.assertEqual(len(feature_ids), 637)
        self.assertEqual(len(set(feature_ids)), 637)
        frozen = frozen_feature_ids()
        self.assertEqual(len(frozen), 807)
        self.assertEqual(set(feature_ids).intersection(frozen), set())

    def test_manifest_schema_statuses_and_proposed_view_counts(self) -> None:
        rows = build_manifest_rows()
        self.assertEqual(len(rows), 637)
        self.assertTrue(all(tuple(row) == MANIFEST_COLUMNS for row in rows))
        self.assertEqual(
            Counter(row["proposed_view"] for row in rows),
            Counter({"V1": 48, "V2": 48, "V3": 144, "V4": 144, "V5": 253}),
        )
        self.assertEqual(
            Counter(row["implementation_status"] for row in rows),
            Counter({"COMPUTED_FROM_EXISTING_NPZ": 627, "REUSED_FROM_EXISTING_STATIC": 10}),
        )
        temporal = [row for row in rows if row["family_id"] == "eeg_temporal"]
        self.assertEqual(
            [row["minimum_support"] for row in temporal[:4]],
            ["20", "20", "96", "1"],
        )
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "manifest.csv"
            write_manifest(target)
            with target.open(newline="", encoding="utf-8") as handle:
                stored = list(csv.DictReader(handle))
            self.assertEqual(stored, rows)

    def test_tracked_manifest_matches_deterministic_generator(self) -> None:
        repository_root = Path(__file__).resolve().parents[3]
        target = repository_root / "audit/featurelab_batch1_manifest.csv"
        with target.open(newline="", encoding="utf-8") as handle:
            stored = list(csv.DictReader(handle))
        self.assertEqual(stored, build_manifest_rows())


if __name__ == "__main__":
    unittest.main(verbosity=2)
