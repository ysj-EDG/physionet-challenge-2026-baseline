from __future__ import annotations

import unittest
from pathlib import Path


DOCUMENT = (
    Path(__file__).resolve().parents[3]
    / "docs/featurelab/BATCH2_PREPROCESSING_REVIEW.md"
)
REQUIRED_COLUMNS = (
    "family_id",
    "source_channel",
    "current_reader",
    "current_preprocessing",
    "candidate_preprocessing",
    "sample_rate",
    "filtering",
    "window",
    "validity",
    "threshold_reference",
    "reuse_decision",
    "new_branch",
    "intermediate_cache",
    "edf_required",
    "estimated_cost",
    "overlap_audit",
    "unresolved_parameters",
    "expected_dimension",
    "reference_and_adaptation",
    "status",
)
EXPECTED_FAMILIES = {
    "eeg_spectral": 288,
    "eeg_complexity": 192,
    "ecg_fragmentation": 20,
    "caisr_intervals": 9,
    "resp_complexity": 24,
    "spo2_hypoxia": 20,
}


def _table_rows(text: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("|")]
    for index, line in enumerate(lines):
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if tuple(cells) != REQUIRED_COLUMNS:
            continue
        separator = lines[index + 1]
        self_cells = [cell.strip() for cell in separator.strip("|").split("|")]
        if len(self_cells) != len(REQUIRED_COLUMNS) or not all(
            set(cell) <= {"-", ":"} for cell in self_cells
        ):
            raise AssertionError("Batch2 table separator is malformed")
        rows = []
        for candidate in lines[index + 2 :]:
            values = [cell.strip() for cell in candidate.strip("|").split("|")]
            if len(values) != len(REQUIRED_COLUMNS):
                break
            rows.append(dict(zip(REQUIRED_COLUMNS, values)))
        return rows
    raise AssertionError("Required Batch2 preprocessing table was not found")


class Batch2DocumentContractTests(unittest.TestCase):
    def test_all_six_candidate_families_have_complete_review_rows(self) -> None:
        text = DOCUMENT.read_text(encoding="utf-8")
        rows = _table_rows(text)
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            {row["family_id"]: int(row["expected_dimension"]) for row in rows},
            EXPECTED_FAMILIES,
        )
        self.assertEqual(sum(int(row["expected_dimension"]) for row in rows), 553)
        for row in rows:
            with self.subTest(row["family_id"]):
                for column in REQUIRED_COLUMNS:
                    self.assertTrue(row[column], column)
                self.assertIn(
                    row["status"], {"READY_TO_IMPLEMENT", "REQUIRES_REVIEW"}
                )

    def test_review_is_explicitly_nonimplementation_and_lists_decision_risks(self) -> None:
        text = DOCUMENT.read_text(encoding="utf-8")
        self.assertIn("BATCH2_IMPLEMENTATION = NOT_STARTED", text)
        self.assertIn("## Risk register and approval questions", text)
        self.assertIn("## Intermediate-cache recommendations", text)


class Phase4PublicDocumentContractTests(unittest.TestCase):
    def _document(self, name: str) -> str:
        path = Path(__file__).resolve().parents[3] / "docs/featurelab" / name
        return path.read_text(encoding="utf-8")

    def test_pilot_report_is_aggregate_only_and_complete(self) -> None:
        text = self._document("FEATURELAB_BATCH1_PILOT_REPORT.md")
        for required in (
            "CROSS_SITE_PILOT = COMPLETE",
            "SOURCE_IMMUTABILITY = VERIFIED",
            "CACHE_READBACK = VERIFIED",
            "FULL_6600_STATUS = NOT_STARTED",
            "BATCH2_553_STATUS = AUDIT_ONLY",
            "I0002",
            "I0006",
            "S0001",
            "eeg_tail",
            "resp_stage",
        ):
            self.assertIn(required, text)
        for forbidden in (
            "record_key",
            "record_id",
            "source_path",
            "cache_path",
            "rec_",
        ):
            self.assertNotIn(forbidden, text)

    def test_full_plan_requires_explicit_authorization_and_has_stop_rules(self) -> None:
        text = self._document("FEATURELAB_BATCH1_FULL_EXTRACTION_PLAN.md")
        self.assertIn("FULL_6600_STATUS = NOT_STARTED", text)
        self.assertIn("FULL_EXTRACTION_AUTHORIZATION = REQUIRED", text)
        self.assertIn("--execution-mode full", text)
        self.assertIn("--expected-inventory-size 6600", text)
        self.assertIn("## Stop conditions", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
