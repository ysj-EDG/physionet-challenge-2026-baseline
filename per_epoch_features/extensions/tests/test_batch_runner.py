from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np

from extensions import registry
from extensions.batch_runner import (
    BatchVerificationError,
    assert_result_payload_equal,
    build_path_inventory,
    run_batch,
    select_pilot_inventory,
)
from extensions.cache import read_cache
from extensions.contracts import FamilyRunStatus, RecordRunStatus
from extensions.tests.fixtures import source_payload, write_source_npz


class BatchRunnerTests(unittest.TestCase):
    def _sources(self, root: Path, count: int = 2):
        source_root = root / "source"
        paths = tuple(
            write_source_npz(
                source_root,
                site_id=f"SITE_{index % 2}",
                record_id=f"record-{index:03d}",
                payload=source_payload(
                    site_id=f"SITE_{index % 2}",
                    record_id=f"record-{index:03d}",
                    epochs=240,
                ),
            )
            for index in range(count)
        )
        return source_root, paths

    @staticmethod
    def _family_ids():
        return tuple(spec.family_id for spec in registry.list_families())

    def _run(
        self,
        root: Path,
        inventory,
        source_root: Path,
        *,
        family_ids=None,
        checkpoint_name="checkpoint.json",
        **kwargs,
    ):
        return run_batch(
            inventory,
            source_root,
            root / "cache",
            root / checkpoint_name,
            family_ids or self._family_ids(),
            kwargs.pop("config_overrides", {}),
            max_records=kwargs.pop("max_records", len(inventory)),
            **kwargs,
        )

    def test_inventory_is_deterministic_path_only_and_collision_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root, paths = self._sources(root, 3)
            with mock.patch(
                "extensions.sources.load_source_npz",
                side_effect=AssertionError("inventory must not decode NPZ"),
            ):
                first = build_path_inventory(source_root, reversed(paths))
                second = build_path_inventory(source_root, paths)
            self.assertEqual(first, second)
            self.assertEqual(
                [item.relative_path for item in first],
                sorted(item.relative_path for item in first),
            )
            with mock.patch(
                "extensions.batch_runner.record_key_for", return_value="rec_" + "0" * 64
            ):
                with self.assertRaises(ValueError):
                    build_path_inventory(source_root, paths)

    def test_dry_run_and_pilot_limit_never_compute_or_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root, paths = self._sources(root, 1)
            inventory = build_path_inventory(source_root, paths)
            with mock.patch("extensions.batch_runner.run_record") as run:
                summary = self._run(root, inventory, source_root, dry_run=True)
            run.assert_not_called()
            self.assertTrue(summary.dry_run)
            self.assertFalse((root / "cache").exists())
            self.assertFalse((root / "checkpoint.json").exists())
            with self.assertRaises(ValueError):
                self._run(root, inventory, source_root, max_records=13)
            with self.assertRaises(PermissionError):
                self._run(
                    root,
                    inventory,
                    source_root,
                    execution_mode="full",
                )

    def test_initial_build_then_identical_batch_is_all_verified_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root, paths = self._sources(root, 2)
            inventory = build_path_inventory(source_root, paths)
            first = self._run(root, inventory, source_root)
            second = self._run(root, inventory, source_root)
            self.assertEqual(first.status_counts["BUILT"], 12)
            self.assertEqual(second.status_counts["SKIP_VERIFIED"], 12)
            self.assertTrue(all(item.readback_verified for item in first.records))

    def test_interruption_checkpoint_resume_only_builds_missing_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root, paths = self._sources(root, 2)
            inventory = build_path_inventory(source_root, paths)
            interrupted = self._run(
                root, inventory, source_root, stop_after=1
            )
            self.assertTrue(interrupted.interrupted)
            self.assertEqual(interrupted.executed_count, 1)
            resumed = self._run(root, inventory, source_root)
            statuses = [
                [family.status for family in record.run.family_results]
                for record in resumed.records
            ]
            self.assertEqual(statuses[0], [FamilyRunStatus.SKIP_VERIFIED] * 6)
            self.assertEqual(statuses[1], [FamilyRunStatus.BUILT] * 6)

    def test_existing_single_family_is_reused_by_six_family_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root, paths = self._sources(root, 1)
            inventory = build_path_inventory(source_root, paths)
            first_family = self._family_ids()[0]
            self._run(
                root,
                inventory,
                source_root,
                family_ids=(first_family,),
                checkpoint_name="single.json",
            )
            combined = self._run(root, inventory, source_root)
            statuses = [item.status for item in combined.records[0].run.family_results]
            self.assertEqual(statuses[0], FamilyRunStatus.SKIP_VERIFIED)
            self.assertEqual(statuses[1:], [FamilyRunStatus.BUILT] * 5)

    def test_corrupt_family_cache_stops_invalid_without_clobber(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root, paths = self._sources(root, 1)
            inventory = build_path_inventory(source_root, paths)
            built = self._run(root, inventory, source_root)
            target = built.records[0].run.family_results[2].cache_path
            target.write_bytes(b"corrupt test copy")
            original = target.read_bytes()
            resumed = self._run(root, inventory, source_root)
            statuses = [item.status for item in resumed.records[0].run.family_results]
            self.assertEqual(statuses[2], FamilyRunStatus.STOP_INVALID)
            self.assertEqual(target.read_bytes(), original)

    def test_family_input_review_and_record_stop_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            missing = source_payload(epochs=240)
            del missing["emg_epoch_success"]
            path = write_source_npz(source_root, payload=missing)
            inventory = build_path_inventory(source_root, (path,))
            result = self._run(root, inventory, source_root)
            by_family = {
                item.family_id: item.status
                for item in result.records[0].run.family_results
            }
            self.assertEqual(
                by_family["emg_stage"], FamilyRunStatus.REQUIRES_INPUT_REVIEW
            )
            self.assertTrue(
                all(
                    status is FamilyRunStatus.BUILT
                    for family, status in by_family.items()
                    if family != "emg_stage"
                )
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            bad = source_payload(site_id="OTHER", epochs=240)
            path = write_source_npz(source_root, payload=bad)
            inventory = build_path_inventory(source_root, (path,))
            stopped = self._run(root, inventory, source_root)
            self.assertEqual(stopped.records[0].run.status, RecordRunStatus.STOP_RECORD)
            self.assertEqual(stopped.records[0].run.family_results, ())
            self.assertFalse((root / "cache").exists())

    def test_two_workers_competing_for_same_targets_never_clobber(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root, paths = self._sources(root, 1)
            inventory = build_path_inventory(source_root, paths)

            def invoke(index):
                return self._run(
                    root,
                    inventory,
                    source_root,
                    checkpoint_name=f"race-{index}.json",
                    workers=2,
                )

            with ThreadPoolExecutor(max_workers=2) as pool:
                summaries = tuple(pool.map(invoke, (0, 1)))
            for family_index in range(6):
                statuses = {
                    summary.records[0].run.family_results[family_index].status
                    for summary in summaries
                }
                self.assertTrue(statuses <= {
                    FamilyRunStatus.BUILT,
                    FamilyRunStatus.SKIP_VERIFIED,
                })
                self.assertIn(FamilyRunStatus.SKIP_VERIFIED, statuses)
            self.assertEqual(len(list((root / "cache").glob("*/*/*.npz"))), 6)

    def test_one_family_configuration_change_only_builds_new_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root, paths = self._sources(root, 1)
            inventory = build_path_inventory(source_root, paths)
            baseline = self._run(root, inventory, source_root)
            changed = self._run(
                root,
                inventory,
                source_root,
                config_overrides={"eeg_tail": {"minimum_support": 21}},
            )
            statuses = {
                item.family_id: item.status
                for item in changed.records[0].run.family_results
            }
            self.assertEqual(statuses["eeg_tail"], FamilyRunStatus.BUILT)
            self.assertTrue(
                all(
                    status is FamilyRunStatus.SKIP_VERIFIED
                    for family, status in statuses.items()
                    if family != "eeg_tail"
                )
            )
            old_path = baseline.records[0].run.family_results[0].cache_path
            new_path = changed.records[0].run.family_results[0].cache_path
            self.assertNotEqual(old_path, new_path)

    def test_built_payloads_match_all_feature_result_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root, paths = self._sources(root, 1)
            inventory = build_path_inventory(source_root, paths)
            built = self._run(root, inventory, source_root)
            for family in built.records[0].run.family_results:
                payload = read_cache(family.cache_path)
                assert_result_payload_equal(
                    family.result, payload, family.cache_path.parent.name
                )
            family = built.records[0].run.family_results[0]
            changed = read_cache(family.cache_path)
            changed["values"] = changed["values"].copy()
            changed["values"][0] += 1.0
            with self.assertRaises(BatchVerificationError):
                assert_result_payload_equal(
                    family.result, changed, family.cache_path.parent.name
                )

    def test_pilot_selection_is_cross_site_bounded_and_label_free(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            paths = []
            for site_index in range(3):
                for record_index in range(3):
                    site_id = f"SITE_{site_index}"
                    record_id = f"record-{record_index:03d}"
                    paths.append(write_source_npz(
                        source_root,
                        site_id=site_id,
                        record_id=record_id,
                        payload=source_payload(
                            site_id=site_id,
                            record_id=record_id,
                            epochs=240,
                        ),
                    ))
            inventory = build_path_inventory(source_root, reversed(paths))
            selection = select_pilot_inventory(
                inventory,
                source_root,
                max_candidates=6,
                max_records=6,
                per_site=2,
                preferred_paths=(paths[-1],),
            )
            self.assertEqual(selection.checked_count, 6)
            self.assertEqual(len(selection.inventory), 6)
            self.assertEqual(
                dict(selection.site_counts),
                {"SITE_0": 2, "SITE_1": 2, "SITE_2": 2},
            )
            self.assertIn(
                paths[-1].relative_to(source_root).as_posix(),
                {entry.relative_path for entry in selection.inventory},
            )

    def test_one_bad_record_does_not_terminate_remaining_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            bad = write_source_npz(
                source_root,
                site_id="SITE_0",
                record_id="bad",
                payload=source_payload(site_id="OTHER", record_id="bad", epochs=240),
            )
            good = write_source_npz(
                source_root,
                site_id="SITE_1",
                record_id="good",
                payload=source_payload(site_id="SITE_1", record_id="good", epochs=240),
            )
            inventory = build_path_inventory(source_root, (bad, good))
            summary = self._run(root, inventory, source_root)
            self.assertEqual(summary.executed_count, 2)
            by_record = {
                item.entry.record_id: item.run.status for item in summary.records
            }
            self.assertEqual(by_record["bad"], RecordRunStatus.STOP_RECORD)
            self.assertEqual(by_record["good"], RecordRunStatus.COMPLETE)

    def test_pilot_selection_does_not_hide_unexpected_programming_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root, paths = self._sources(root, 1)
            inventory = build_path_inventory(source_root, paths)
            with mock.patch(
                "extensions.batch_runner.load_source_npz",
                side_effect=RuntimeError("unexpected selector defect"),
            ):
                with self.assertRaises(RuntimeError):
                    select_pilot_inventory(
                        inventory,
                        source_root,
                        max_candidates=1,
                        max_records=1,
                    )


if __name__ == "__main__":
    unittest.main()
