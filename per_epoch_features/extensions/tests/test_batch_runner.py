from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np

from extensions import batch_runner, registry
from extensions.batch_runner import (
    BatchVerificationError,
    SourceInventoryEntry,
    assert_result_payload_equal,
    build_path_inventory,
    run_batch,
    select_pilot_inventory,
)
from extensions.cache import read_cache
from extensions.contracts import (
    FamilyRunResult,
    FamilyRunStatus,
    RecordRunStatus,
    RecordRunSummary,
)
from extensions.sources import record_key_for
from extensions.tests.fixtures import source_payload, write_source_npz


class BatchRunnerTests(unittest.TestCase):
    FULL_SITE_COUNTS = {"I0002": 319, "I0006": 1142, "S0001": 5139}
    FULL_FAMILY_IDS = (
        "eeg_tail",
        "eeg_temporal",
        "eeg_coherence_stage",
        "caisr_stage",
        "emg_stage",
        "resp_stage",
    )

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
    def _full_inventory(self, root: Path, payloads=()):
        source_root = root / "source"
        entries = []
        for site_id, count in self.FULL_SITE_COUNTS.items():
            for index in range(count):
                record_id = f"record-{index:05d}"
                entries.append(
                    SourceInventoryEntry(
                        relative_path=f"{site_id}/{record_id}.npz",
                        site_id=site_id,
                        record_id=record_id,
                        record_key=record_key_for(site_id, record_id),
                    )
                )
        ordered = tuple(sorted(entries))
        for index, payload in enumerate(payloads):
            entry = ordered[index]
            write_source_npz(
                source_root,
                site_id=entry.site_id,
                record_id=entry.record_id,
                payload=payload,
            )
        return source_root, ordered

    def _run_full(
        self,
        root: Path,
        inventory,
        source_root: Path,
        *,
        authorization=True,
        authorization_path=None,
        **kwargs,
    ):
        auth = authorization_path or root / "FULL_EXTRACTION_APPROVED.txt"
        if authorization:
            auth.write_text("FEATURELAB_FULL_EXTRACTION_APPROVED\n", encoding="utf-8")
        with (
            mock.patch.object(
                batch_runner, "FULL_AUTHORIZATION_PATH", auth, create=True
            ),
            mock.patch.object(
                batch_runner,
                "_available_cache_bytes",
                return_value=10 * 1024**3,
                create=True,
            ),
            mock.patch.object(
                batch_runner,
                "_peak_rss_kb",
                return_value=100_000,
                create=True,
            ),
        ):
            return run_batch(
                inventory,
                source_root,
                root / "cache",
                root / "full-checkpoint.json",
                kwargs.pop("family_ids", self.FULL_FAMILY_IDS),
                kwargs.pop("config_overrides", {}),
                max_records=kwargs.pop("max_records", 6600),
                workers=kwargs.pop("workers", 1),
                execution_mode="full",
                full_authorization_file=auth if authorization else None,
                expected_inventory_size=kwargs.pop("expected_inventory_size", 6600),
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

    def test_full_stop_invalid_persists_checkpoint_and_does_not_run_next_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = [
                source_payload(
                    site_id="I0002", record_id=f"record-{index:05d}", epochs=240
                )
                for index in range(2)
            ]
            source_root, inventory = self._full_inventory(root, payloads)
            seed = self._run(
                root,
                inventory[:1],
                source_root,
                checkpoint_name="seed.json",
            )
            target = seed.records[0].run.family_results[0].cache_path
            target.write_bytes(b"corrupt synthetic cache")
            corrupted = target.read_bytes()

            summary = self._run_full(
                root, inventory, source_root, stop_after=2
            )

            self.assertEqual(summary.executed_count, 1)
            self.assertEqual(summary.safety_stop.reason_code, "CACHE_STOP_INVALID")
            self.assertFalse(summary.safety_stop.safe_to_resume)
            self.assertEqual(target.read_bytes(), corrupted)
            checkpoint = json.loads(
                (root / "full-checkpoint.json").read_text(encoding="utf-8")
            )
            state = checkpoint["requests"][summary.request_digest]
            self.assertEqual(len(state["records"]), 1)
            self.assertEqual(
                state["active_safety_stop"]["reason_code"], "CACHE_STOP_INVALID"
            )

    def test_full_source_immutability_failure_stops_after_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = [
                source_payload(
                    site_id="I0002", record_id=f"record-{index:05d}", epochs=240
                )
                for index in range(2)
            ]
            source_root, inventory = self._full_inventory(root, payloads)
            original_snapshot = batch_runner._source_snapshot
            calls = {}

            def changed_after_first(path):
                snapshot = original_snapshot(path)
                key = str(path)
                calls[key] = calls.get(key, 0) + 1
                if key.endswith("record-00000.npz") and calls[key] == 2:
                    snapshot["sha256"] = "f" * 64
                return snapshot

            with mock.patch(
                "extensions.batch_runner._source_snapshot",
                side_effect=changed_after_first,
            ):
                summary = self._run_full(
                    root, inventory, source_root, stop_after=2
                )

            self.assertEqual(summary.executed_count, 1)
            self.assertEqual(summary.safety_stop.reason_code, "SOURCE_IMMUTABILITY")
            checkpoint = json.loads(
                (root / "full-checkpoint.json").read_text(encoding="utf-8")
            )
            state = checkpoint["requests"][summary.request_digest]
            self.assertFalse(next(iter(state["records"].values()))["source_immutable"])
            self.assertEqual(
                state["active_safety_stop"]["completed_count"], 1
            )

    def test_full_readback_failure_stops_and_requires_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = [
                source_payload(
                    site_id="I0002", record_id=f"record-{index:05d}", epochs=240
                )
                for index in range(2)
            ]
            source_root, inventory = self._full_inventory(root, payloads)
            with mock.patch(
                "extensions.batch_runner.read_cache",
                side_effect=BatchVerificationError("synthetic readback failure"),
            ):
                summary = self._run_full(
                    root, inventory, source_root, stop_after=2
                )
            self.assertEqual(summary.executed_count, 1)
            self.assertEqual(summary.safety_stop.reason_code, "CACHE_READBACK_FAILED")
            self.assertNotEqual(batch_runner.summary_exit_code(summary), 0)

    def test_full_repeated_same_site_family_input_error_is_systemic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = []
            for index in range(2):
                payload = source_payload(
                    site_id="I0002", record_id=f"record-{index:05d}", epochs=240
                )
                del payload["emg_epoch_success"]
                payloads.append(payload)
            source_root, inventory = self._full_inventory(root, payloads)

            summary = self._run_full(
                root, inventory, source_root, stop_after=2
            )

            self.assertEqual(summary.executed_count, 2)
            self.assertEqual(
                summary.safety_stop.reason_code, "REPEATED_FAMILY_INPUT"
            )
            self.assertTrue(
                all(item.run.status is RecordRunStatus.PARTIAL for item in summary.records)
            )

    def test_full_single_family_input_error_is_isolated_and_queue_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = source_payload(
                site_id="I0002", record_id="record-00000", epochs=240
            )
            del first["emg_epoch_success"]
            second = source_payload(
                site_id="I0002", record_id="record-00001", epochs=240
            )
            source_root, inventory = self._full_inventory(root, (first, second))

            summary = self._run_full(
                root, inventory, source_root, stop_after=2
            )

            self.assertEqual(summary.executed_count, 2)
            self.assertIsNone(summary.safety_stop)
            self.assertEqual(summary.records[0].run.status, RecordRunStatus.PARTIAL)
            self.assertEqual(summary.records[1].run.status, RecordRunStatus.COMPLETE)

    def test_full_legal_low_support_nan_does_not_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root, inventory = self._full_inventory(
                root,
                (
                    source_payload(
                        site_id="I0002", record_id="record-00000", epochs=4
                    ),
                ),
            )

            summary = self._run_full(
                root, inventory, source_root, stop_after=1
            )

            self.assertIsNone(summary.safety_stop)
            self.assertEqual(summary.records[0].run.status, RecordRunStatus.COMPLETE)
            self.assertTrue(
                any(
                    item.result is not None and np.any(~item.result.valid_mask)
                    for item in summary.records[0].run.family_results
                )
            )

    def test_full_rejects_changed_family_order_and_nondefault_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root, inventory = self._full_inventory(root)
            with self.assertRaises(ValueError):
                self._run_full(
                    root,
                    inventory,
                    source_root,
                    family_ids=tuple(reversed(self.FULL_FAMILY_IDS)),
                    dry_run=True,
                )
            with self.assertRaises(ValueError):
                self._run_full(
                    root,
                    inventory,
                    source_root,
                    config_overrides={"eeg_tail": {"minimum_support": 21}},
                    dry_run=True,
                )

    def test_full_rejects_multiple_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root, inventory = self._full_inventory(root)
            with self.assertRaises(ValueError):
                self._run_full(
                    root, inventory, source_root, workers=2, dry_run=True
                )

    def test_full_rejects_wrong_site_distribution_and_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root, inventory = self._full_inventory(root)
            wrong_site = list(inventory)
            first = wrong_site[0]
            wrong_site[0] = SourceInventoryEntry(
                relative_path=first.relative_path,
                site_id="WRONG",
                record_id=first.record_id,
                record_key=first.record_key,
            )
            with self.assertRaises(ValueError):
                self._run_full(
                    root, tuple(wrong_site), source_root, dry_run=True
                )

            frozen = {
                spec.family_id: "0" * 64 for spec in registry.list_families()
            }
            with mock.patch.object(
                batch_runner, "FULL_RECIPE_IDS", frozen, create=True
            ):
                with self.assertRaises(ValueError):
                    self._run_full(root, inventory, source_root, dry_run=True)

    def test_full_safety_stop_checkpoint_resume_does_not_fake_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = [
                source_payload(
                    site_id="I0002", record_id=f"record-{index:05d}", epochs=240
                )
                for index in range(2)
            ]
            source_root, inventory = self._full_inventory(root, payloads)
            seed = self._run(
                root,
                inventory[:1],
                source_root,
                checkpoint_name="seed.json",
            )
            target = seed.records[0].run.family_results[0].cache_path
            target.write_bytes(b"corrupt synthetic cache")
            first = self._run_full(root, inventory, source_root, stop_after=2)
            second = self._run_full(root, inventory, source_root, stop_after=2)

            self.assertEqual(first.executed_count, 1)
            self.assertEqual(second.executed_count, 0)
            self.assertEqual(second.safety_stop.reason_code, "CACHE_STOP_INVALID")
            checkpoint = json.loads(
                (root / "full-checkpoint.json").read_text(encoding="utf-8")
            )
            record = checkpoint["requests"][first.request_digest]["records"][
                inventory[0].record_key
            ]
            self.assertNotEqual(record["record_status"], "COMPLETE")

    def test_full_requires_exact_private_authorization_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root, inventory = self._full_inventory(root)
            with self.assertRaises(PermissionError):
                self._run_full(
                    root, inventory, source_root, authorization=False, dry_run=True
                )
            expected = root / "expected" / "FULL_EXTRACTION_APPROVED.txt"
            wrong = root / "wrong" / "FULL_EXTRACTION_APPROVED.txt"
            expected.parent.mkdir(parents=True)
            wrong.parent.mkdir(parents=True)
            wrong.write_text(
                "FEATURELAB_FULL_EXTRACTION_APPROVED\n", encoding="utf-8"
            )
            with mock.patch.object(
                batch_runner, "FULL_AUTHORIZATION_PATH", expected, create=True
            ):
                with self.assertRaises(PermissionError):
                    run_batch(
                        inventory,
                        source_root,
                        root / "cache",
                        root / "checkpoint.json",
                        self.FULL_FAMILY_IDS,
                        {},
                        max_records=6600,
                        workers=1,
                        dry_run=True,
                        execution_mode="full",
                        full_authorization_file=wrong,
                        expected_inventory_size=6600,
                    )

    def test_full_dry_run_never_computes_or_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root, inventory = self._full_inventory(root)
            with mock.patch("extensions.batch_runner.run_record") as compute:
                summary = self._run_full(
                    root, inventory, source_root, dry_run=True
                )
            compute.assert_not_called()
            self.assertTrue(summary.dry_run)
            self.assertEqual(summary.executed_count, 0)
            self.assertFalse((root / "cache").exists())
            self.assertFalse((root / "full-checkpoint.json").exists())

    def test_full_runtime_disk_guard_stops_after_persisting_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = [
                source_payload(
                    site_id="I0002", record_id=f"record-{index:05d}", epochs=240
                )
                for index in range(2)
            ]
            source_root, inventory = self._full_inventory(root, payloads)
            auth = root / "FULL_EXTRACTION_APPROVED.txt"
            auth.write_text("FEATURELAB_FULL_EXTRACTION_APPROVED\n", encoding="utf-8")
            with (
                mock.patch.object(batch_runner, "FULL_AUTHORIZATION_PATH", auth, create=True),
                mock.patch.object(
                    batch_runner,
                    "_available_cache_bytes",
                    side_effect=(3 * 1024**3, 512 * 1024**2),
                    create=True,
                ),
                mock.patch.object(
                    batch_runner, "_peak_rss_kb", return_value=100_000, create=True
                ),
            ):
                summary = run_batch(
                    inventory,
                    source_root,
                    root / "cache",
                    root / "full-checkpoint.json",
                    self.FULL_FAMILY_IDS,
                    {},
                    max_records=6600,
                    workers=1,
                    stop_after=2,
                    execution_mode="full",
                    full_authorization_file=auth,
                    expected_inventory_size=6600,
                )
            self.assertEqual(summary.executed_count, 1)
            self.assertEqual(summary.safety_stop.reason_code, "LOW_DISK_RUNTIME")
            self.assertTrue(summary.safety_stop.safe_to_resume)

    def test_full_read_only_preflight_needs_no_authorization_or_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root, inventory = self._full_inventory(root)
            expected_auth = root / "FULL_EXTRACTION_APPROVED.txt"
            with (
                mock.patch.object(
                    batch_runner, "FULL_AUTHORIZATION_PATH", expected_auth, create=True
                ),
                mock.patch.object(
                    batch_runner,
                    "_available_cache_bytes",
                    return_value=10 * 1024**3,
                    create=True,
                ),
                mock.patch.object(
                    batch_runner, "_peak_rss_kb", return_value=100_000, create=True
                ),
                mock.patch("extensions.batch_runner.run_record") as compute,
            ):
                report = batch_runner.run_full_preflight(
                    inventory,
                    source_root,
                    root / "cache",
                    root / "full-checkpoint.json",
                    self.FULL_FAMILY_IDS,
                    {},
                    max_records=6600,
                    workers=1,
                    full_authorization_file=expected_auth,
                    expected_inventory_size=6600,
                )
            compute.assert_not_called()
            self.assertTrue(report.contract_valid)
            self.assertFalse(report.authorization_present)
            self.assertFalse(report.execution_permitted)
            self.assertFalse((root / "cache").exists())
            self.assertFalse((root / "full-checkpoint.json").exists())
    def test_full_isolated_compute_value_error_does_not_stop_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = [
                source_payload(
                    site_id="I0002", record_id=f"record-{index:05d}", epochs=240
                )
                for index in range(2)
            ]
            source_root, inventory = self._full_inventory(root, payloads)
            original_run_record = batch_runner.run_record
            calls = 0

            def inject_one_compute_failure(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return RecordRunSummary(
                        status=RecordRunStatus.PARTIAL,
                        source_identity=None,
                        family_results=(
                            FamilyRunResult(
                                family_id="eeg_tail",
                                status=FamilyRunStatus.FAILED,
                                reason_code="COMPUTE:ValueError",
                            ),
                        ),
                    )
                return original_run_record(*args, **kwargs)

            with mock.patch(
                "extensions.batch_runner.run_record",
                side_effect=inject_one_compute_failure,
            ):
                summary = self._run_full(
                    root, inventory, source_root, stop_after=2
                )

            self.assertEqual(summary.executed_count, 2)
            self.assertIsNone(summary.safety_stop)
            self.assertEqual(summary.records[0].run.status, RecordRunStatus.PARTIAL)
            self.assertEqual(summary.records[1].run.status, RecordRunStatus.COMPLETE)
    def test_full_unavailable_source_snapshot_stops_before_compute(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = source_payload(
                site_id="I0002", record_id="record-00000", epochs=240
            )
            source_root, inventory = self._full_inventory(root, (payload,))
            with (
                mock.patch(
                    "extensions.batch_runner._source_snapshot",
                    side_effect=OSError("synthetic snapshot failure"),
                ),
                mock.patch("extensions.batch_runner.run_record") as compute,
            ):
                summary = self._run_full(
                    root, inventory, source_root, stop_after=2
                )
            compute.assert_not_called()
            self.assertEqual(summary.executed_count, 1)
            self.assertEqual(
                summary.safety_stop.reason_code, "SOURCE_SNAPSHOT_UNAVAILABLE"
            )

    def test_full_repeated_same_site_source_contract_error_is_systemic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = [
                source_payload(
                    site_id="OTHER", record_id=f"record-{index:05d}", epochs=240
                )
                for index in range(2)
            ]
            source_root, inventory = self._full_inventory(root, payloads)

            summary = self._run_full(
                root, inventory, source_root, stop_after=2
            )

            self.assertEqual(summary.executed_count, 2)
            self.assertEqual(
                summary.safety_stop.reason_code, "REPEATED_SOURCE_CONTRACT"
            )
            self.assertTrue(
                all(
                    item.run.status is RecordRunStatus.STOP_RECORD
                    for item in summary.records
                )
            )

    def test_full_startup_disk_guard_writes_stop_without_computing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root, inventory = self._full_inventory(root)
            auth = root / "FULL_EXTRACTION_APPROVED.txt"
            auth.write_text("FEATURELAB_FULL_EXTRACTION_APPROVED\n", encoding="utf-8")
            with (
                mock.patch.object(batch_runner, "FULL_AUTHORIZATION_PATH", auth),
                mock.patch.object(
                    batch_runner,
                    "_available_cache_bytes",
                    return_value=1024**3,
                ),
                mock.patch.object(
                    batch_runner, "_peak_rss_kb", return_value=100_000
                ),
                mock.patch("extensions.batch_runner.run_record") as compute,
            ):
                summary = run_batch(
                    inventory,
                    source_root,
                    root / "cache",
                    root / "full-checkpoint.json",
                    self.FULL_FAMILY_IDS,
                    {},
                    max_records=6600,
                    workers=1,
                    execution_mode="full",
                    full_authorization_file=auth,
                    expected_inventory_size=6600,
                )
            compute.assert_not_called()
            self.assertEqual(summary.executed_count, 0)
            self.assertEqual(summary.safety_stop.reason_code, "LOW_DISK_STARTUP")
            self.assertTrue((root / "full-checkpoint.json").is_file())

    def test_full_runtime_rss_guard_stops_after_persisting_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = [
                source_payload(
                    site_id="I0002", record_id=f"record-{index:05d}", epochs=240
                )
                for index in range(2)
            ]
            source_root, inventory = self._full_inventory(root, payloads)
            auth = root / "FULL_EXTRACTION_APPROVED.txt"
            auth.write_text("FEATURELAB_FULL_EXTRACTION_APPROVED\n", encoding="utf-8")
            with (
                mock.patch.object(batch_runner, "FULL_AUTHORIZATION_PATH", auth),
                mock.patch.object(
                    batch_runner,
                    "_available_cache_bytes",
                    side_effect=(3 * 1024**3, 3 * 1024**3),
                ),
                mock.patch.object(
                    batch_runner,
                    "_peak_rss_kb",
                    side_effect=(100_000, 2 * 1024 * 1024, 2 * 1024 * 1024),
                ),
            ):
                summary = run_batch(
                    inventory,
                    source_root,
                    root / "cache",
                    root / "full-checkpoint.json",
                    self.FULL_FAMILY_IDS,
                    {},
                    max_records=6600,
                    workers=1,
                    stop_after=2,
                    execution_mode="full",
                    full_authorization_file=auth,
                    expected_inventory_size=6600,
                )
            self.assertEqual(summary.executed_count, 1)
            self.assertEqual(summary.safety_stop.reason_code, "HIGH_RSS_RUNTIME")
            self.assertTrue(summary.safety_stop.safe_to_resume)




if __name__ == "__main__":
    unittest.main()
