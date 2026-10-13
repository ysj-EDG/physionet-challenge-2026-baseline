import csv
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import g1a_audit as audit

from g1a_audit import (
    ALL_STAGE_CODES,
    NREM_STAGE_CODES,
    STAGE_CODES,
    build_draft_manifest,
    aggregate_private_stage_support,
    _band_index,
    _beta_index,
    _p0_index,
    compute_stage_masks,
    eeg_validity_mask,
    intersected_count,
    spatial_candidate_rows,
)


def test_stage_masks_exclude_invalid_and_define_nrem_and_all_stage():
    codes = np.array([5, 4, 3, 2, 1, 0, 9, np.nan], dtype=float)
    valid = np.array([True, True, True, True, True, True, True, True])
    masks = compute_stage_masks(codes, valid)

    assert sum(masks["Wake"]) == 1
    assert sum(masks["REM"]) == 1
    assert sum(masks["N1"]) == 1
    assert sum(masks["N2"]) == 1
    assert sum(masks["N3"]) == 1
    assert sum(masks["NREM"]) == 3
    assert sum(masks["All-stage"]) == 5
    assert set(STAGE_CODES) == {"Wake", "REM", "N1", "N2", "N3", "NREM", "All-stage"}
    assert set(NREM_STAGE_CODES) == {1, 2, 3}
    assert set(ALL_STAGE_CODES) == {1, 2, 3, 4, 5}


def test_stage_masks_require_stage_valid():
    codes = np.array([5, 4, 3, 2, 1], dtype=float)
    valid = np.array([True, False, True, False, True])
    masks = compute_stage_masks(codes, valid)
    assert sum(masks["Wake"]) == 1
    assert sum(masks["REM"]) == 0
    assert sum(masks["NREM"]) == 2
    assert sum(masks["All-stage"]) == 3


def test_eeg_validity_uses_channel_availability_and_clean_count_not_global_mask():
    channel_available = np.array([True, True, False, True, True, True])
    clean = np.array(
        [
            [1, 0, 10, 1, 0, 1],
            [0, 1, 10, 0, 1, 0],
        ],
        dtype=int,
    )
    global_ecg_mask = np.array([False, True])
    result = eeg_validity_mask(channel_available, clean)

    assert result.tolist() == [
        [True, False, False, True, False, True],
        [False, True, False, False, True, False],
    ]
    assert not np.array_equal(result.any(axis=1), global_ecg_mask)


def test_common_counts_use_same_epoch_intersection_not_minimum_marginals():
    valid = np.array(
        [
            [True, False, True, False, True, True],
            [False, True, True, True, False, True],
            [True, True, False, True, True, False],
        ]
    )
    assert intersected_count(valid, [0, 1]) == 1
    assert intersected_count(valid, [0, 1, 4, 5]) == 0
    assert min(valid[:, 0].sum(), valid[:, 1].sum()) == 2


def test_spatial_candidate_rows_are_exactly_36_and_use_only_approved_formulas():
    rows = spatial_candidate_rows()
    assert len(rows) == 36
    assert sum(r["family"] == "spatial_asymmetry_candidate" for r in rows) == 24
    assert sum(r["family"] == "spatial_ap_gradient_candidate" for r in rows) == 12
    assert {r["stage_group"] for r in rows if r["family"] == "spatial_asymmetry_candidate"} == {"N2", "REM"}
    assert {r["stage_group"] for r in rows if r["family"] == "spatial_ap_gradient_candidate"} == {"N2", "N3", "REM"}
    assert all(r["specification_status"] == "PENDING_G1B_REVIEW" for r in rows)
    assert all("logPbandPbeta + logPbeta - logP0" in r["formula"] for r in rows)


def test_draft_manifest_has_827_rows_contiguous_view_positions_and_pending_demo5():
    rows = build_draft_manifest()
    assert len(rows) == 827
    for view_id in ["V1", "V2", "V3", "V4", "V5"]:
        view_rows = [r for r in rows if r["view_id"] == view_id]
        assert [int(r["view_position"]) for r in view_rows] == list(range(len(view_rows)))
    demo_rows = [r for r in rows if r["family"] == "demographic_candidate"]
    assert len(demo_rows) == 25
    assert len({r["unique_feature_id"] for r in demo_rows}) == 5
    assert all(r["specification_status"] == "PENDING_G1B_REVIEW" for r in demo_rows)
    assert not any(r["family"] == "demographic_candidate" and r["specification_status"] == "FROZEN" for r in rows)


def test_spatial_candidate_source_indices_are_exact_for_all_36_rows():
    rows = spatial_candidate_rows()
    bands = ("delta", "theta", "alpha", "sigma")
    four = ("F3", "F4", "O1", "O2")
    for row in rows:
        indices = [int(value) for value in row["source_indices"].split(";")]
        assert "logPbandPbeta + logPbeta - logP0" in row["formula"]
        if row["family"] == "spatial_asymmetry_candidate":
            assert len(indices) == 6
            left, right = row["channel"].split("-")
            band = row["feature_name"].rsplit("_", 2)[-2]
            assert indices == [
                _band_index(left, band), _band_index(right, band),
                _beta_index(left), _beta_index(right),
                _p0_index(left), _p0_index(right),
            ]
        else:
            assert row["family"] == "spatial_ap_gradient_candidate"
            assert len(indices) == 12
            band = row["feature_name"].rsplit("_", 2)[-2]
            expected = (
                [_band_index(ch, band) for ch in four]
                + [_beta_index(ch) for ch in four]
                + [_p0_index(ch) for ch in four]
            )
            assert indices == expected
            assert all(ch not in row["source_indices"] for ch in ("C3", "C4"))


def test_public_stage_support_omits_record_level_rows():
    from g1a_audit import STAGE_ORDER

    fields = ["section", "site_id", "stage_group", "threshold", "metric", "count", "total", "proportion", "notes"]
    with TemporaryDirectory() as temp:
        private = Path(temp) / "private.csv"
        public = Path(temp) / "public.csv"
        with private.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for _ in range(2):
                for stage in STAGE_ORDER:
                    for metric in ("F3_effective_epochs", "F4_effective_epochs", "C3_effective_epochs", "C4_effective_epochs", "O1_effective_epochs", "O2_effective_epochs"):
                        writer.writerow({"section": "lead_support", "site_id": "I0002", "stage_group": stage, "threshold": "", "metric": metric, "count": 10, "total": 1, "proportion": "", "notes": "detail"})
                    for metric in ("pair_F3-F4", "pair_C3-C4", "pair_O1-O2", "F3_F4_O1_O2", "all_six"):
                        writer.writerow({"section": "intersection_support", "site_id": "I0002", "stage_group": stage, "threshold": "", "metric": metric, "count": 10, "total": 1, "proportion": "", "notes": "detail"})
        summary = aggregate_private_stage_support(private, public)
        with public.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert summary["private_records"] == 2
        assert rows
        assert all(row["section"] not in {"lead_support", "intersection_support"} for row in rows)
        assert all(row["total"] != "1" for row in rows)
        assert any(row["metric"] == "EEG54_six_lead" for row in rows)


def _support_record(site="I0002", lead_count=10, six_lead_count=10, n1_count=None):
    stage_counts = {}
    for stage in ("Wake", "REM", "N1", "N2", "N3", "NREM", "All-stage"):
        count = n1_count if stage == "N1" and n1_count is not None else lead_count
        stage_counts[stage] = {
            "raw": count,
            "effective": count,
            "lead": [count] * 6,
            "pair": {"F3-F4": count, "C3-C4": count, "O1-O2": count},
            "four_lead": count,
            "six_lead": six_lead_count if stage != "N1" else count,
        }
    return {"site_id": site, "stage_counts": stage_counts}


def test_model_eeg_required_support_excludes_standalone_n1():
    record = _support_record(n1_count=0)
    support = audit.required_feature_support(record, threshold=5)
    assert audit.AUDIT_STAGE_ORDER == ("Wake", "REM", "N1", "N2", "N3", "NREM", "All-stage")
    assert audit.MODEL_EEG_STAGE_ORDER == ("Wake", "REM", "N2", "N3", "NREM", "All-stage")
    assert support["EEG54_individual_leads_all_valid"]
    assert support["all_required_EEG_spatial_actual"]
    assert support["all_required_EEG_spatial_joint6_conservative"]


def test_eeg648_support_uses_individual_leads_not_joint_six_epoch_count():
    record = _support_record(lead_count=10, six_lead_count=0)
    support = audit.required_feature_support(record, threshold=5)
    assert support["EEG54_individual_leads_all_valid"]
    assert support["Spatial36_all_combinations_valid"]
    assert support["all_required_EEG_spatial_actual"]
    assert not support["all_required_EEG_spatial_joint6_conservative"]


def test_spatial_required_support_uses_same_epoch_intersection_counts():
    record = _support_record()
    record["stage_counts"]["REM"]["pair"]["F3-F4"] = 4
    support = audit.required_feature_support(record, threshold=5)
    assert support["EEG54_individual_leads_all_valid"]
    assert not support["Spatial36_all_combinations_valid"]
    assert not support["all_required_EEG_spatial_actual"]
    assert not support["all_required_EEG_spatial_joint6_conservative"]


def test_public_aggregate_actual_ge_conservative_and_conservative_complements_legacy_metric():
    public = Path(__file__).with_name("g1a_stage_support.csv")
    with public.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    aggregate = {
        (row["site_id"], int(row["threshold"]), row["metric"]): int(row["count"])
        for row in rows if row["section"] == "aggregate_stage_support"
    }
    legacy_insufficient = {
        (row["site_id"], int(row["threshold"])): (int(row["count"]), int(row["total"]))
        for row in rows
        if row["section"] == "candidate_feature_support"
        and row["metric"] == "at_least_one_required_feature_insufficient"
    }
    expected_conservative = {
        ("I0002", 5): 239, ("I0002", 10): 228, ("I0002", 20): 206,
        ("I0006", 5): 874, ("I0006", 10): 805, ("I0006", 20): 704,
        ("S0001", 5): 4429, ("S0001", 10): 4274, ("S0001", 20): 4012,
        ("ALL", 5): 5542, ("ALL", 10): 5307, ("ALL", 20): 4922,
    }
    for site in ("ALL", "I0002", "I0006", "S0001"):
        for threshold in audit.THRESHOLDS:
            actual = aggregate[(site, threshold, "all_required_EEG_spatial_actual")]
            conservative = aggregate[(site, threshold, "all_required_EEG_spatial_joint6_conservative")]
            insufficient, total = legacy_insufficient[(site, threshold)]
            assert actual >= conservative
            assert conservative == total - insufficient
    for key, expected in expected_conservative.items():
        assert aggregate[(key[0], key[1], "all_required_EEG_spatial_joint6_conservative")] == expected


def test_full_audit_stage_output_route_keeps_details_private_and_refuses_overwrite():
    with TemporaryDirectory() as temp:
        audit_dir = Path(temp) / "audit"
        summary = audit.write_stage_support_outputs([_support_record()], audit_dir)
        private = audit_dir / "private/g1a_stage_support_full_record.csv"
        public = audit_dir / "g1a_stage_support.csv"
        with private.open(newline="", encoding="utf-8") as handle:
            private_rows = list(csv.DictReader(handle))
        with public.open(newline="", encoding="utf-8") as handle:
            public_rows = list(csv.DictReader(handle))
        assert summary["private_records"] == 1
        assert any(row["section"] == "lead_support" for row in private_rows)
        assert any(row["section"] == "intersection_support" for row in private_rows)
        assert all(row["section"] not in {"lead_support", "intersection_support"} for row in public_rows)
        before = private.read_bytes()
        try:
            audit.write_stage_support_outputs([_support_record()], audit_dir)
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing private audit detail must not be overwritten")
        assert private.read_bytes() == before


def test_run_audit_refuses_existing_private_detail_before_scanning_npz():
    with TemporaryDirectory() as temp:
        output_root = Path(temp)
        private = output_root / "audit/private/g1a_stage_support_full_record.csv"
        private.parent.mkdir(parents=True)
        private.write_text("protected\n", encoding="utf-8")
        try:
            audit.run_audit(
                npz_root=output_root / "missing_npz",
                demographics_path=output_root / "missing_demographics.csv",
                output_root=output_root,
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("run_audit must refuse an existing private detail file before scanning")
        assert private.read_text(encoding="utf-8") == "protected\n"
        assert not (output_root / "audit/g1a_stage_support.csv").exists()


def test_preserved_private_stage_support_sha256():
    private = audit.PRIVATE_STAGE_SUPPORT
    if not private.exists():
        return
    digest = hashlib.sha256(private.read_bytes()).hexdigest()
    assert digest == "b655ef2474084305ccc81f15328922615a4a5e94faae93deb6b09f1e012c7cd6"
