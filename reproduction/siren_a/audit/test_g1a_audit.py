import numpy as np

from g1a_audit import (
    ALL_STAGE_CODES,
    NREM_STAGE_CODES,
    STAGE_CODES,
    build_draft_manifest,
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
