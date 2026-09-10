import numpy as np

from per_epoch_features.per_epoch_extractor import PerEpochExtractor
from per_epoch_features.feature_extractor_algorithmic import AlgorithmicMixin
from per_epoch_features.eeg_sleep_features import eeg_segment_coherence


def v2(data, stages):
    out = dict(data)
    out["__extraction_mode__"] = "timegrid_v2"
    out["__sampling_frequencies__"] = {
        "stage_caisr": 1 / 30, "arousal_caisr": 2.0,
        "resp_caisr": 1.0, "limb_caisr": 1.0,
    }
    out["__aligned_stage_code__"] = np.asarray(stages, float)
    return out


def test_unknown_stage_preserves_epoch_and_event_time():
    stages = np.array([2, 9, 4, 1], float)
    arousal = np.zeros(240, float)
    arousal[120:180] = 1  # real time 60--90 seconds at 2 Hz
    data = v2({"stage_caisr": stages, "arousal_caisr": arousal}, stages)
    features, metadata = PerEpochExtractor()._extract_per_epoch_onehot(
        data, n_epochs=4, return_metadata=True)
    assert features.shape == (4, 13)
    assert features[1, :5].sum() == 0
    assert not metadata["stage_valid"][1]
    np.testing.assert_array_equal(features[:, 5], [0, 0, 1, 0])
    assert features[2, 3] == 1  # REM remains at 60--90 seconds


def test_unknown_splits_bouts_and_transitions():
    mixin = AlgorithmicMixin()
    stages = np.array([2, 9, 2], float)
    features = mixin.extract_algorithmic_annotations_features(
        v2({"stage_caisr": stages}, stages))
    assert features[21] == 0  # unknown boundary is not a transition
    assert features[26] == 2  # two N2 bouts, not one joined bout
    assert features[0] == 90  # TRT includes unknown exposure
    assert features[1] == 60  # TST includes only known sleep


def test_complete_grid_legacy_and_v2_match():
    stages = np.array([5, 2, 4, 1], float)
    arousal = np.zeros(240); arousal[20:24] = 1
    resp = np.zeros(120); resp[40:50] = 4
    limb = np.zeros(120); limb[75:80] = 2
    data = {"stage_caisr": stages, "arousal_caisr": arousal,
            "resp_caisr": resp, "limb_caisr": limb}
    legacy_onehot = PerEpochExtractor()._extract_per_epoch_onehot(data)
    v2_data = v2(data, stages)
    v2_onehot = PerEpochExtractor()._extract_per_epoch_onehot(v2_data, n_epochs=4)
    np.testing.assert_allclose(v2_onehot, legacy_onehot, rtol=0, atol=0)
    mixin = AlgorithmicMixin()
    np.testing.assert_allclose(
        mixin.extract_all_algorithmic_features(v2_data),
        mixin.extract_all_algorithmic_features(data), rtol=1e-6, atol=1e-6)


def test_quality_comes_from_flags_and_metadata_is_numerically_passive():
    rng = np.random.default_rng(7)
    eeg = rng.normal(scale=5, size=(6, 3000))
    plain, legacy_pvalues = eeg_segment_coherence(eeg, 100)
    with_meta, pvalues, metadata = eeg_segment_coherence(
        eeg, 100, return_metadata=True)
    np.testing.assert_allclose(with_meta, plain, rtol=0, atol=0)
    np.testing.assert_array_equal(pvalues, legacy_pvalues)
    assert np.all(pvalues == 0)  # documented legacy placeholder, not quality
    assert metadata["total_subsegment_count"].sum() > 0
    assert metadata["clean_subsegment_count"].sum() > 0
    assert np.all(metadata["clean_subsegment_count"] <= metadata["total_subsegment_count"])
