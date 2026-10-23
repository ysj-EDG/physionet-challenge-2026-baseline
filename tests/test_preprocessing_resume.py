import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from full_training_data import process_record


class PreprocessingResumeTests(unittest.TestCase):
    def test_process_record_reuses_matching_existing_caches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "canonical.h5"
            annotation = root / "annotation.h5"
            record_id = "record-1"
            modalities = ("eeg", "eog", "ecg", "resp", "spo2", "emg")
            with h5py.File(cache, "w") as handle:
                handle.attrs["record_id"] = record_id
                handle.attrs["complete_epoch_count"] = 2
                signals = handle.create_group("signals")
                quality = handle.create_group("quality")
                present = quality.create_group("channel_present")
                for modality in modalities:
                    present.create_dataset(modality, data=np.ones(1, dtype=bool))
                    signals.create_dataset(modality, data=np.zeros((2, 1, 1), dtype=np.int16))
            with h5py.File(annotation, "w") as handle:
                handle.attrs["record_id"] = record_id
                handle.attrs["complete_epoch_count"] = 2
                handle.create_group("annotations")
            payload = {
                "record_id": record_id,
                "cache_path": str(cache),
                "annotation_path": str(annotation),
                "psg_path": str(root / "missing.edf"),
                "SiteID": "I0002",
                "human_path": "",
                "caisr_path": "",
            }
            with patch(
                "full_training_data.training_features.build_psg_cache",
                side_effect=AssertionError("existing cache was not reused"),
            ), patch(
                "full_training_data.caisr_feature_extractor.extract_caisr_features_from_hdf5",
                return_value={"feature_status": "missing"},
            ):
                result = process_record(payload)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["record_id"], record_id)
            self.assertEqual(result["complete_epoch_count"], 2)


if __name__ == "__main__":
    unittest.main()
