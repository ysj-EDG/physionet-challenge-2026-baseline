import tempfile
import unittest
from pathlib import Path
import numpy as np
from extensions.seed42_runner_v2 import (publish_prediction_unit_v2, UnitIdentityError, verify_prediction_unit_v2,
    model_identities_v2, prediction_identities_v2)

class Seed42RunnerV2Tests(unittest.TestCase):
    def test_frozen_unit_cardinality(self):
        self.assertEqual(len(model_identities_v2()), 30)
        self.assertEqual(len(prediction_identities_v2()), 90)

    def test_prediction_publish_resume_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"pred"
            payload={"test_global_indices":np.array([3,7],dtype=np.int64),"scores":np.array([0.2,0.8],dtype=np.float64)}
            metadata={"schema":"featurelab_v2_seed42_prediction_v1","identity":{"holdout_site":"I0002","protocol":"transductive","view":"V1","model_type":"lr","seed":42},"score_type":"positive_probability"}
            self.assertEqual(publish_prediction_unit_v2(path,payload,metadata)["status"],"BUILD_NEW")
            original=(path/"prediction.npz").read_bytes()
            self.assertEqual(publish_prediction_unit_v2(path,payload,metadata)["status"],"SKIP_VERIFIED")
            self.assertEqual((path/"prediction.npz").read_bytes(),original)
            bad=dict(metadata); bad["identity"]={**metadata["identity"],"view":"V2"}
            with self.assertRaises(UnitIdentityError): publish_prediction_unit_v2(path,payload,bad)
            self.assertTrue(verify_prediction_unit_v2(path,metadata))

    def test_corrupt_prediction_is_rejected_without_replacement(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"pred"
            payload={"test_global_indices":np.array([3],dtype=np.int64),"scores":np.array([0.2],dtype=np.float64)}
            metadata={"schema":"featurelab_v2_seed42_prediction_v1","identity":{"holdout_site":"I0002","protocol":"transductive","view":"V1","model_type":"lr","seed":42},"score_type":"positive_probability"}
            publish_prediction_unit_v2(path,payload,metadata)
            data=(path/"prediction.npz").read_bytes(); (path/"prediction.npz").write_bytes(data+b"bad")
            with self.assertRaises(UnitIdentityError): publish_prediction_unit_v2(path,payload,metadata)
            self.assertNotEqual((path/"prediction.npz").read_bytes(),data)

if __name__=="__main__": unittest.main()
