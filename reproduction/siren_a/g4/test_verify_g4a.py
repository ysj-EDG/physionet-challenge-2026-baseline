import json, tempfile, unittest
from pathlib import Path

from reproduction.siren_a.g4.verify_g4a import complete_unit, verify_unit, verify_model_parameters


class VerifyTests(unittest.TestCase):
    def test_model_parameter_drift_is_rejected(self):
        from reproduction.siren_a.g4.config import LR_PARAMS
        self.assertTrue(verify_model_parameters("lr", dict(LR_PARAMS), None))
        changed = dict(LR_PARAMS); changed["C"] = 2.0
        with self.assertRaises(ValueError): verify_model_parameters("lr", changed, None)

    def test_complete_unit_detects_content_change_and_extra_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"unit"; root.mkdir(); (root/"data.txt").write_text("ok")
            complete_unit(root,"test",{"source":"x"})
            self.assertTrue(verify_unit(root,"test",{"source":"x"}))
            (root/"data.txt").write_text("bad")
            with self.assertRaises(ValueError): verify_unit(root,"test",{"source":"x"})

    def test_identity_drift_and_incomplete_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"unit"; root.mkdir(); (root/"data.txt").write_text("ok")
            complete_unit(root,"test",{"source":"x"})
            with self.assertRaises(ValueError): verify_unit(root,"test",{"source":"y"})
            Path(root/"complete.json").unlink()
            with self.assertRaises(ValueError): verify_unit(root,"test",{"source":"x"})


if __name__ == "__main__": unittest.main()
