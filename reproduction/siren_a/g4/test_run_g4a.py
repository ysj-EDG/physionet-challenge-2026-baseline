import tempfile, unittest
from pathlib import Path

from reproduction.siren_a.g4.config import MODEL_TYPES, PROTOCOLS, TRAIN_SITES, VIEWS
from reproduction.siren_a.g4.run_g4a import atomic_json_no_clobber, model_identities, prediction_identities


class RunnerTests(unittest.TestCase):
    def test_identity_cardinality_and_reuse(self):
        models=model_identities()
        predictions=prediction_identities()
        self.assertEqual(len(models),30); self.assertEqual(len(set(models)),30)
        self.assertEqual(len(predictions),90); self.assertEqual(len(set(predictions)),90)
        self.assertEqual(set(m[0] for m in models),set(TRAIN_SITES))
        self.assertEqual(set(m[2] for m in models),set(VIEWS))
        self.assertEqual(set(m[3] for m in models),set(MODEL_TYPES))
        self.assertEqual(set(p[1] for p in predictions),set(PROTOCOLS))

    def test_no_clobber(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"x.json"
            atomic_json_no_clobber(path,{"a":1})
            with self.assertRaises(FileExistsError): atomic_json_no_clobber(path,{"a":2})

    def test_no_later_seed(self):
        self.assertTrue(all(model[1] == 42 for model in model_identities()))


if __name__ == "__main__": unittest.main()
