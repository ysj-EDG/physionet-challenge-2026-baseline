import tempfile, unittest
from pathlib import Path
import numpy as np

from reproduction.siren_a.g4.models import fit_frozen_model, load_checkpoint, predict_scores, save_checkpoint
from reproduction.siren_a.g4.queries import build_queries


class ModelTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(7)
        self.X=rng.normal(size=(80,6)).astype(np.float64); self.X[::7,2]=np.nan
        self.y=np.tile(np.array([0,1],dtype=np.int8),40)
        self.ages=np.r_[np.full(40,55.),np.full(40,60.)]
        self.ids=tuple(f"f{i}" for i in range(6)); self.idx=np.arange(80,dtype=np.int64)
        self.query=build_queries(self.ages,self.y,self.idx,self.idx)

    def _roundtrip(self, model_type):
        trained=fit_frozen_model(model_type,self.X,self.y,self.ids,self.idx,self.query)
        scores=predict_scores(trained.model_type,trained.model,trained.adapter,self.X)
        self.assertTrue(np.isfinite(scores).all()); self.assertGreater(np.ptp(scores),0)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            files=save_checkpoint(p,trained)
            loaded=load_checkpoint(p,model_type,files)
            replay=predict_scores(model_type,loaded.model,loaded.adapter,self.X)
            np.testing.assert_allclose(scores,replay,rtol=0,atol=1e-12)
        return trained

    def test_lr_subset_adapter_and_probability(self):
        trained=self._roundtrip("lr")
        self.assertEqual(trained.adapter.fit_scope,"seed_subset")
        self.assertFalse(np.isnan(trained.train_matrix).any())
        self.assertEqual(trained.parameters["solver"],"lbfgs")

    def test_hgb_native_nan_and_fixed_parameters(self):
        trained=self._roundtrip("hgb")
        self.assertTrue(np.isnan(trained.train_matrix).any())
        self.assertEqual(trained.parameters["max_depth"],3)
        self.assertFalse(trained.parameters["early_stopping"])

    def test_lambdarank_groups_raw_score_and_fixed_parameters(self):
        trained=self._roundtrip("lambdarank")
        self.assertEqual(trained.parameters["objective"],"lambdarank")
        self.assertEqual(trained.parameters["lambdarank_truncation_level"],40)
        self.assertEqual(trained.parameters["label_gain"],[0,1])

    def test_unknown_model_rejected(self):
        with self.assertRaises(ValueError):
            fit_frozen_model("x",self.X,self.y,self.ids,self.idx,self.query)


if __name__ == "__main__": unittest.main()
