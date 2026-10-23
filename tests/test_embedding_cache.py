import tempfile
import unittest
from pathlib import Path

import h5py

from export_full_embeddings import shard_matches_checkpoint


class EmbeddingCacheFingerprintTests(unittest.TestCase):
    def test_existing_shard_requires_matching_checkpoint_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "shard.h5"
            sidecar = root / "shard.parquet"
            sidecar.touch()
            with h5py.File(shard, "w") as handle:
                handle.attrs["checkpoint_sha256"] = "old-checkpoint"

            self.assertFalse(
                shard_matches_checkpoint(shard, sidecar, "new-checkpoint")
            )
            self.assertTrue(
                shard_matches_checkpoint(shard, sidecar, "old-checkpoint")
            )


if __name__ == "__main__":
    unittest.main()
