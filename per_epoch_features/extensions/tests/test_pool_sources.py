from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from extensions.pool_sources import AlignmentError, discover_alignment, parse_record_key

class PoolSourcesTests(unittest.TestCase):
    def test_parse_record_key_is_metadata_based_not_filename_order(self):
        self.assertEqual(parse_record_key('I0002/sub-I0002150000076_ses-1.npz'), ('I0002', 'sub-I0002150000076_ses-1'))
    def test_discover_alignment_rejects_duplicate_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old = root/'old'; new = root/'new'
            old.mkdir(); new.mkdir()
            (old/'private_index.json').write_text(json.dumps([{'cache_name':'cache_000000.npz','relative_path':'I0002/a.npz','site':'I0002'}]), encoding='utf-8')
            with self.assertRaises(AlignmentError):
                discover_alignment(old, new, family_ids=('eeg_tail',))

if __name__ == '__main__': unittest.main()
