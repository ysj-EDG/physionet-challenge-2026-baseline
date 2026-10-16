import csv
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


from reproduction.siren_a.pipeline.export_v1_snapshot import _atomic_write, export_snapshot


def run(*args, cwd):
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True).stdout


class SnapshotExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        run("git", "init", "-q", cwd=self.repo)
        run("git", "config", "user.email", "test@example.invalid", cwd=self.repo)
        run("git", "config", "user.name", "Snapshot Test", cwd=self.repo)
        files = {
            "reproduction/siren_a/pipeline/module.py": b"from helper_code import useful\n",
            "reproduction/siren_a/pipeline/test_module.py": b"assert True\n",
            "reproduction/siren_a/docs/G1B_FEATURE_MANIFEST_FROZEN.csv": b"feature_id\nf1\n",
            "reproduction/siren_a/docs/G1B_FEATURE_SPEC_FROZEN.md": b"# Frozen\n",
            "reproduction/siren_a/docs/G2A_SMOKE_SPEC.md": b"# Spec\n",
            "reproduction/siren_a/docs/G2A_SMOKE_REPORT.md": b"patient result\n",
            "per_epoch_features/extractor.py": b"VALUE = 1\n",
            "channel_table.csv": b"name,type\nEEG,eeg\n",
            "evaluate_model.py": b"def score(): return 1\n",
            "helper_code.py": b"def useful(): return 1\n",
            "README.md": b"# Test repo\n",
            "LICENSE": b"test license\n",
            "requirements.txt": b"numpy\n",
            "cache/patient.npz": b"forbidden",
            "outputs/predictions.csv": b"patient_id,score\n1,.5\n",
        }
        for relative, payload in files.items():
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        run("git", "add", ".", cwd=self.repo)
        run("git", "commit", "-qm", "source", cwd=self.repo)
        self.commit = run("git", "rev-parse", "HEAD", cwd=self.repo).decode().strip()
        (self.repo / "reproduction/siren_a/pipeline/new_after_commit.py").write_text("NEW = 1\n")
        self.destination = self.root / "snapshot"

    def tearDown(self):
        self.temp.cleanup()

    def test_exports_exact_commit_and_approved_roles_only(self):
        result = export_snapshot(self.repo, self.commit, self.destination)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual((self.destination / "helper_code.py").read_bytes(), b"def useful(): return 1\n")
        self.assertFalse((self.destination / "cache/patient.npz").exists())
        self.assertFalse((self.destination / "outputs/predictions.csv").exists())
        self.assertFalse((self.destination / "reproduction/siren_a/docs/G2A_SMOKE_REPORT.md").exists())
        self.assertFalse((self.destination / "reproduction/siren_a/pipeline/new_after_commit.py").exists())
        with (self.destination / "_snapshot/manifest.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(rows)
        self.assertEqual({"repo_relative_path", "source_git_blob_oid", "n_bytes", "content_sha256", "role"},
                         set(rows[0]))
        for row in rows:
            source = run("git", "show", f"{self.commit}:{row['repo_relative_path']}", cwd=self.repo)
            self.assertEqual((self.destination / row["repo_relative_path"]).read_bytes(), source)

    def test_identical_destination_reuses_and_changed_destination_is_rejected(self):
        first = export_snapshot(self.repo, self.commit, self.destination)
        second = export_snapshot(self.repo, self.commit, self.destination)
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
        self.assertEqual(second["action"], "VERIFY_AND_REUSE")
        (self.destination / "helper_code.py").write_text("tampered\n")
        with self.assertRaises(ValueError):
            export_snapshot(self.repo, self.commit, self.destination)

    def test_extra_file_in_existing_snapshot_is_rejected(self):
        export_snapshot(self.repo, self.commit, self.destination)
        extra = self.destination / "unexpected/patient.npz"
        extra.parent.mkdir()
        extra.write_bytes(b"private")
        with self.assertRaises(ValueError):
            export_snapshot(self.repo, self.commit, self.destination)

    def test_changed_snapshot_metadata_is_rejected(self):
        for relative in ("_snapshot/README.md", "_snapshot/environment.txt"):
            with self.subTest(relative=relative):
                destination = self.root / relative.replace("/", "_")
                export_snapshot(self.repo, self.commit, destination)
                (destination / relative).write_text("tampered\n")
                with self.assertRaises(ValueError):
                    export_snapshot(self.repo, self.commit, destination)

    def test_atomic_write_never_clobbers_a_racing_publisher(self):
        target = self.root / "atomic.txt"
        real_link = __import__("os").link

        def publish_racer(source, destination):
            Path(destination).write_bytes(b"racer")
            return real_link(source, destination)

        with patch("reproduction.siren_a.pipeline.export_v1_snapshot.os.link", side_effect=publish_racer):
            with self.assertRaises(FileExistsError):
                _atomic_write(target, b"ours")
        self.assertEqual(target.read_bytes(), b"racer")


if __name__ == "__main__":
    unittest.main()
