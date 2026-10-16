"""Export a byte-exact, code-only snapshot from a fixed Git commit."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Iterable


SCHEMA = "siren_a_v1_code_snapshot_v1"
ROOT_ENTRYPOINTS = {
    "baseline.py", "create_labels.py", "evaluate_model.py", "extract_data2_timegrid_v2.py",
    "feature_scaling.py", "helper_code.py", "run_model.py", "team_code.py", "train_model.py",
}
ROOT_METADATA = {"Dockerfile", "LICENSE", "AUTHORS", "README.md", "requirements.txt", "channel_table.csv"}
FORBIDDEN_SUFFIXES = {
    ".edf", ".npz", ".npy", ".pkl", ".pickle", ".joblib", ".pt", ".pth", ".ckpt", ".h5", ".hdf5",
}
FORBIDDEN_PARTS = {"checkpoint", "checkpoints", "output", "outputs", "cache", "caches", "predictions", "logs",
                   ".building", "__pycache__"}


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(("git", *args), cwd=repo, check=True, capture_output=True).stdout


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tracked(repo: Path, commit: str) -> list[str]:
    resolved = _git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
    if resolved != commit:
        raise ValueError(f"source commit must be a full immutable SHA: {resolved}")
    return _git(repo, "ls-tree", "-r", "--name-only", commit).decode().splitlines()


def _blob(repo: Path, commit: str, path: str) -> bytes:
    return _git(repo, "show", f"{commit}:{path}")


def _blob_oid(repo: Path, commit: str, path: str) -> str:
    return _git(repo, "rev-parse", f"{commit}:{path}").decode().strip()


def _forbidden(path: str) -> bool:
    item = PurePosixPath(path)
    return item.suffix.lower() in FORBIDDEN_SUFFIXES or any(part.lower() in FORBIDDEN_PARTS for part in item.parts)


def _role(path: str) -> str:
    name = PurePosixPath(path).name.lower()
    if path.endswith(".py"):
        return "test" if name.startswith("test_") else "source"
    if "manifest" in name or "spec" in name:
        return "spec"
    if name.startswith("license") or name.startswith("authors"):
        return "license"
    if path.endswith((".csv", ".json", ".yaml", ".yml")):
        return "config"
    return "documentation"


def _initial_selection(paths: Iterable[str]) -> set[str]:
    selected: set[str] = set()
    for path in paths:
        item = PurePosixPath(path)
        if _forbidden(path):
            continue
        if path.endswith(".py") and (path.startswith("reproduction/siren_a/") or path.startswith("per_epoch_features/")):
            selected.add(path)
        elif path in ROOT_ENTRYPOINTS or path in ROOT_METADATA:
            selected.add(path)
        elif path == "per_epoch_features/README.md" or path == "per_epoch_features/requirements.txt":
            selected.add(path)
        elif path in {
            "reproduction/siren_a/docs/G1B_FEATURE_MANIFEST_FROZEN.csv",
            "reproduction/siren_a/docs/G1B_FEATURE_SPEC_FROZEN.md",
            "reproduction/siren_a/docs/BRANCH_A_SCOPE_V1.md",
        }:
            selected.add(path)
        elif path.startswith("reproduction/siren_a/docs/") and path.endswith(("_SPEC.md", "_PLAN.md")):
            selected.add(path)
    return selected


def _import_candidates(path: str, tree: ast.AST) -> set[str]:
    candidates: set[str] = set()
    package = PurePosixPath(path).parent.parts
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = list(package[: max(0, len(package) - node.level + 1)])
                if node.module:
                    base.extend(node.module.split("."))
                names.append(".".join(base))
            elif node.module:
                names.append(node.module)
        for name in names:
            if name:
                module_path = name.replace(".", "/")
                candidates.add(module_path + ".py")
                candidates.add(module_path + "/__init__.py")
    return candidates


def select_snapshot_files(repo: Path, commit: str) -> list[str]:
    tracked = set(_tracked(repo, commit))
    selected = _initial_selection(tracked)
    pending = list(selected)
    while pending:
        path = pending.pop()
        if not path.endswith(".py"):
            continue
        payload = _blob(repo, commit, path)
        tree = ast.parse(payload, filename=path)
        for candidate in _import_candidates(path, tree):
            if candidate in tracked and candidate not in selected and not _forbidden(candidate):
                selected.add(candidate)
                pending.append(candidate)
    for path in selected:
        if _forbidden(path):
            raise ValueError(f"forbidden asset selected: {path}")
        if path.endswith(".py"):
            ast.parse(_blob(repo, commit, path), filename=path)
    return sorted(selected)


def _manifest_bytes(rows: list[dict[str, object]]) -> bytes:
    with tempfile.TemporaryFile(mode="w+", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("repo_relative_path", "source_git_blob_oid", "n_bytes",
                                                        "content_sha256", "role"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.seek(0)
        return handle.read().encode("utf-8")


def _environment_text() -> str:
    packages = ("numpy", "scipy", "scikit-learn", "joblib", "lightgbm")
    lines = [f"python_executable={sys.executable}", f"python_version={sys.version.split()[0]}"]
    for package in packages:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "NOT_INSTALLED"
        lines.append(f"{package}={version}")
    return "\n".join(lines) + "\n"


def _snapshot_readme(repo: Path, source_commit: str) -> str:
    return f"""# SIREN Branch A v1 code snapshot

Source repository: `{repo}`
Source branch: `research/siren-a-stagepool-v1`
Source commit: `{source_commit}`

This is a code archive, not a data backup or a validated portable runtime package. It contains the G1–G5-C
source/tests, extraction code, frozen feature contract, run specifications, channel configuration and dependency
documentation selected from the fixed Git commit. The principal flow is per-epoch extraction → night cache →
rank256 preprocessing → model-input adaptation → G4 training → G5 rank ensemble/scoring.

No patient data, caches, model checkpoints, predictions or private identity mappings are included. External PSG,
metadata, private cache, model and prediction inputs remain required. Historical absolute paths and provenance
checks were preserved byte-for-byte and were not migrated. Continued experiments remain in the main worktree.
"""


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise ValueError(f"refusing to overwrite different file: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_snapshot(repo: Path, commit: str, destination: Path) -> dict[str, object]:
    complete_path = destination / "_snapshot/complete.json"
    manifest_path = destination / "_snapshot/manifest.csv"
    if not complete_path.is_file() or not manifest_path.is_file():
        raise ValueError("existing snapshot is incomplete or lacks provenance")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("schema") != SCHEMA or complete.get("source_commit") != commit or complete.get("status") != "PASS":
        raise ValueError("existing snapshot identity mismatch")
    manifest_payload = manifest_path.read_bytes()
    if _sha(manifest_payload) != complete.get("manifest_sha256"):
        raise ValueError("snapshot manifest hash mismatch")
    rows = _read_manifest(manifest_path)
    expected = select_snapshot_files(repo, commit)
    if [row["repo_relative_path"] for row in rows] != expected:
        raise ValueError("snapshot file list differs from source commit selection")
    metadata = {"_snapshot/README.md", "_snapshot/manifest.csv", "_snapshot/environment.txt",
                "_snapshot/complete.json"}
    expected_files = set(expected) | metadata
    actual_files = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
    if any(path.is_symlink() for path in destination.rglob("*")) or actual_files != expected_files:
        raise ValueError("snapshot contains extra, missing, or symlinked files")
    if (destination / "_snapshot/README.md").read_text(encoding="utf-8") != _snapshot_readme(repo, commit):
        raise ValueError("snapshot README content mismatch")
    if (destination / "_snapshot/environment.txt").read_text(encoding="utf-8") != _environment_text():
        raise ValueError("snapshot environment content mismatch")
    source_inodes = {}
    for row in rows:
        relative = row["repo_relative_path"]
        path = destination / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"snapshot file missing or symlinked: {relative}")
        payload = path.read_bytes()
        source = _blob(repo, commit, relative)
        if payload != source or _sha(payload) != row["content_sha256"] or len(payload) != int(row["n_bytes"]):
            raise ValueError(f"snapshot content mismatch: {relative}")
        current = repo / relative
        if current.is_file():
            source_inodes[relative] = current.stat().st_ino
            if current.stat().st_dev == path.stat().st_dev and current.stat().st_ino == path.stat().st_ino:
                raise ValueError(f"snapshot file is hard-linked to worktree: {relative}")
        if relative.endswith(".py"):
            ast.parse(payload, filename=relative)
    return {"schema": SCHEMA, "status": "PASS", "action": "VERIFY_AND_REUSE",
            "source_commit": commit, "file_count": len(rows), "n_bytes": sum(int(row["n_bytes"]) for row in rows),
            "manifest_sha256": complete["manifest_sha256"]}


def export_snapshot(repo: Path | str, source_commit: str, destination: Path | str) -> dict[str, object]:
    repo = Path(repo).resolve()
    destination = Path(destination)
    if destination.exists():
        return verify_snapshot(repo, source_commit, destination)
    paths = select_snapshot_files(repo, source_commit)
    blobs = {path: _blob(repo, source_commit, path) for path in paths}
    rows = [{"repo_relative_path": path, "source_git_blob_oid": _blob_oid(repo, source_commit, path),
             "n_bytes": len(blobs[path]), "content_sha256": _sha(blobs[path]), "role": _role(path)} for path in paths]
    manifest_payload = _manifest_bytes(rows)
    manifest_sha = _sha(manifest_payload)
    destination.mkdir(parents=True, exist_ok=False)
    for path in paths:
        _atomic_write(destination / path, blobs[path])
    _atomic_write(destination / "_snapshot/README.md", _snapshot_readme(repo, source_commit).encode("utf-8"))
    _atomic_write(destination / "_snapshot/manifest.csv", manifest_payload)
    _atomic_write(destination / "_snapshot/environment.txt", _environment_text().encode("utf-8"))
    complete = {"schema": SCHEMA, "status": "PASS", "source_commit": source_commit,
                "file_count": len(rows), "n_bytes": sum(len(value) for value in blobs.values()),
                "manifest_sha256": manifest_sha, "verification": "byte_exact_no_symlink_no_worktree_hardlink_ast_pass"}
    _atomic_write(destination / "_snapshot/complete.json",
                  (json.dumps(complete, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    result = verify_snapshot(repo, source_commit, destination)
    result["action"] = "BUILD_NEW"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--public-manifest", type=Path)
    args = parser.parse_args()
    result = export_snapshot(args.repo, args.source_commit, args.destination)
    if args.public_manifest:
        _atomic_write(args.public_manifest, (args.destination / "_snapshot/manifest.csv").read_bytes())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
