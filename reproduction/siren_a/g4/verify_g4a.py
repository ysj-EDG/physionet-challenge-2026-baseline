"""Independent integrity primitives and full G4-A verifier."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(root: Path):
    result = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            raise ValueError("symlink forbidden")
        if path.is_file() and path.name != "complete.json":
            result[path.relative_to(root).as_posix()] = sha256_file(path)
    return result


def complete_unit(root: Path, schema: str, identity: Mapping[str, object]) -> None:
    root = Path(root)
    target = root / "complete.json"
    if target.exists():
        raise FileExistsError(target)
    payload = {"schema": schema, "identity": dict(identity), "files": _files(root)}
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=root, prefix="complete.", suffix=".tmp",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def verify_unit(root: Path, schema: str, identity: Mapping[str, object]) -> bool:
    root = Path(root)
    complete = root / "complete.json"
    if root.is_symlink() or not root.is_dir() or not complete.is_file() or complete.is_symlink():
        raise ValueError("unit incomplete")
    payload = json.loads(complete.read_text(encoding="utf-8"))
    if payload.get("schema") != schema or payload.get("identity") != dict(identity):
        raise ValueError("unit identity mismatch")
    if payload.get("files") != _files(root):
        raise ValueError("unit content mismatch")
    return True


def verify_model_parameters(model_type: str, observed: Mapping[str, object], truncation_level):
    from .config import HGB_PARAMS, LGBM_PARAMS, LR_PARAMS
    if model_type == "lr":
        expected = dict(LR_PARAMS)
    elif model_type == "hgb":
        expected = dict(HGB_PARAMS)
    elif model_type == "lambdarank":
        expected = dict(LGBM_PARAMS)
        expected["lambdarank_truncation_level"] = int(truncation_level)
    else:
        raise ValueError("unknown model type")
    if dict(observed) != expected:
        raise ValueError("frozen model parameter drift")
    return True

def main():
    from .run_g4a import default_config, verify_all
    result = verify_all(default_config())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
