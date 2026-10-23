"""Private resumable runner and post-hoc diagnostics for V5 family LOO.

The public-facing helpers in this module are deterministic and data-shape
checked.  The full execution entry point is intentionally explicit so that
it can never silently fall back to another view or rebuild frozen inputs.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Mapping, Sequence
from dataclasses import replace

import numpy as np

from .v5_family_loo import CONFIGS, _canonical


class UnitIdentityError(RuntimeError):
    pass


PRIVATE_FIELD_NAMES = {
    "record_id", "patient_id", "label", "labels", "age", "ages", "raw_scores",
    "prediction", "predictions", "private_path", "bootstrap_multiplicity",
}


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_array(left: np.ndarray, right: np.ndarray) -> bool:
    left = np.asarray(left); right = np.asarray(right)
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    return np.array_equal(left, right, equal_nan=True) if left.dtype.kind in "fc" else np.array_equal(left, right)


def _payload_summary(path: Path) -> dict[str, object]:
    return {item.name: {"size": item.stat().st_size, "sha256": _file_sha(item)}
            for item in sorted(Path(path).iterdir())
            if item.is_file() and item.name != "complete.json"}


def _validate_unit(path: Path, payload: Mapping[str, np.ndarray], metadata: Mapping[str, object]) -> None:
    path = Path(path)
    if not (path.is_dir() and (path / "payload.npz").is_file() and (path / "metadata.json").is_file() and (path / "complete.json").is_file()):
        raise UnitIdentityError(f"incomplete unit: {path}")
    saved = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if saved != json.loads(_canonical(metadata).decode("utf-8")):
        raise UnitIdentityError("unit metadata identity mismatch")
    complete = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    if complete.get("status") != "COMPLETE" or complete.get("payload_summary") != _payload_summary(path):
        raise UnitIdentityError("unit completion digest mismatch")
    with np.load(path / "payload.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(payload):
            raise UnitIdentityError("payload keys mismatch")
        for key, value in payload.items():
            if not _same_array(np.asarray(archive[key]), np.asarray(value)):
                raise UnitIdentityError(f"payload drift: {key}")


def publish_npz_unit(path: Path, payload: Mapping[str, np.ndarray], metadata: Mapping[str, object]) -> str:
    """Atomically publish a private NPZ unit without overwriting any target."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        _validate_unit(path, payload, metadata)
        return "SKIP_VERIFIED"
    path.parent.mkdir(parents=True, exist_ok=True)
    building = Path(tempfile.mkdtemp(prefix=f".{path.name}.building-", dir=str(path.parent)))
    try:
        with (building / "payload.npz").open("wb") as handle:
            np.savez_compressed(handle, **{key: np.asarray(value) for key, value in payload.items()})
            handle.flush(); os.fsync(handle.fileno())
        (building / "metadata.json").write_bytes(_canonical(metadata) + b"\n")
        (building / "complete.json").write_bytes(_canonical({
            "schema": "featurelab_v2_v5_family_loo_unit_v1",
            "status": "COMPLETE",
            "payload_summary": _payload_summary(building),
        }) + b"\n")
        _validate_unit(building, payload, metadata)
        if path.exists() or path.is_symlink():
            raise UnitIdentityError("unit target appeared during publish")
        os.rename(building, path)
        _validate_unit(path, payload, metadata)
        return "BUILD_NEW"
    except Exception:
        if building.exists():
            shutil.rmtree(building)
        raise


def comparison_name(config_id: str, baseline: str) -> str:
    if config_id not in CONFIGS or baseline not in {"E0", "E2"}:
        raise ValueError("invalid LOO comparison")
    return f"{config_id}-{baseline}"


def bootstrap_delta(loo: np.ndarray, baseline: np.ndarray, *, seed: int, n_bootstrap: int) -> np.ndarray:
    a = np.asarray(loo, dtype=np.float64); b = np.asarray(baseline, dtype=np.float64)
    if a.ndim != 1 or b.shape != a.shape or len(a) == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("bootstrap vectors must be finite and aligned")
    if int(n_bootstrap) <= 0:
        raise ValueError("n_bootstrap must be positive")
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(a), size=(int(n_bootstrap), len(a)))
    return (a[draws].mean(axis=1) - b[draws].mean(axis=1)).astype(np.float64)


def validate_public_payload(payload: Mapping[str, object]) -> bool:
    if not isinstance(payload, Mapping):
        raise ValueError("public payload must be a mapping")
    lower = {str(key).lower() for key in payload}
    forbidden = {key for key in lower if key in PRIVATE_FIELD_NAMES or any(token in key for token in ("patient", "record_id", "raw_prediction", "private_path"))}
    if forbidden:
        raise ValueError(f"private fields in public payload: {sorted(forbidden)}")
    return True




# The following execution helpers intentionally import the frozen data/model
# modules lazily.  Importing this module for unit tests therefore never loads
# labels, demographics, or rank arrays.
def project_fold_for_config(fold, rows):
    from .v5_family_loo import project_matrix
    train_values, train_mask = project_matrix(fold.X_train, fold.train_valid_mask, rows)
    test_values, test_mask = project_matrix(fold.X_test, fold.test_valid_mask, rows)
    return replace(fold, X_train=train_values, X_test=test_values,
                   train_valid_mask=train_mask, test_valid_mask=test_mask,
                   feature_ids=tuple(row.feature_id for row in rows))


def _tree_summary(path: Path) -> dict[str, object]:
    out = {}
    for item in sorted(Path(path).rglob('*')):
        if item.is_file() and item.name != 'complete.json':
            out[str(item.relative_to(path))] = {'size': item.stat().st_size, 'sha256': _file_sha(item)}
    return out


def _publish_model(path: Path, trained, metadata: Mapping[str, object]) -> str:
    from reproduction.siren_a.g4.models import load_checkpoint, predict_scores, save_checkpoint
    path = Path(path)
    if path.exists() or path.is_symlink():
        if not (path.is_dir() and (path / 'metadata.json').is_file() and (path / 'complete.json').is_file()):
            raise UnitIdentityError(f'incomplete model unit: {path}')
        saved = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
        if any(saved.get(k) != v for k, v in metadata.items()):
            raise UnitIdentityError('model identity mismatch')
        complete = json.loads((path / 'complete.json').read_text(encoding='utf-8'))
        if complete.get('status') != 'COMPLETE' or complete.get('tree_summary') != _tree_summary(path):
            raise UnitIdentityError('model digest mismatch')
        return 'SKIP_VERIFIED'
    path.parent.mkdir(parents=True, exist_ok=True)
    building = Path(tempfile.mkdtemp(prefix=f'.{path.name}.building-', dir=str(path.parent)))
    try:
        files = save_checkpoint(building, trained)
        saved = dict(metadata); saved['checkpoint_files'] = dict(files)
        (building / 'metadata.json').write_bytes(_canonical(saved) + b'\n')
        (building / 'complete.json').write_bytes(_canonical({'schema':'featurelab_v2_v5_family_loo_model_v1','status':'COMPLETE','tree_summary':_tree_summary(building)}) + b'\n')
        if path.exists() or path.is_symlink():
            raise UnitIdentityError('model target appeared during publish')
        os.rename(building, path)
        return 'BUILD_NEW'
    except Exception:
        if building.exists(): shutil.rmtree(building)
        raise


def _validate_prediction_unit_identity(path: Path, expected: Mapping[str, object]) -> str:
    """Validate an existing prediction unit without loading a model or recomputing scores."""
    path = Path(path)
    if not (path.is_dir() and (path / "payload.npz").is_file() and (path / "metadata.json").is_file() and (path / "complete.json").is_file()):
        raise UnitIdentityError(f"incomplete prediction unit: {path}")
    saved = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    for key, value in expected.items():
        if key != "prediction_sha256" and saved.get(key) != value:
            raise UnitIdentityError(f"prediction identity mismatch: {key}")
    complete = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    if complete.get("status") != "COMPLETE" or complete.get("payload_summary") != _payload_summary(path):
        raise UnitIdentityError("prediction completion digest mismatch")
    with np.load(path / "payload.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"test_global_indices", "scores"}:
            raise UnitIdentityError("prediction payload keys mismatch")
        indices = np.asarray(archive["test_global_indices"])
        scores = np.asarray(archive["scores"])
        if indices.dtype != np.dtype("int64") or scores.dtype != np.dtype("float64") or indices.ndim != 1 or scores.ndim != 1 or len(indices) != len(scores):
            raise UnitIdentityError("prediction payload schema mismatch")
        if saved.get("test_indices_sha256") != _array_sha(indices) or saved.get("prediction_sha256") != _array_sha(scores):
            raise UnitIdentityError("prediction payload digest mismatch")
    return "SKIP_VERIFIED"


def _load_model(path: Path, model_type: str):
    from reproduction.siren_a.g4.models import load_checkpoint
    metadata = json.loads((Path(path) / 'metadata.json').read_text(encoding='utf-8'))
    return load_checkpoint(Path(path), model_type, metadata['checkpoint_files']), metadata


def _array_sha(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256(); digest.update(str(array.shape).encode()); digest.update(array.dtype.str.encode()); digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def _sample_query_bridge(prepared, old_root: Path) -> bool:
    """Compare all deterministic sample/query arrays against the frozen root."""
    seed = int(prepared.seed); train_site = prepared.fold.train_site
    if seed == 42:
        candidates = [Path(old_root) / 'seed42_smoke_v1',
                      Path('/database/home/gaohaojie/workspace/python-example-2026-siren-a-g4a-private/seed42_smoke_v1')]
        base = next((item for item in candidates if (item / 'samples' / f'{train_site}_seed42').is_dir()), None)
        if base is None:
            raise UnitIdentityError('seed42 frozen sample/query bridge unavailable')
        sample_dir = base / 'samples' / f'{train_site}_seed42'
        query_dir = base / 'queries' / f'{train_site}_seed42'
    else:
        base = Path(old_root) / 'seeds43_56_v1' / 'seeds' / f'seed{seed}'
        sample_dir = base / 'samples' / train_site
        query_dir = base / 'queries' / train_site
    with np.load(sample_dir / 'selected_indices.npz', allow_pickle=False) as a:
        np.testing.assert_array_equal(a['local_positions'], prepared.sample.local_positions)
        np.testing.assert_array_equal(a['global_indices'], prepared.sample.global_indices)
    with np.load(query_dir / 'query.npz', allow_pickle=False) as a:
        for key in ('permutation','inverse_permutation','sorted_local_positions','sorted_global_indices','sorted_group_ids','sorted_labels','group_ids','group_sizes','group_positive','group_negative'):
            np.testing.assert_array_equal(a[key], getattr(prepared.query, key))
    audit = json.loads((query_dir / 'query_audit.json').read_text(encoding='utf-8'))
    if str(audit.get('query_fingerprint')) != prepared.query.fingerprint:
        raise UnitIdentityError('frozen query fingerprint mismatch')
    return True


def run_models_and_predictions(*, rank_root: Path, metadata, manifest_path: Path, source_fingerprints: Mapping[str, str], output_root: Path, old_root: Path, run_identity: Mapping[str, object], configs: Sequence[str] | None = None, selector=None, allowed_configs: Sequence[str] | None = None):
    from .train_inputs_v2 import load_v2_fold
    from reproduction.siren_a.g4.queries import build_queries
    from reproduction.siren_a.g4.sampler import sample_training_rows
    from reproduction.siren_a.g4b.models import fit_seed_model, verify_runtime_parameters
    from reproduction.siren_a.g4.models import fit_frozen_model
    from reproduction.siren_a.g4b.config import parameters_for_seed
    from reproduction.siren_a.g4.models import predict_scores
    from .v5_family_loo import load_v5_manifest, select_config, _sha
    from .train_inputs_v2 import PROTOCOLS
    model_types = ('lr','hgb','lambdarank'); seeds = tuple(range(42,57)); holdouts=('I0002','I0006','S0001')
    Path(output_root).mkdir(parents=True, exist_ok=True)
    from .v5_family_loo import select_config
    selector = select_config if selector is None else selector
    allowed = tuple(CONFIGS if allowed_configs is None else allowed_configs)
    config_ids = tuple(allowed if configs is None else configs)
    if set(config_ids) != set(config_ids).intersection(allowed) or len(set(config_ids)) != len(config_ids):
        raise ValueError('invalid configuration subset')
    manifest = load_v5_manifest(manifest_path)
    identity_sha = hashlib.sha256(_canonical(run_identity)).hexdigest()
    model_count = prediction_count = bridge_count = replay_max = 0.0
    fold_cache = {}
    def get_fold(config_id, holdout, protocol):
        key=(config_id,holdout,protocol)
        if key not in fold_cache:
            full=load_v2_fold(rank_root,holdout,protocol,'V5',metadata,manifest_path,source_fingerprints)
            fold_cache[key]=project_fold_for_config(full,selector(manifest,config_id))
        return fold_cache[key]
    for config_id in config_ids:
        rows = selector(manifest, config_id)
        for train_site, representative in (('S0001','I0002'),('I0006','S0001')):
            fold = get_fold(config_id, representative, 'transductive')
            for seed in seeds:
                sample = sample_training_rows(train_site,seed,fold.train_age_groups,fold.y_train,fold.train_indices)
                query = build_queries(fold.age_train[sample.local_positions],fold.y_train[sample.local_positions],sample.local_positions,sample.global_indices)
                prep = type('Prepared',(),{'fold':fold,'seed':seed,'sample':sample,'query':query})()
                if train_site in ('S0001','I0006'):
                    _sample_query_bridge(prep, old_root); bridge_count += 1
                for model_type in model_types:
                    model_path = Path(output_root)/'models'/config_id/train_site/f'seed{seed}'/model_type
                    meta = {'config_id':config_id,'train_site':train_site,'seed':seed,'view':'V5','model_type':model_type,'feature_count':len(rows),'feature_ids_sha256':_sha([r.feature_id for r in rows]),'sample_sha256':sample.index_sha256,'query_sha256':query.fingerprint,'run_identity_sha256':identity_sha,'fit_scope':'seed_subset'}
                    if model_path.exists():
                        status = _publish_model(model_path,None,meta)
                    else:
                        if seed == 42:
                            trained=fit_frozen_model(model_type,fold.X_train[sample.local_positions],fold.y_train[sample.local_positions],fold.feature_ids,sample.global_indices,query)
                        else:
                            trained=fit_seed_model(model_type,seed,fold.X_train[sample.local_positions],fold.y_train[sample.local_positions],fold.feature_ids,sample.global_indices,query)
                            expected=parameters_for_seed(model_type,seed,query.truncation_level); verify_runtime_parameters(model_type,trained.model,expected)
                        reload_dir=Path(tempfile.mkdtemp(prefix='loo-replay-',dir=str(output_root)))
                        try:
                            files = __import__('reproduction.siren_a.g4.models',fromlist=['save_checkpoint']).save_checkpoint(reload_dir,trained)
                            reloaded=__import__('reproduction.siren_a.g4.models',fromlist=['load_checkpoint']).load_checkpoint(reload_dir,model_type,files)
                            replay=float(np.max(np.abs(predict_scores(model_type,reloaded.model,reloaded.adapter,fold.X_train[sample.local_positions])-trained.train_score)))
                            replay_max=max(replay_max,replay)
                        finally: shutil.rmtree(reload_dir,ignore_errors=True)
                        meta.update({'parameters':dict(trained.parameters),'score_type':'raw_ranking_score' if model_type=='lambdarank' else 'positive_probability','reload_max_abs_diff':replay})
                        status=_publish_model(model_path,trained,meta); model_count += 1
                    if status == 'SKIP_VERIFIED':
                        pass
    # Models must all be complete before predictions begin.
    for config_id in config_ids:
        rows=selector(manifest,config_id)
        for holdout in holdouts:
            train_site={'I0002':'S0001','I0006':'S0001','S0001':'I0006'}[holdout]
            for protocol in PROTOCOLS:
                fold=get_fold(config_id,holdout,protocol)
                for seed in seeds:
                    for model_type in model_types:
                        model_path=Path(output_root)/'models'/config_id/train_site/f'seed{seed}'/model_type
                        pred_path=Path(output_root)/'predictions'/config_id/f'seed{seed}'/f'holdout_{holdout}'/protocol/model_type
                        pred_identity={'config_id':config_id,'holdout_site':holdout,'train_site':train_site,'protocol':protocol,'seed':seed,'view':'V5','model_type':model_type,'feature_count':len(rows),'feature_ids_sha256':_sha([r.feature_id for r in rows]),'model_complete_sha256':_file_sha(model_path/'complete.json'),'test_indices_sha256':_array_sha(np.asarray(fold.test_indices,dtype=np.int64)),'score_type':'raw_ranking_score' if model_type=='lambdarank' else 'positive_probability','run_identity_sha256':identity_sha}
                        if pred_path.exists():
                            _validate_prediction_unit_identity(pred_path, pred_identity)
                            continue
                        model, model_meta=_load_model(model_path,model_type)
                        scores=predict_scores(model_type,model.model,model.adapter,fold.X_test)
                        payload={'test_global_indices':np.asarray(fold.test_indices,dtype=np.int64),'scores':np.asarray(scores,dtype=np.float64)}
                        pred_meta=dict(pred_identity); pred_meta['prediction_sha256']=_array_sha(payload['scores'])
                        status=publish_npz_unit(pred_path,payload,pred_meta)
                        if status=='BUILD_NEW': prediction_count += 1
    return {'models_built':int(model_count),'predictions_built':int(prediction_count),'sample_query_matches':int(bridge_count),'replay_max_abs_diff':float(replay_max)}


__all__ = ["UnitIdentityError", "bootstrap_delta", "comparison_name", "publish_npz_unit", "validate_public_payload", "project_fold_for_config", "run_models_and_predictions"]
