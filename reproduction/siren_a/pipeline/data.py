"""Strict G2-B1 cache loader for G3-A."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FeatureData:
    values: np.ndarray
    source_valid_mask: np.ndarray
    feature_ids: list
    cache_names: list
    sites: np.ndarray
    fingerprint: str
    provenance: dict


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def _scalar(z,key):
    x=np.asarray(z[key])
    if x.shape!=(): raise ValueError(f"{key} scalar")
    return str(x.item())


def load_features(cache_root,manifest):
    root=Path(cache_root)
    identity=json.loads((root/"run_identity.json").read_text())
    inventory=json.loads((root/"inventory.json").read_text()); index=json.loads((root/"private_index.json").read_text())
    if identity.get("cache_schema_version")!="g2b1_full_v1": raise ValueError("G2-B1 schema")
    if identity.get("manifest_sha256")!=manifest.sha256: raise ValueError("Manifest fingerprint")
    if inventory["summary"]["total"]!=6600 or len(index)!=6600: raise ValueError("G2-B1 record count")
    if inventory["summary"]["site_counts"]!={"I0002":319,"I0006":1142,"S0001":5139}: raise ValueError("site counts")
    values=np.empty((6600,807),dtype=np.float64); masks=np.empty((6600,807),dtype=bool)
    cache_names=[]; sites=[]; aggregate=hashlib.sha256()
    aggregate.update(json.dumps(identity,sort_keys=True,separators=(",",":")).encode())
    aggregate.update(inventory["summary"]["fingerprint"].encode())
    expected_ids=manifest.unique_ids
    for i,item in enumerate(index):
        name=item["cache_name"]
        if name!=f"cache_{i:06d}.npz" or item["site"] not in ("I0002","I0006","S0001"): raise ValueError("private index")
        path=root/"cache"/name; digest=_sha(path); aggregate.update(name.encode()); aggregate.update(digest.encode())
        with np.load(path,allow_pickle=False) as z:
            if _scalar(z,"cache_schema_version")!="g2b1_full_v1" or _scalar(z,"manifest_sha256")!=manifest.sha256: raise ValueError("cache provenance")
            ids=np.asarray(z["unique_feature_ids"]); v=np.asarray(z["unique_values"]); m=np.asarray(z["unique_valid_mask"])
            if ids.shape!=(807,) or ids.tolist()!=expected_ids: raise ValueError("feature order")
            if v.shape!=(807,) or v.dtype!=np.float64 or m.shape!=(807,) or m.dtype!=bool: raise ValueError("cache array schema")
            if not np.isfinite(v[m]).all() or not np.isnan(v[~m]).all(): raise ValueError("source value/mask")
            values[i]=v; masks[i]=m
        cache_names.append(name); sites.append(item["site"])
    return FeatureData(values,masks,list(expected_ids),cache_names,np.asarray(sites),aggregate.hexdigest(),identity)


def validate_coverage(data,rows):
    lookup={(r["site"],r["unique_feature_id"]):r for r in rows}
    for site in sorted(set(data.sites.tolist())):
        selected=data.sites==site
        for j,uid in enumerate(data.feature_ids):
            key=(site,uid)
            if key not in lookup: raise ValueError(f"coverage missing {key}")
            valid=int(data.source_valid_mask[selected,j].sum()); total=int(selected.sum())
            row=lookup[key]
            if int(row["valid_count"])!=valid or int(row["missing_count"])!=total-valid: raise ValueError(f"coverage mismatch {key}")
    return True
