"""Independent read-only verifier for frozen G3-A Rank256 artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCHEMA="g3av_verification_v1"
G3_SCHEMA="g3a_rank256_v1"
SITES=("I0002","I0006","S0001")
EXPECTED_COUNTS={"I0002":319,"I0006":1142,"S0001":5139}
EXPECTED_FILES={"indices.npz","reference.npz","percentile.npz","quantized.npz",
                "views.npz","rank_audit.csv","metadata.json"}
PRIVATE_TOKENS=("/database/",".npz","BidsFolder","SessionID","BDSPPatientID",
                "record_id","patient_id")


@dataclass(frozen=True)
class ManifestContract:
    rows: tuple
    unique_ids: tuple
    families: tuple
    view_positions: dict
    view_sizes: dict
    sha256: str


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()


def check_index_inventory(inventory,index,expected_counts=EXPECTED_COUNTS):
    if len(inventory)!=len(index) or len(inventory)!=sum(expected_counts.values()):
        raise ValueError("inventory/index record count")
    sites=[]; cache_names=set(); relative_paths=set()
    for position,(row,item) in enumerate(zip(inventory,index)):
        expected_name=f"cache_{position:06d}.npz"
        fields=("cache_name","relative_path","site")
        if any(row.get(key)!=item.get(key) for key in fields):
            raise ValueError("inventory/index positional mismatch")
        if item.get("cache_name")!=expected_name:
            raise ValueError("anonymous cache order")
        if item["site"] not in expected_counts:
            raise ValueError("unknown site")
        if item["cache_name"] in cache_names or item["relative_path"] in relative_paths:
            raise ValueError("duplicate record mapping")
        cache_names.add(item["cache_name"]); relative_paths.add(item["relative_path"]); sites.append(item["site"])
    actual={site:sites.count(site) for site in expected_counts}
    if actual!=expected_counts:
        raise ValueError("site counts")
    return np.asarray(sites)


def rebuild_loso(sites):
    sites=np.asarray(sites)
    if sites.ndim!=1 or set(sites.tolist())!=set(SITES):
        raise ValueError("frozen three-site cohort required")
    mapping={
        "I0002":("S0001","I0006"),
        "I0006":("S0001","I0002"),
        "S0001":("I0006","I0002"),
    }
    folds={}
    for holdout,(train,unused) in mapping.items():
        folds[holdout]={
            "holdout_site":holdout,"train_site":train,"unused_site":unused,
            "train_indices":np.flatnonzero(sites==train).astype(np.int64),
            "test_indices":np.flatnonzero(sites==holdout).astype(np.int64),
            "unused_indices":np.flatnonzero(sites==unused).astype(np.int64),
        }
    return folds


def check_indices(arrays,fold,total):
    names=("train_indices","test_indices","unused_indices")
    loaded={}
    for name in names:
        if name not in arrays:
            raise ValueError(f"missing {name}")
        value=np.asarray(arrays[name])
        if value.dtype!=np.int64 or value.ndim!=1:
            raise ValueError(f"{name} schema")
        if len(np.unique(value))!=len(value) or np.any((value<0)|(value>=total)):
            raise ValueError(f"{name} invalid")
        if not np.array_equal(value,np.asarray(fold[name],dtype=np.int64)):
            raise ValueError(f"{name} exact mismatch")
        loaded[name]=value
    if any(np.intersect1d(loaded[a],loaded[b]).size for a,b in
           (("train_indices","test_indices"),("train_indices","unused_indices"),("test_indices","unused_indices"))):
        raise ValueError("index overlap")
    joined=np.sort(np.concatenate(tuple(loaded.values())))
    if not np.array_equal(joined,np.arange(total,dtype=np.int64)):
        raise ValueError("index coverage")
    return True


def check_unit_integrity(unit):
    unit=Path(unit)
    if unit.with_name(unit.name+".building").exists():
        raise ValueError("incomplete building directory")
    if not unit.is_dir() or unit.is_symlink():
        raise ValueError("unit directory")
    entries={item.name for item in unit.iterdir()}
    if entries!=EXPECTED_FILES|{"complete.json"}:
        raise ValueError("unit file set")
    complete_path=unit/"complete.json"
    if complete_path.is_symlink() or not complete_path.is_file():
        raise ValueError("completion file")
    complete=json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("schema")!=G3_SCHEMA or set(complete.get("files",{}))!=EXPECTED_FILES:
        raise ValueError("completion schema")
    for name in EXPECTED_FILES:
        path=unit/name
        if path.is_symlink() or not path.is_file() or sha256_file(path)!=complete["files"][name]:
            raise ValueError(f"file integrity {name}")
        if path.suffix==".npz":
            try:
                with np.load(path,allow_pickle=False) as payload:
                    for key in payload.files:
                        if np.asarray(payload[key]).dtype==object:
                            raise ValueError(f"object dtype {name}:{key}")
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError(f"invalid NPZ {name}") from exc
    return True


def compare_matrix(observed,expected,label):
    observed=np.asarray(observed); expected=np.asarray(expected)
    if observed.shape!=expected.shape or observed.dtype!=expected.dtype:
        raise ValueError(f"{label} schema mismatch")
    if not np.array_equal(observed,expected,equal_nan=True):
        mismatch=~((observed==expected)|(np.isnan(observed)&np.isnan(expected))) if np.issubdtype(observed.dtype,np.floating) else observed!=expected
        raise ValueError(f"{label} mismatch count {int(np.sum(mismatch))}")
    return True


def independent_views(values,masks,manifest):
    values=np.asarray(values); masks=np.asarray(masks)
    if values.ndim!=2 or values.shape!=masks.shape or values.shape[1]!=len(manifest.unique_ids):
        raise ValueError("unique matrix shape")
    output_values={}; output_masks={}
    for view,positions in manifest.view_positions.items():
        output_values[view]=values[:,positions].copy()
        output_masks[view]=masks[:,positions].copy()
    return output_values,output_masks


def _normal(value,expected):
    if isinstance(expected,(int,np.integer)) and not isinstance(expected,bool):
        return int(value)
    if isinstance(expected,(float,np.floating)):
        return float(value)
    if isinstance(expected,bool):
        return str(value).lower()=="true"
    return str(value)


def compare_audit_rows(expected,observed,key_fields):
    if len(expected)!=len(observed):
        raise ValueError("audit row count")
    def keyed(rows):
        result={}
        for row in rows:
            key=tuple(str(row[field]) for field in key_fields)
            if key in result:
                raise ValueError("duplicate audit key")
            result[key]=row
        return result
    left=keyed(expected); right=keyed(observed)
    if set(left)!=set(right):
        raise ValueError("audit keys")
    for key,row in left.items():
        other=right[key]
        if set(row)!=set(other):
            raise ValueError(f"audit fields {key}")
        for field,want in row.items():
            got=_normal(other[field],want)
            if isinstance(want,(float,np.floating)) and np.isnan(want):
                if not np.isnan(got): raise ValueError(f"audit value {key}:{field}")
            elif got!=want:
                raise ValueError(f"audit value {key}:{field}")
    return True


def atomic_json_no_clobber(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary=None; linked=False
    try:
        with tempfile.NamedTemporaryFile("wb",dir=path.parent,prefix=path.name+".",suffix=".tmp",delete=False) as handle:
            temporary=Path(handle.name); handle.write(canonical_json(payload)); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary,path); linked=True
        if json.loads(path.read_text(encoding="utf-8"))!=payload:
            raise ValueError("JSON round trip")
    except Exception:
        if linked and path.exists(): path.unlink()
        raise
    finally:
        if temporary is not None and temporary.exists(): temporary.unlink()


def verify_private_output(root,expected_identity=None):
    root=Path(root)
    if (root/".building").exists():
        raise ValueError("incomplete private output")
    complete_path=root/"complete.json"
    if not complete_path.is_file() or complete_path.is_symlink():
        raise ValueError("private completion missing")
    complete=json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("schema")!=SCHEMA:
        raise ValueError("private completion schema")
    files=complete.get("files",{})
    actual={p.name for p in root.iterdir() if p.is_file() and p.name!="complete.json"}
    if set(files)!=actual:
        raise ValueError("private output file set")
    for name,digest in files.items():
        path=root/name
        if path.is_symlink() or not path.is_file() or sha256_file(path)!=digest:
            raise ValueError(f"private file integrity {name}")
    if expected_identity is not None:
        summaries=[json.loads((root/name).read_text(encoding="utf-8")) for name in files if name.endswith(".json")]
        if not any(all(item.get(key)==value for key,value in expected_identity.items()) for item in summaries):
            raise ValueError("private identity mismatch")
    return True


def validate_public_payload(rows):
    text=json.dumps(rows,sort_keys=True)
    if any(token.lower() in text.lower() for token in PRIVATE_TOKENS):
        raise ValueError("private identity in public output")
    return True


@dataclass(frozen=True)
class SourceBundle:
    values: np.ndarray
    masks: np.ndarray
    feature_ids: tuple
    families: tuple
    sites: np.ndarray
    cache_names: tuple
    g2_identity: dict
    inventory_fingerprint: str
    cache_fingerprint: str
    cache_hashes: tuple


def inventory_fingerprint(rows):
    return hashlib.sha256(canonical_json(rows)).hexdigest()


def load_manifest_contract(path, expected_sha):
    path=Path(path)
    digest=sha256_file(path)
    if digest!=expected_sha:
        raise ValueError("Frozen Manifest SHA256")
    with path.open(newline="",encoding="utf-8") as handle:
        rows=tuple(dict(row) for row in csv.DictReader(handle))
    if len(rows)!=827:
        raise ValueError("Manifest row count")
    by_id={}; family_by_id={}
    for row in rows:
        uid=row["unique_feature_id"]
        if uid not in by_id:
            by_id[uid]=len(by_id); family_by_id[uid]=row["family"]
        elif family_by_id[uid]!=row["family"]:
            raise ValueError("Manifest family mismatch")
    if len(by_id)!=807:
        raise ValueError("Manifest unique count")
    sizes={"V1":113,"V2":129,"V3":349,"V4":113,"V5":123}
    positions={}
    for view,size in sizes.items():
        selected=sorted((r for r in rows if r["view_id"]==view),key=lambda r:int(r["view_position"]))
        if len(selected)!=size or [int(r["view_position"]) for r in selected]!=list(range(size)):
            raise ValueError(f"Manifest {view} positions")
        positions[view]=np.asarray([by_id[r["unique_feature_id"]] for r in selected],dtype=np.int64)
    if sum(sizes.values())!=827 or set(r["view_id"] for r in rows)!=set(sizes):
        raise ValueError("Manifest views")
    return ManifestContract(rows,tuple(by_id),tuple(family_by_id[x] for x in by_id),positions,sizes,digest)


def _scalar(payload,key):
    if key not in payload:
        raise ValueError(f"cache missing {key}")
    value=np.asarray(payload[key])
    if value.shape!=():
        raise ValueError(f"cache scalar {key}")
    return str(value.item())


def _is_sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)


def load_sources(cache_root,manifest):
    root=Path(cache_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("G2-B1 root")
    identity=json.loads((root/"run_identity.json").read_text(encoding="utf-8"))
    inventory=json.loads((root/"inventory.json").read_text(encoding="utf-8"))
    index=json.loads((root/"private_index.json").read_text(encoding="utf-8"))
    rows=inventory.get("rows",[])
    sites=check_index_inventory(rows,index)
    rebuilt_inventory=inventory_fingerprint(rows)
    summary=inventory.get("summary",{})
    if rebuilt_inventory!=summary.get("fingerprint") or rebuilt_inventory!=identity.get("inventory_fingerprint"):
        raise ValueError("inventory fingerprint")
    if summary.get("duplicates")!=0 or summary.get("symlinks")!=0:
        raise ValueError("inventory summary")
    if identity.get("cache_schema_version")!="g2b1_full_v1":
        raise ValueError("G2-B1 schema")
    if identity.get("manifest_sha256")!=manifest.sha256:
        raise ValueError("G2-B1 Manifest provenance")
    cache_dir=root/"cache"
    expected_names={f"cache_{i:06d}.npz" for i in range(6600)}
    actual_names={x.name for x in cache_dir.iterdir()}
    if actual_names!=expected_names:
        raise ValueError("G2-B1 cache file set")
    values=np.empty((6600,807),dtype=np.float64)
    masks=np.empty((6600,807),dtype=bool)
    hashes=[]; aggregate=hashlib.sha256()
    aggregate.update(canonical_json(identity)); aggregate.update(rebuilt_inventory.encode())
    provenance_keys=("manifest_sha256","spec_sha256","g2a_features_sha256","g2a_io_sha256",
                     "g2b0_batch_safety_sha256","g2b1_cache_contract_sha256",
                     "g2b1_full_run_sha256","python_version","numpy_version","cache_schema_version")
    for i,item in enumerate(index):
        name=item["cache_name"]; path=cache_dir/name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"cache path {i}")
        digest=sha256_file(path); hashes.append(digest)
        aggregate.update(name.encode()); aggregate.update(digest.encode())
        with np.load(path,allow_pickle=False) as payload:
            if any(np.asarray(payload[k]).dtype==object for k in payload.files):
                raise ValueError(f"cache object dtype {i}")
            if _scalar(payload,"source_npz_sha256") is None or not _is_sha(_scalar(payload,"source_npz_sha256")):
                raise ValueError(f"source SHA {i}")
            for key in provenance_keys:
                if _scalar(payload,key)!=str(identity[key]):
                    raise ValueError(f"cache provenance {i}:{key}")
            ids=np.asarray(payload["unique_feature_ids"])
            row_values=np.asarray(payload["unique_values"])
            row_masks=np.asarray(payload["unique_valid_mask"])
            support=np.asarray(payload["unique_support_count"])
            if ids.shape!=(807,) or tuple(ids.tolist())!=manifest.unique_ids:
                raise ValueError(f"feature order {i}")
            if row_values.shape!=(807,) or row_values.dtype!=np.float64 or row_masks.shape!=(807,) or row_masks.dtype!=bool:
                raise ValueError(f"unique schema {i}")
            if support.shape!=(807,) or support.dtype!=np.int64 or np.any(support<0):
                raise ValueError(f"support schema {i}")
            if not np.isfinite(row_values[row_masks]).all() or not np.isnan(row_values[~row_masks]).all():
                raise ValueError(f"value mask {i}")
            for view,size in manifest.view_sizes.items():
                vv=np.asarray(payload[f"{view}_values"]); vm=np.asarray(payload[f"{view}_valid_mask"])
                if vv.shape!=(size,) or vv.dtype!=np.float64 or vm.shape!=(size,) or vm.dtype!=bool:
                    raise ValueError(f"{view} schema {i}")
                pos=manifest.view_positions[view]
                if not np.array_equal(vv,row_values[pos],equal_nan=True) or not np.array_equal(vm,row_masks[pos]):
                    raise ValueError(f"{view} mapping {i}")
            values[i]=row_values; masks[i]=row_masks
    return SourceBundle(values,masks,manifest.unique_ids,manifest.families,sites,
                        tuple(x["cache_name"] for x in index),identity,rebuilt_inventory,
                        aggregate.hexdigest(),tuple(hashes))


def check_g3_identity(identity,code_files,cache_fingerprint,inventory_fp,manifest_sha,spec_sha):
    expected={"schema":G3_SCHEMA,"g2b1_cache_fingerprint":cache_fingerprint,
              "g2b1_inventory_fingerprint":inventory_fp,"manifest_sha256":manifest_sha,
              "spec_sha256":spec_sha,"formula":"(L+0.5E)/n","levels":256,"rounding":"half_up"}
    for key,value in expected.items():
        if identity.get(key)!=value:
            raise ValueError(f"G3 identity {key}")
    for key,path in code_files.items():
        if identity.get(key)!=sha256_file(path):
            raise ValueError(f"G3 code identity {key}")
    return True


def _reference_bounds(reference):
    lo=np.full(len(reference.unique_values),np.nan,dtype=np.float64)
    hi=np.full(len(reference.unique_values),np.nan,dtype=np.float64)
    for j,column in enumerate(reference.unique_values):
        if len(column):
            lo[j]=column[0]; hi[j]=column[-1]
    return lo,hi


def audit_rank_independent(raw,percentile,quantized,source_mask,rank_mask,levels,
                           feature_ids,families,reference_n,reference_min,reference_max,
                           include_lawful_zero=True):
    raw=np.asarray(raw); percentile=np.asarray(percentile); quantized=np.asarray(quantized)
    source=np.asarray(source_mask); rank=np.asarray(rank_mask); levels=np.asarray(levels)
    if not(raw.shape==percentile.shape==quantized.shape==source.shape==rank.shape==levels.shape):
        raise ValueError("audit shape")
    if raw.dtype!=np.float64 or percentile.dtype!=np.float64 or quantized.dtype!=np.float64:
        raise ValueError("audit float dtype")
    if source.dtype!=bool or rank.dtype!=bool or levels.dtype!=np.int16:
        raise ValueError("audit mask/level dtype")
    if np.any(rank&~source) or not np.isnan(percentile[~rank]).all() or not np.isnan(quantized[~rank]).all():
        raise ValueError("audit missing semantics")
    if not np.isfinite(percentile[rank]).all() or not np.isfinite(quantized[rank]).all():
        raise ValueError("audit finite")
    rows=[]
    for j,uid in enumerate(feature_ids):
        idx=np.flatnonzero(rank[:,j]); rv=raw[idx,j]; pv=percentile[idx,j]
        qv=quantized[idx,j]; kv=levels[idx,j]
        pu=np.unique(pv); qu=np.unique(kv)
        grouped={int(level):len(np.unique(pv[kv==level])) for level in qu}
        error=np.abs(qv-pv)
        row={"feature_id":uid,"family":families[j],"n_total":len(raw),
             "n_source_valid":int(source[:,j].sum()),"n_rank_valid":len(idx),
             "reference_n":int(reference_n[j]),
             "reference_unavailable_count":int(np.sum(source[:,j]&~rank[:,j])),
             "n_percentile_unique":len(pu),"n_quantized_unique":len(qu),
             "occupied_bins":len(qu),"occupied_bin_fraction":len(qu)/256,
             "max_abs_quantization_error":float(np.max(error)) if len(error) else 0.0,
             "mean_abs_quantization_error":float(np.mean(error)) if len(error) else 0.0,
             "new_tie_group_count":sum(value>1 for value in grouped.values()),
             "distinct_rank_levels_collapsed":len(pu)-len(qu),
             "zero_level_count":int(np.sum(kv==0)),"top_level_count":int(np.sum(kv==255)),
             "percentile_min":float(np.min(pv)) if len(pv) else np.nan,
             "percentile_max":float(np.max(pv)) if len(pv) else np.nan,
             "quantized_min":float(np.min(qv)) if len(qv) else np.nan,
             "quantized_max":float(np.max(qv)) if len(qv) else np.nan,
             "below_reference_count":int(np.sum(rv<reference_min[j])) if np.isfinite(reference_min[j]) else 0,
             "above_reference_count":int(np.sum(rv>reference_max[j])) if np.isfinite(reference_max[j]) else 0}
        if include_lawful_zero:
            row["lawful_zero_count"]=int(np.sum(rv==0))
        rows.append(row)
    return rows


def _load_csv(path):
    with Path(path).open(newline="",encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _unit_metadata(g3_identity,fold,protocol):
    return {"schema":G3_SCHEMA,"holdout_site":fold["holdout_site"],
            "train_site":fold["train_site"],"unused_site":fold["unused_site"],
            "train_count":len(fold["train_indices"]),"test_count":len(fold["test_indices"]),
            "protocol":protocol,
            "run_identity_sha256":hashlib.sha256(canonical_json(g3_identity)).hexdigest(),
            "levels":256,"rounding":"half_up"}


def _npz_dict(path):
    with np.load(path,allow_pickle=False) as payload:
        return {key:np.asarray(payload[key]) for key in payload.files}


def verify_unit(source,manifest,fold,protocol,unit,g3_identity):
    from verify_rank import build_reference,check_quantized,check_reference,quantize_values,rank_values
    unit=Path(unit); check_unit_integrity(unit)
    metadata=json.loads((unit/"metadata.json").read_text(encoding="utf-8"))
    if metadata!=_unit_metadata(g3_identity,fold,protocol):
        raise ValueError("unit metadata")
    indices=_npz_dict(unit/"indices.npz")
    if set(indices)!={"train_indices","test_indices","unused_indices"}:
        raise ValueError("indices keys")
    check_indices(indices,fold,len(source.sites))
    train_idx=fold["train_indices"]; test_idx=fold["test_indices"]
    train=source.values[train_idx]; test=source.values[test_idx]
    train_source=source.masks[train_idx]; test_source=source.masks[test_idx]
    train_ref=build_reference(train,train_source)
    test_ref=build_reference(test,test_source) if protocol=="transductive" else train_ref
    saved_ref=_npz_dict(unit/"reference.npz")
    expected_ref_keys={"train_values","train_offsets","train_counts","train_fingerprint"}
    if protocol=="transductive":
        expected_ref_keys|={"test_values","test_offsets","test_counts","test_fingerprint"}
    if set(saved_ref)!=expected_ref_keys:
        raise ValueError("reference keys")
    check_reference(saved_ref,train_ref,"train")
    if protocol=="transductive":
        check_reference(saved_ref,test_ref,"test")
    train_pct,train_rank=rank_values(train,train_source,train_ref)
    test_pct,test_rank=rank_values(test,test_source,test_ref)
    saved_pct=_npz_dict(unit/"percentile.npz")
    expected_pct={"train_percentile":train_pct,"test_percentile":test_pct,
                  "train_source_valid_mask":train_source,"test_source_valid_mask":test_source,
                  "train_rank_valid_mask":train_rank,"test_rank_valid_mask":test_rank}
    if set(saved_pct)!=set(expected_pct):
        raise ValueError("percentile keys")
    for key,value in expected_pct.items():
        compare_matrix(saved_pct[key],value,key)
    train_q,train_levels=quantize_values(train_pct,train_rank)
    test_q,test_levels=quantize_values(test_pct,test_rank)
    saved_q=_npz_dict(unit/"quantized.npz")
    expected_q={"train_quantized":train_q,"test_quantized":test_q,
                "train_levels":train_levels,"test_levels":test_levels}
    if set(saved_q)!=set(expected_q):
        raise ValueError("quantized keys")
    for key,value in expected_q.items():
        compare_matrix(saved_q[key],value,key)
    check_quantized(saved_q["train_levels"],saved_q["train_quantized"],train_pct,train_rank)
    check_quantized(saved_q["test_levels"],saved_q["test_quantized"],test_pct,test_rank)
    saved_views=_npz_dict(unit/"views.npz")
    expected_view_keys=set()
    for role,values,masks in (("train",train_q,train_rank),("test",test_q,test_rank)):
        view_values,view_masks=independent_views(values,masks,manifest)
        for view in manifest.view_sizes:
            vk=f"{role}_{view}_values"; mk=f"{role}_{view}_rank_valid_mask"
            expected_view_keys|={vk,mk}
            compare_matrix(saved_views[vk],view_values[view],vk)
            compare_matrix(saved_views[mk],view_masks[view],mk)
    if set(saved_views)!=expected_view_keys:
        raise ValueError("view keys")
    train_lo,train_hi=_reference_bounds(train_ref); test_lo,test_hi=_reference_bounds(test_ref)
    expected_audit=[]
    for role,raw,pct,q,sm,rm,levels,ref,lo,hi in (
        ("train",train,train_pct,train_q,train_source,train_rank,train_levels,train_ref,train_lo,train_hi),
        ("test",test,test_pct,test_q,test_source,test_rank,test_levels,test_ref,test_lo,test_hi),
    ):
        rows=audit_rank_independent(raw,pct,q,sm,rm,levels,source.feature_ids,source.families,
                                    ref.counts,lo,hi,include_lawful_zero=False)
        for row in rows:
            row.update({"holdout_site":fold["holdout_site"],"train_site":fold["train_site"],
                        "protocol":protocol,"role":role})
        expected_audit.extend(rows)
    observed_audit=_load_csv(unit/"rank_audit.csv")
    compare_audit_rows(expected_audit,observed_audit,("feature_id","holdout_site","protocol","role"))
    max_error=max(row["max_abs_quantization_error"] for row in expected_audit)
    below=sum(row["below_reference_count"] for row in expected_audit if row["role"]=="test")
    above=sum(row["above_reference_count"] for row in expected_audit if row["role"]=="test")
    return {"holdout_site":fold["holdout_site"],"train_site":fold["train_site"],
            "protocol":protocol,"train_count":len(train),"test_count":len(test),
            "index_verified":True,"reference_verified":True,"percentile_verified":True,
            "quantized_verified":True,"views_verified":True,"audit_verified":True,
            "source_valid_positions":int(train_source.sum()+test_source.sum()),
            "rank_valid_positions":int(train_rank.sum()+test_rank.sum()),
            "train_source_valid":int(train_source.sum()),"test_source_valid":int(test_source.sum()),
            "train_rank_valid":int(train_rank.sum()),"test_rank_valid":int(test_rank.sum()),
            "train_percentile_checked_positions":int(train_pct.size),
            "test_percentile_checked_positions":int(test_pct.size),
            "train_valid_percentiles":int(train_rank.sum()),"test_valid_percentiles":int(test_rank.sum()),
            "percentile_mismatch_count":0,"level_mismatch_count":0,"view_mismatch_count":0,
            "audit_mismatch_count":0,"max_percentile_difference":0.0,
            "max_quantization_error":float(max_error),"off_grid_count":0,
            "formula_mismatch_count":0,"below_reference_count":int(below),
            "above_reference_count":int(above),"status":"PASS",
            "_audit_rows":expected_audit,"_train_percentile":train_pct,
            "_train_rank_mask":train_rank,"_train_levels":train_levels,
            "_train_quantized":train_q}


def _verify_coverage(source,coverage_path):
    rows=_load_csv(coverage_path)
    if len(rows)!=3228:
        raise ValueError("G2 coverage row count")
    lookup={(r["site"],r["unique_feature_id"]):r for r in rows}
    expected_sites=(*SITES,"Overall")
    if len(lookup)!=len(rows):
        raise ValueError("G2 coverage duplicate")
    totals={}
    for site in expected_sites:
        selected=np.ones(len(source.sites),bool) if site=="Overall" else source.sites==site
        total=int(selected.sum()); valid_total=0
        for j,uid in enumerate(source.feature_ids):
            row=lookup.get((site,uid))
            if row is None or row["family"]!=source.families[j] or int(row["records"])!=total:
                raise ValueError(f"G2 coverage identity {site}:{uid}")
            valid=int(source.masks[selected,j].sum()); missing=total-valid
            if int(row["valid_count"])!=valid or int(row["missing_count"])!=missing:
                raise ValueError(f"G2 coverage value {site}:{uid}")
            valid_total+=valid
        totals[site]=valid_total
    expected={"I0002":238476,"I0006":881160,"S0001":4002096,"Overall":5121732}
    if totals!=expected:
        raise ValueError(f"G2 coverage totals {totals}")
    return totals


def _verify_public_audit(results,public_path):
    rows=_load_csv(public_path)
    if len(rows)!=9684:
        raise ValueError("G3 public audit row count")
    expected=[]
    for result in results:
        expected.extend(result["_audit_rows"])
    compare_audit_rows(expected,rows,("feature_id","holdout_site","protocol","role"))
    return len(rows)


def _verify_root(g3_root,source,manifest,spec_sha,pipeline_dir):
    root=Path(g3_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("G3 root")
    expected_root={"run_identity.json","data_manifest.json","split_manifest.json","reports",
                   "holdout_I0002","holdout_I0006","holdout_S0001"}
    if {p.name for p in root.iterdir()}!=expected_root:
        raise ValueError("G3 root file set")
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("G3 symlink")
    if any(p.name.endswith(".building") for p in root.rglob("*")):
        raise ValueError("G3 incomplete directory")
    identity=json.loads((root/"run_identity.json").read_text(encoding="utf-8"))
    code_files={name+"_sha256":Path(pipeline_dir)/(name+".py")
                for name in ("data","split","normalize","views","audit","run_prep")}
    check_g3_identity(identity,code_files,source.cache_fingerprint,source.inventory_fingerprint,
                      manifest.sha256,spec_sha)
    data_manifest=json.loads((root/"data_manifest.json").read_text(encoding="utf-8"))
    expected_data={"records":6600,"features":807,
                   "site_counts":EXPECTED_COUNTS,"cache_fingerprint":source.cache_fingerprint,
                   "feature_ids":list(source.feature_ids)}
    if data_manifest!=expected_data:
        raise ValueError("G3 data manifest")
    folds=rebuild_loso(source.sites)
    split_manifest=json.loads((root/"split_manifest.json").read_text(encoding="utf-8"))
    expected_split={}
    for holdout,fold in folds.items():
        expected_split[holdout]={"train_site":fold["train_site"],"test_site":holdout,
                                 "unused_site":fold["unused_site"],
                                 "train_count":len(fold["train_indices"]),
                                 "test_count":len(fold["test_indices"]),
                                 "unused_count":len(fold["unused_indices"])}
    if split_manifest!=expected_split:
        raise ValueError("G3 split manifest")
    return identity,folds


def verify_all(config,holdouts=None):
    started=__import__("time").perf_counter()
    manifest=load_manifest_contract(config["manifest_path"],config["manifest_sha"])
    if sha256_file(config["spec_path"])!=config["spec_sha"]:
        raise ValueError("Frozen Spec SHA256")
    source=load_sources(config["g2_root"],manifest)
    if source.cache_fingerprint!=config["expected_cache_fingerprint"]:
        raise ValueError("G2-B1 cache fingerprint")
    identity,folds=_verify_root(config["g3_root"],source,manifest,config["spec_sha"],config["pipeline_dir"])
    coverage_totals=_verify_coverage(source,config["g2_coverage"])
    selected=tuple(holdouts or SITES); results=[]
    for holdout in selected:
        if holdout not in folds:
            raise ValueError("holdout")
        for protocol in ("transductive","inductive"):
            unit=Path(config["g3_root"])/f"holdout_{holdout}"/protocol
            results.append(verify_unit(source,manifest,folds[holdout],protocol,unit,identity))
        left,right=results[-2],results[-1]
        for field in ("_train_percentile","_train_rank_mask","_train_levels","_train_quantized"):
            compare_matrix(left[field],right[field],f"{holdout} cross-protocol {field}")
    public_rows=None
    if set(selected)==set(SITES):
        public_rows=_verify_public_audit(results,config["g3_public_audit"])
    duration=__import__("time").perf_counter()-started
    public=[]
    for result in results:
        public.append({k:v for k,v in result.items() if not k.startswith("_")})
    validate_public_payload(public)
    return {"schema":SCHEMA,"status":"PASS","source_records":6600,
            "source_cache_fingerprint":source.cache_fingerprint,
            "inventory_fingerprint":source.inventory_fingerprint,
            "manifest_sha256":manifest.sha256,"spec_sha256":config["spec_sha"],
            "coverage_totals":coverage_totals,"public_audit_rows":public_rows,
            "units":public,"duration_seconds":duration,
            "g2_cache_files_verified":len(source.cache_hashes),
            "source_file_hash_aggregate":hashlib.sha256("".join(source.cache_hashes).encode()).hexdigest(),
            "verifier_sha256":sha256_file(__file__),
            "rank_verifier_sha256":sha256_file(Path(__file__).with_name("verify_rank.py"))}


def _atomic_text_no_clobber(path,text):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary=None; linked=False
    try:
        with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=path.parent,
                                         prefix=path.name+".",suffix=".tmp",delete=False) as handle:
            temporary=Path(handle.name); handle.write(text); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary,path); linked=True
    except Exception:
        if linked and path.exists(): path.unlink()
        raise
    finally:
        if temporary is not None and temporary.exists(): temporary.unlink()


def write_private_result(root,result):
    root=Path(root)
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    identity={k:result[k] for k in ("schema","status","source_cache_fingerprint",
                                     "inventory_fingerprint","manifest_sha256","spec_sha256",
                                     "verifier_sha256","rank_verifier_sha256")}
    payloads={
        "verification_identity.json":identity,
        "source_checks.json":{k:result[k] for k in ("source_records","g2_cache_files_verified",
                                                     "source_file_hash_aggregate","coverage_totals")},
        "verification_summary.json":{"status":result["status"],"units":result["units"],
                                     "public_audit_rows":result["public_audit_rows"],
                                     "duration_seconds":result["duration_seconds"]},
    }
    for name,payload in payloads.items():
        atomic_json_no_clobber(root/name,payload)
    files={name:sha256_file(root/name) for name in payloads}
    atomic_json_no_clobber(root/"complete.json",{"schema":SCHEMA,"files":files})
    verify_private_output(root,{"source_cache_fingerprint":result["source_cache_fingerprint"]})
    return root


def public_rows(result):
    rows=[]
    fields=("holdout_site","train_site","protocol","train_count","test_count",
            "index_verified","reference_verified","percentile_verified","quantized_verified",
            "views_verified","audit_verified","source_valid_positions","rank_valid_positions",
            "percentile_mismatch_count","level_mismatch_count","view_mismatch_count",
            "audit_mismatch_count","max_quantization_error","below_reference_count",
            "above_reference_count","status")
    for unit in result["units"]:
        rows.append({key:unit[key] for key in fields})
    validate_public_payload(rows)
    return rows


def write_public_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile("w",newline="",encoding="utf-8",dir=path.parent,
                                     prefix=path.name+".",suffix=".tmp",delete=False) as handle:
        temporary=Path(handle.name); writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader(); writer.writerows(rows); handle.flush(); os.fsync(handle.fileno())
    try:
        os.link(temporary,path)
    finally:
        if temporary.exists(): temporary.unlink()


def _default_config(repo):
    repo=Path(repo)
    return {
        "g2_root":Path("/database/home/gaohaojie/workspace/python-example-2026-siren-a-g2b1-private/night_cache_v1"),
        "g3_root":Path("/database/home/gaohaojie/workspace/python-example-2026-siren-a-g3a-private/rank256_v1"),
        "manifest_path":repo/"reproduction/siren_a/docs/G1B_FEATURE_MANIFEST_FROZEN.csv",
        "spec_path":repo/"reproduction/siren_a/docs/G1B_FEATURE_SPEC_FROZEN.md",
        "manifest_sha":"7aef4e0462936ad0fbb3ee504a71062b0f50d82de5c54e1d70df2d13691e455a",
        "spec_sha":"f9f6ae6b7fd3126569914dad9c643c17ee9fa56354c460c694f46060cb174d9a",
        "expected_cache_fingerprint":"cf88b4e87f8b524258c685db696c278198dae1e6a10bd39fb63e893b2df031df",
        "pipeline_dir":repo/"reproduction/siren_a/pipeline",
        "g2_coverage":repo/"reproduction/siren_a/audit/g2b1_feature_coverage_aggregate.csv",
        "g3_public_audit":repo/"reproduction/siren_a/audit/g3a_rank_summary.csv",
    }


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description="Independent read-only G3-A verifier")
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only",action="store_true")
    modes.add_argument("--smoke",action="store_true")
    modes.add_argument("--full",action="store_true")
    modes.add_argument("--verify-output",action="store_true")
    parser.add_argument("--repo",default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--private-root",default="/database/home/gaohaojie/workspace/python-example-2026-siren-a-g3av-private/verification_v1")
    args=parser.parse_args(argv)
    if args.verify_output:
        verify_private_output(args.private_root)
        print(json.dumps({"status":"PASS","mode":"verify-output"},sort_keys=True))
        return 0
    config=_default_config(args.repo)
    holdouts=("I0002",) if args.smoke else SITES
    result=verify_all(config,holdouts)
    if args.preflight_only:
        print(json.dumps({"status":"PASS","mode":"preflight","records":result["source_records"],
                          "fingerprint":result["source_cache_fingerprint"]},sort_keys=True))
        return 0
    if args.full:
        write_private_result(args.private_root,result)
    print(json.dumps({"status":"PASS","mode":"full" if args.full else "smoke",
                      "units":len(result["units"]),"duration_seconds":result["duration_seconds"]},sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
