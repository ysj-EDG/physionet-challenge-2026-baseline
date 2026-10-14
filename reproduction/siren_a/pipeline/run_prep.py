"""Recoverable G3-A LOSO Transductive/Inductive Rank256 runner."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
NIGHT=HERE.parent/"night_cache"
import sys
if str(NIGHT) not in sys.path: sys.path.insert(0,str(NIGHT))

import audit as audit_module
import data as data_module
import normalize as normalize_module
import split as split_module
import views as views_module
from audit import audit_rank, audit_raw, compare_protocols
from data import load_features, validate_coverage
from normalize import apply_rank_reference, fit_rank_reference, quantize_rank
from split import make_loso
from views import get_views
from g2a_io import load_manifest

SCHEMA="g3a_rank256_v1"
PROTOCOLS=("transductive","inductive")
HOLDOUTS=("I0002","I0006","S0001")
MANIFEST_SHA="7aef4e0462936ad0fbb3ee504a71062b0f50d82de5c54e1d70df2d13691e455a"
SPEC_SHA="f9f6ae6b7fd3126569914dad9c643c17ee9fa56354c460c694f46060cb174d9a"


class UnitAction(str,Enum):
    COMPUTE="COMPUTE"; SKIP_VERIFIED="SKIP_VERIFIED"; STOP_INVALID="STOP_INVALID"


@dataclass(frozen=True)
class UnitValidation:
    valid: bool
    errors: tuple


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def _json_bytes(x): return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def _atomic_json(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    with tempfile.NamedTemporaryFile("wb",dir=path.parent,prefix=path.name+".",suffix=".tmp",delete=False) as f:
        tmp=Path(f.name); f.write(_json_bytes(payload)); f.flush(); os.fsync(f.fileno())
    try: os.link(tmp,path)
    finally:
        if tmp.exists(): tmp.unlink()
    if json.loads(path.read_text())!=payload: raise ValueError("JSON round trip")


def _atomic_npz(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    arrays={k:np.asarray(v) for k,v in payload.items()}
    if any(v.dtype==object for v in arrays.values()): raise ValueError("object array forbidden")
    with tempfile.NamedTemporaryFile("wb",dir=path.parent,prefix=path.name+".",suffix=".tmp",delete=False) as f:
        tmp=Path(f.name); np.savez_compressed(f,**arrays); f.flush(); os.fsync(f.fileno())
    try: os.link(tmp,path)
    finally:
        if tmp.exists(): tmp.unlink()
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!=set(arrays): raise ValueError("NPZ round trip keys")
        for k,v in arrays.items():
            loaded=np.asarray(z[k])
            same=np.array_equal(loaded,v,equal_nan=True) if np.issubdtype(v.dtype,np.floating) else np.array_equal(loaded,v)
            if not same: raise ValueError(f"NPZ round trip {k}")


def _atomic_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    fields=list(rows[0]) if rows else []
    with tempfile.NamedTemporaryFile("w",newline="",encoding="utf-8",dir=path.parent,prefix=path.name+".",suffix=".tmp",delete=False) as f:
        tmp=Path(f.name); writer=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); writer.writeheader(); writer.writerows(rows); f.flush(); os.fsync(f.fileno())
    try: os.link(tmp,path)
    finally:
        if tmp.exists(): tmp.unlink()


def validate_unit(path,expected_metadata):
    path=Path(path); errors=[]
    try:
        if not path.is_dir(): return UnitValidation(False,("unit missing",))
        complete=json.loads((path/"complete.json").read_text()); metadata=json.loads((path/"metadata.json").read_text())
        if metadata!=expected_metadata: errors.append("metadata mismatch")
        files=complete.get("files",{})
        for name,digest in files.items():
            target=path/name
            if not target.is_file() or _sha(target)!=digest: errors.append(f"file mismatch {name}")
        if set(files)!={p.name for p in path.iterdir() if p.is_file() and p.name!="complete.json"}: errors.append("unit file set")
        for target in path.glob("*.npz"):
            with np.load(target,allow_pickle=False) as z:
                for key in z.files:
                    if np.asarray(z[key]).dtype==object: errors.append(f"object dtype {target.name}:{key}")
    except Exception as exc: errors.append(f"invalid unit {type(exc).__name__}")
    return UnitValidation(not errors,tuple(errors))


def decide_unit_action(path,expected_metadata):
    path=Path(path)
    if path.with_name(path.name+".building").exists(): return UnitAction.STOP_INVALID
    if not path.exists(): return UnitAction.COMPUTE
    return UnitAction.SKIP_VERIFIED if validate_unit(path,expected_metadata).valid else UnitAction.STOP_INVALID


def publish_unit_atomic(path,payload,metadata):
    path=Path(path); building=path.with_name(path.name+".building")
    if path.exists() or building.exists(): raise FileExistsError(path)
    building.mkdir(parents=True)
    _atomic_npz(building/"arrays.npz",payload); _atomic_json(building/"metadata.json",metadata)
    files={p.name:_sha(p) for p in building.iterdir() if p.is_file()}
    _atomic_json(building/"complete.json",{"schema":SCHEMA,"files":files})
    os.rename(building,path)
    if not validate_unit(path,metadata).valid: raise ValueError("published unit invalid")


def _families(manifest):
    first={}
    for row in manifest.rows: first.setdefault(row["unique_feature_id"],row["family"])
    return [first[uid] for uid in manifest.unique_ids]


def _ref_payload(reference,prefix):
    offsets=[0]; pieces=[]
    for x in reference.sorted_values: pieces.append(x); offsets.append(offsets[-1]+len(x))
    return {f"{prefix}_values":np.concatenate(pieces) if pieces else np.array([],float),
            f"{prefix}_offsets":np.asarray(offsets,np.int64),f"{prefix}_counts":reference.counts,
            f"{prefix}_fingerprint":np.asarray(reference.fingerprint)}


def _reference_bounds(reference):
    lo=np.full(len(reference.sorted_values),np.nan); hi=lo.copy()
    for j,x in enumerate(reference.sorted_values):
        if len(x): lo[j]=x[0]; hi[j]=x[-1]
    return lo,hi


def _code_hashes():
    modules=(data_module,split_module,normalize_module,views_module,audit_module)
    result={Path(m.__file__).stem+"_sha256":_sha(m.__file__) for m in modules}
    result["run_prep_sha256"]=_sha(__file__)
    return result


def make_run_identity(data,manifest):
    return {"schema":SCHEMA,"g2b1_cache_fingerprint":data.fingerprint,"g2b1_inventory_fingerprint":data.provenance.get("inventory_fingerprint"),
            "manifest_sha256":manifest.sha256,"spec_sha256":data.provenance.get("spec_sha256"),"formula":"(L+0.5E)/n",
            "levels":256,"rounding":"half_up","python_version":platform.python_version(),"numpy_version":np.__version__,**_code_hashes()}


def _unit_metadata(run_identity,fold,protocol):
    return {"schema":SCHEMA,"holdout_site":fold.holdout_site,"train_site":fold.train_site,"unused_site":fold.unused_site,
            "train_count":len(fold.train_indices),"test_count":len(fold.test_indices),"protocol":protocol,
            "run_identity_sha256":hashlib.sha256(_json_bytes(run_identity)).hexdigest(),"levels":256,"rounding":"half_up"}


def _write_full_unit(path,metadata,fold,protocol,manifest,train,test,train_source,test_source):
    building=path.with_name(path.name+".building")
    if path.exists() or building.exists(): raise FileExistsError(path)
    building.mkdir(parents=True); families=_families(manifest)
    train_ref=fit_rank_reference(train,train_source)
    train_pct,train_rank=apply_rank_reference(train,train_source,train_ref)
    if protocol=="transductive":
        test_ref=fit_rank_reference(test,test_source); test_pct,test_rank=apply_rank_reference(test,test_source,test_ref)
    elif protocol=="inductive":
        test_ref=train_ref; test_pct,test_rank=apply_rank_reference(test,test_source,train_ref)
    else: raise ValueError("protocol")
    train_q,train_levels=quantize_rank(train_pct,train_rank); test_q,test_levels=quantize_rank(test_pct,test_rank)
    train_views,train_view_masks=get_views(train_q,train_rank,manifest); test_views,test_view_masks=get_views(test_q,test_rank,manifest)
    lo_train,hi_train=_reference_bounds(train_ref); lo_test,hi_test=_reference_bounds(test_ref)
    train_audit=audit_rank(train,train_pct,train_q,train_source,train_rank,{"levels":train_levels,"feature_ids":manifest.unique_ids,"families":families,"reference_n":train_ref.counts,"reference_min":lo_train,"reference_max":hi_train})
    test_audit=audit_rank(test,test_pct,test_q,test_source,test_rank,{"levels":test_levels,"feature_ids":manifest.unique_ids,"families":families,"reference_n":test_ref.counts,"reference_min":lo_test,"reference_max":hi_test})
    for role,rows in (("train",train_audit),("test",test_audit)):
        for row in rows: row.update({"holdout_site":fold.holdout_site,"train_site":fold.train_site,"protocol":protocol,"role":role})
    _atomic_npz(building/"indices.npz",{"train_indices":fold.train_indices,"test_indices":fold.test_indices,"unused_indices":fold.unused_indices})
    _atomic_npz(building/"percentile.npz",{"train_percentile":train_pct,"test_percentile":test_pct,"train_source_valid_mask":train_source,"test_source_valid_mask":test_source,"train_rank_valid_mask":train_rank,"test_rank_valid_mask":test_rank})
    _atomic_npz(building/"quantized.npz",{"train_quantized":train_q,"test_quantized":test_q,"train_levels":train_levels,"test_levels":test_levels})
    view_payload={}
    for view in manifest.view_sizes:
        view_payload[f"train_{view}_values"]=train_views[view]; view_payload[f"train_{view}_rank_valid_mask"]=train_view_masks[view]
        view_payload[f"test_{view}_values"]=test_views[view]; view_payload[f"test_{view}_rank_valid_mask"]=test_view_masks[view]
    _atomic_npz(building/"views.npz",view_payload)
    ref_payload=_ref_payload(train_ref,"train")
    if protocol=="transductive": ref_payload.update(_ref_payload(test_ref,"test"))
    _atomic_npz(building/"reference.npz",ref_payload)
    _atomic_csv(building/"rank_audit.csv",train_audit+test_audit); _atomic_json(building/"metadata.json",metadata)
    files={p.name:_sha(p) for p in building.iterdir() if p.is_file()}
    _atomic_json(building/"complete.json",{"schema":SCHEMA,"files":files}); os.rename(building,path)
    result=validate_unit(path,metadata)
    if not result.valid: raise ValueError("unit publication: "+";".join(result.errors))
    return {"train_reference_counts":train_ref.counts,"test_reference_counts":test_ref.counts,
            "train_valid":int(train_rank.sum()),"test_valid":int(test_rank.sum())}


def prepare_fold(config):
    data=config["data"]; manifest=config["manifest"]; fold=config["fold"]; protocol=config["protocol"]
    path=Path(config["output_root"])/f"holdout_{fold.holdout_site}"/protocol
    metadata=_unit_metadata(config["run_identity"],fold,protocol); action=decide_unit_action(path,metadata)
    if action==UnitAction.STOP_INVALID: raise ValueError(f"STOP_INVALID {path}")
    if action==UnitAction.SKIP_VERIFIED: return {"action":action.value}
    path.parent.mkdir(parents=True,exist_ok=True)
    result=_write_full_unit(path,metadata,fold,protocol,manifest,data.values[fold.train_indices],data.values[fold.test_indices],data.source_valid_mask[fold.train_indices],data.source_valid_mask[fold.test_indices])
    return {"action":action.value,**result}


def _load_coverage(path):
    with Path(path).open(newline="",encoding="utf-8") as f: return [r for r in csv.DictReader(f) if r["site"]!="Overall"]


def preflight(cache_root,manifest_path,coverage_path,output_root):
    manifest=load_manifest(manifest_path,MANIFEST_SHA); data=load_features(cache_root,manifest); validate_coverage(data,_load_coverage(coverage_path))
    folds=make_loso(data.sites); identity=make_run_identity(data,manifest)
    output=Path(output_root).resolve(strict=False); source=Path(cache_root).resolve()
    if output==source or output in source.parents or source in output.parents: raise ValueError("source/output overlap")
    parent=output.parent
    while not parent.exists(): parent=parent.parent
    estimated=4*data.values.nbytes
    if shutil.disk_usage(parent).free<estimated*3: raise ValueError("disk safety margin")
    return {"data":data,"manifest":manifest,"folds":folds,"run_identity":identity,"output_root":output,
            "disk_free":shutil.disk_usage(parent).free,"estimated_bytes":estimated}


def _write_or_match(path,payload):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=payload: raise ValueError(f"identity mismatch {path.name}")
    else: _atomic_json(path,payload)


def _init_run(pf):
    root=pf["output_root"]; root.mkdir(parents=True,exist_ok=True)
    _write_or_match(root/"run_identity.json",pf["run_identity"])
    counts={s:int(np.sum(pf["data"].sites==s)) for s in HOLDOUTS}
    _write_or_match(root/"data_manifest.json",{"records":6600,"features":807,"site_counts":counts,"cache_fingerprint":pf["data"].fingerprint,"feature_ids":pf["data"].feature_ids})
    split_payload={h:{"train_site":f.train_site,"test_site":h,"unused_site":f.unused_site,"train_count":len(f.train_indices),"test_count":len(f.test_indices),"unused_count":len(f.unused_indices)} for h,f in pf["folds"].items()}
    _write_or_match(root/"split_manifest.json",split_payload)
    raw_path=root/"reports"/"raw_feature_audit.csv"
    if not raw_path.exists():
        families=_families(pf["manifest"]); rows=[]
        for site in HOLDOUTS:
            selected=pf["data"].sites==site; rows+=audit_raw(pf["data"].values[selected],pf["data"].source_valid_mask[selected],pf["data"].feature_ids,families,site)
        _atomic_csv(raw_path,rows)


def verify_all(pf,require_all=True):
    root=pf["output_root"]; errors=[]; verified=0
    for holdout,fold in pf["folds"].items():
        for protocol in PROTOCOLS:
            path=root/f"holdout_{holdout}"/protocol; meta=_unit_metadata(pf["run_identity"],fold,protocol); result=validate_unit(path,meta)
            if not result.valid:
                if require_all: errors.extend(f"{holdout}/{protocol}:{x}" for x in result.errors)
                continue
            with np.load(path/"percentile.npz",allow_pickle=False) as p, np.load(path/"quantized.npz",allow_pickle=False) as q, np.load(path/"views.npz",allow_pickle=False) as v:
                for role,n in (("train",len(fold.train_indices)),("test",len(fold.test_indices))):
                    pct=np.asarray(p[f"{role}_percentile"]); rank=np.asarray(p[f"{role}_rank_valid_mask"]); source=np.asarray(p[f"{role}_source_valid_mask"])
                    quant=np.asarray(q[f"{role}_quantized"]); levels=np.asarray(q[f"{role}_levels"])
                    if pct.shape!=(n,807) or rank.shape!=pct.shape or source.shape!=pct.shape or quant.shape!=pct.shape or levels.shape!=pct.shape: errors.append(f"{holdout}/{protocol}/{role}:shape")
                    if np.any(rank&~source) or not np.isnan(pct[~rank]).all() or not np.isnan(quant[~rank]).all(): errors.append(f"{holdout}/{protocol}/{role}:mask")
                    if np.any(np.abs(quant[rank]-pct[rank])>1/510+1e-14): errors.append(f"{holdout}/{protocol}/{role}:grid")
                    vv,vm=get_views(quant,rank,pf["manifest"])
                    for view in pf["manifest"].view_sizes:
                        if not np.array_equal(vv[view],v[f"{role}_{view}_values"],equal_nan=True) or not np.array_equal(vm[view],v[f"{role}_{view}_rank_valid_mask"]): errors.append(f"{holdout}/{protocol}/{role}/{view}")
            verified+=1
    if errors: raise ValueError(";".join(errors[:20]))
    if require_all:
        for holdout in HOLDOUTS:
            with np.load(root/f"holdout_{holdout}"/"transductive"/"percentile.npz",allow_pickle=False) as t, np.load(root/f"holdout_{holdout}"/"inductive"/"percentile.npz",allow_pickle=False) as i:
                compare_protocols(np.asarray(t["train_percentile"]),np.asarray(i["train_percentile"]),np.asarray(t["test_percentile"]),np.asarray(i["test_percentile"]),np.asarray(t["train_rank_valid_mask"]),np.asarray(t["test_rank_valid_mask"])&np.asarray(i["test_rank_valid_mask"]))
    return {"verified_units":verified,"errors":0}


def _public_rows(root):
    allowed=("feature_id","family","n_total","n_source_valid","n_rank_valid","reference_n","reference_unavailable_count","n_percentile_unique","n_quantized_unique","occupied_bins","occupied_bin_fraction","max_abs_quantization_error","mean_abs_quantization_error","new_tie_group_count","distinct_rank_levels_collapsed","zero_level_count","top_level_count","percentile_min","percentile_max","quantized_min","quantized_max","below_reference_count","above_reference_count","holdout_site","train_site","protocol","role")
    rows=[]
    for holdout in HOLDOUTS:
        for protocol in PROTOCOLS:
            with (Path(root)/f"holdout_{holdout}"/protocol/"rank_audit.csv").open(newline="",encoding="utf-8") as f:
                for row in csv.DictReader(f): rows.append({k:row[k] for k in allowed})
    return rows


def _write_public_csv(path,rows):
    path=Path(path)
    if path.exists():
        with path.open(newline="",encoding="utf-8") as f: existing=list(csv.DictReader(f))
        expected=[{key:str(value) for key,value in row.items()} for row in rows]
        if existing!=expected: raise ValueError("public aggregate exists with different content")
        return
    _atomic_csv(path,rows)


def render_report(path,pf,rows,elapsed):
    by={(r["holdout_site"],r["protocol"],r["role"]):[] for r in rows}
    for r in rows: by[(r["holdout_site"],r["protocol"],r["role"])].append(r)
    lines=["# G3-A LOSO Rank256 Audit Report","","`G3A_STATUS = SUBMITTED_FOR_REVIEW`","","## Inputs and splits","",
           f"- Parent SHA: `bddf55e15033b7872588f7a630be4c0a178c06a4`",f"- G2-B1 cache fingerprint: `{pf['data'].fingerprint}`","- Records/features/views: 6,600 / 807 / 827.","- Sites: I0002=319, I0006=1,142, S0001=5,139.","",
           "| Holdout | Train | Train N | Test N | Unused |","|---|---|---:|---:|---|"]
    for h,f in pf["folds"].items(): lines.append(f"| {h} | {f.train_site} | {len(f.train_indices)} | {len(f.test_indices)} | {f.unused_site} |")
    lines += ["","## Six preprocessing units","","| Holdout | Protocol | Role | Source valid | Rank valid | Reference-unavailable | Median occupied bins | Max quantization error |","|---|---|---|---:|---:|---:|---:|---:|"]
    for key,items in by.items():
        source=sum(int(r["n_source_valid"]) for r in items); rank=sum(int(r["n_rank_valid"]) for r in items); unavailable=sum(int(r["reference_unavailable_count"]) for r in items)
        bins=float(np.median([int(r["occupied_bins"]) for r in items])); error=max(float(r["max_abs_quantization_error"]) for r in items)
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {source} | {rank} | {unavailable} | {bins:.1f} | {error:.8f} |")
    def feature(uid,holdout="I0002"):
        return [r for r in rows if r["feature_id"]==uid and r["holdout_site"]==holdout]
    bmi=feature("demo_bmi"); entropy=feature("caisr_sleep_mean_stage_entropy"); male=feature("demo_male")
    lines += ["","## S0001 to I0002 focused audit","",
              "- Train reference cohort: S0001=5,139; Transductive target reference cohort: I0002=319.",
              "- Transductive and Inductive training matrices are exactly equal; the test protocols differ only by reference cohort.",
              f"- BMI audit rows: {len(bmi)}; source missingness is retained and no value is filled.",
              f"- Stage entropy audit rows: {len(entropy)}; I0002 remains entirely missing and the feature is retained.",
              f"- Male binary-feature audit rows: {len(male)}; ties use the same average-rank formula as continuous features.",
              "- All 807 columns were audited; no feature-specific ranking rule was introduced.","",
              "## Validation","",
              "- Percentiles are finite on rank-valid positions, within [0,1], tie-consistent, monotone and order-independent.",
              "- Rank256 levels are integers 0-255, missing sentinel is -1, and float outputs lie on k/255.",
              "- Quantization error never exceeds 1/510 plus numerical tolerance.",
              "- Source-valid and rank-valid masks remain separate; reference-unavailable outputs remain NaN.",
              "- Frozen five-view mappings and repeated Demo5 values/masks were verified for every unit.",
              "- G2-B1 coverage counts matched all three sites and all 807 columns.",
              f"- Full G3 build elapsed: {elapsed:.2f} seconds.","",
              "## Interpretation boundary","",
              "Occupied-bin counts and additional ties reflect deterministic quantization, not model performance. BMI site missingness and stage-entropy scarcity remain visible. No feature, sample, protocol, or formula was changed based on these findings.","",
              "- G3A_LOSO_SPLITS: **FROZEN**","- G3A_TRANSDUCTIVE: **VERIFIED**","- G3A_INDUCTIVE: **VERIFIED**","- G3A_256_LEVEL_GRID: **VERIFIED**","- G3A_FEATURE_AUDIT: **COMPLETE**","- G2B1_FULL_CACHE: **UNCHANGED**","- G3B_MODEL_INPUT_ADAPTER: **NOT STARTED**","- G4_MODEL_TRAINING: **NOT STARTED**","- G5_RANK_ENSEMBLE: **NOT STARTED**"]
    text="\n".join(lines)+"\n"
    forbidden=("/database/",".npz","BidsFolder","SessionID","BDSPPatientID","record_id")
    if any(x in text for x in forbidden): raise ValueError("public report privacy")
    target=Path(path)
    if target.exists():
        existing=target.read_text(encoding="utf-8")
        if any(x in existing for x in forbidden): raise ValueError("existing public report privacy")
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=target.parent,prefix=target.name+".",suffix=".tmp",delete=False) as f:
        temporary=Path(f.name); f.write(text); f.flush(); os.fsync(f.fileno())
    try: os.link(temporary,target)
    finally:
        if temporary.exists(): temporary.unlink()


def run(args):
    pf=preflight(args.cache_root,args.manifest,args.coverage_source,args.output_root)
    if args.preflight_only:
        return {"records":len(pf["data"].values),"features":807,"fingerprint":pf["data"].fingerprint,"disk_free":pf["disk_free"]}
    if args.verify_only:
        return verify_all(pf,True)
    _init_run(pf); started=time.perf_counter(); targets=("I0002",) if args.smoke else HOLDOUTS; results={}
    for holdout in targets:
        for protocol in PROTOCOLS:
            config={**pf,"fold":pf["folds"][holdout],"protocol":protocol}
            results[f"{holdout}/{protocol}"]=prepare_fold(config)
    verify_all(pf,require_all=not args.smoke)
    if not args.smoke:
        rows=_public_rows(pf["output_root"]); _write_public_csv(args.public_csv,rows); render_report(args.report,pf,rows,time.perf_counter()-started)
    return {"units":results,"verified":len(targets)*2,"complete":not args.smoke}


def build_parser():
    p=argparse.ArgumentParser(description="G3-A LOSO Rank256 preprocessing only")
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only",action="store_true"); modes.add_argument("--smoke",action="store_true")
    modes.add_argument("--build",action="store_true"); modes.add_argument("--verify-only",action="store_true")
    p.add_argument("--cache-root",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True); p.add_argument("--coverage-source",type=Path,required=True)
    p.add_argument("--public-csv",type=Path,required=True); p.add_argument("--report",type=Path,required=True)
    return p


def main(argv=None):
    result=run(build_parser().parse_args(argv)); print(json.dumps(result,indent=2,sort_keys=True,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
    return result


if __name__=="__main__": main()
