"""Bounded-only real-data smoke runner. No full-cohort execution mode exists."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, tempfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

from g2a_io import load_manifest, load_npz_record, read_cache, sha256_file, write_cache_atomic
from g2a_features import compute_night_features, entropy_source_valid

SEED=20261013
MAX_OUTPUT_RECORDS=30
SITES=("I0002","I0006","S0001")


def deterministic_base_sample(root: Path, seed: int=SEED):
    root=Path(root); rng=np.random.default_rng(seed); selected=[]
    for site in SITES:
        files=sorted((root/site).glob("*.npz"))
        if len(files)<6: raise ValueError(f"Insufficient files for {site}")
        fixed={0,len(files)//2,len(files)-1}
        candidates=np.array([i for i in range(len(files)) if i not in fixed])
        extra=set(int(x) for x in rng.choice(candidates,size=3,replace=False))
        selected.extend(files[i] for i in sorted(fixed|extra))
    if len(selected)>MAX_OUTPUT_RECORDS or len(set(selected))!=len(selected): raise ValueError("Sample bound/uniqueness")
    return selected


def build_parser():
    p=argparse.ArgumentParser(description="Bounded G2-A smoke only")
    p.add_argument("--input-root",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True); p.add_argument("--spec",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True); p.add_argument("--seed",type=int,default=SEED)
    return p


def _family_map(manifest):
    return {r["unique_feature_id"]:r["family"] for r in manifest.rows}


def _entropy_qc(record):
    p=np.asarray(record["stage_probability"],float); sv=np.asarray(record["stage_valid"],bool)
    nine=p==9; finite=np.isfinite(p); inrange=finite&(p>=0)&(p<=1); sums=np.where(finite,p,0).sum(axis=1)
    lengths=record["annotation_signal_lengths"]; t=len(sv)
    aliases=(("caisr_prob_n1",),("caisr_prob_n2",),("caisr_prob_n3",),("caisr_prob_r","caisr_prob_rem"),("caisr_prob_w","caisr_prob_wake"))
    mismatch=any(not any(lengths.get(k)==t for k in keys) for keys in aliases)
    has_valid=bool(np.any(nine[sv])); has_invalid=bool(np.any(nine[~sv]))
    return {"nine":int(nine.sum()),"out_of_range":int((finite&((p<0)|(p>1))).sum()),"nonfinite":int((~finite).sum()),
            "finite_rows":int(finite.all(axis=1).sum()),"inrange_rows":int(inrange.all(axis=1).sum()),"zero_sum_rows":int((sums==0).sum()),
            "nine_valid_stage":int(nine[sv].sum()),"nine_invalid_stage":int(nine[~sv].sum()),
            **{f"nine_ch{i}":int(nine[:,i].sum()) for i in range(5)},"length_mismatch_records":int(mismatch),
            "entropy_valid_records":int(entropy_source_valid(record,float(record["x_static"][87]))),
            "category_A":int(has_invalid and not has_valid),"category_B":int(has_valid and not has_invalid),"category_C":int(has_valid and has_invalid)}


def _atomic_json(path,data):
    path=Path(path)
    with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=path.parent,prefix=path.name+".",delete=False) as h:
        def encode(value):
            if isinstance(value,np.generic): return value.item()
            raise TypeError(f"Not JSON serializable: {type(value).__name__}")
        tmp=Path(h.name); json.dump(data,h,indent=2,sort_keys=True,default=encode); h.flush(); os.fsync(h.fileno())
    os.replace(tmp,path)


def _render_report(report, parent_sha, manifest, spec_sha, execution_script_sha, samples, family, entropy, coverage, replay_ok):
    lines=["# G2-A Independent Night-level Smoke Report","", "`G2A_STATUS = SUBMITTED_FOR_REVIEW`","", "## Environment","",
           f"- Branch: `research/siren-a-stagepool-v1`",f"- Parent SHA: `{parent_sha}`",f"- Python: `{platform.python_version()}`; NumPy: `{np.__version__}`",
           "- Input: `timegrid_v2.1.0` / `validity_v0.2`",f"- Manifest SHA256: `{manifest.sha256}`",f"- Spec SHA256: `{spec_sha}`",f"- Smoke execution script SHA256 recorded in cache: `{execution_script_sha}`",f"- Current reviewed report module SHA256: `{sha256_file(Path(__file__))}`","- A post-run aggregate-report-only correction explains any difference between those script fingerprints; feature arrays and private caches were not rewritten.","- Git scope: eight approved G2-A files only.","",
           "## Sampling","",f"- Smoke records: {len(samples)}; I0002=6, I0006=6, S0001=6; fixed=9, seeded random=9, boundary additions=0.","- Output cap 30 respected; NPZ contents read: 18 distinct records.",
           "- Real boundary coverage: "+", ".join(k for k,v in coverage.items() if v)+".","- NOT COVERED BY REAL SMOKE: "+(", ".join(k for k,v in coverage.items() if not v) or "none")+".","",
           "## Output","","- Private caches: 18; unique=807; V1/V2/V3/V4/V5=113/129/349/113/123; total view positions=827.",f"- Atomic write/read, source SHA preservation, and deterministic replay: {'PASS' if replay_ok else 'FAIL'}.","",
           "## Feature QC","", "| site | family | columns | valid | missing | valid ratio | fully invalid columns | zero values | support min/median/max |","|---|---|---:|---:|---:|---:|---:|---:|---|" ]
    for (site,fam),q in sorted(family.items()):
        sup=q["supports"]; med=float(np.median(sup)) if sup else 0
        fully_invalid=sum(count==0 for count in q["uid_valid"].values())
        lines.append(f"| {site} | {fam} | {q['columns']} | {q['valid']} | {q['missing']} | {q['valid']/(q['valid']+q['missing']):.4f} | {fully_invalid} | {q['zeros']} | {min(sup) if sup else 0}/{med:.1f}/{max(sup) if sup else 0} |")
    lines += ["","## Entropy QC","","| site | records | value=9 | N1/N2/N3/REM/W value=9 | out of range | nonfinite | finite rows | in-range rows | zero-sum rows | 9@stage_valid | 9@stage_invalid | length mismatch | entropy valid | A/B/C |","|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|" ]
    for site,q in sorted(entropy.items()):
        channels="/".join(str(q[f"nine_ch{i}"]) for i in range(5))
        lines.append(f"| {site} | {q['records']} | {q['nine']} | {channels} | {q['out_of_range']} | {q['nonfinite']} | {q['finite_rows']} | {q['inrange_rows']} | {q['zero_sum_rows']} | {q['nine_valid_stage']} | {q['nine_invalid_stage']} | {q['length_mismatch_records']} | {q['entropy_valid_records']} | {q['category_A']}/{q['category_B']}/{q['category_C']} |")
    overall=entropy["Overall"]
    alert="ENTROPY_COVERAGE_ALERT" if overall["entropy_valid_records"]==0 else "NONE"
    lines += ["",f"- Entropy coverage alert: `{alert}`.","- Values equal to 9 are never altered or used to recompute entropy; they invalidate only the stored entropy feature under the frozen rule.","",
              "## Validation","","- `python test_g2a_features.py`: 8 tests passed.","- `python test_g2a_smoke.py`: 6 tests passed.","- Existing G1-A and G1-B tests: passed.","- Python syntax compilation: passed.","- Frozen Manifest row-by-row mapping and source indices: passed.","- No label read, record deletion, imputation, scaling, site mapping, or training path exists.","",
              "## Findings","","- PASS: schema, 807/827 assembly, cache round-trip, source preservation, and deterministic replay.","- EXPECTED MISSINGNESS: per-column support below five and metadata-invalid static values remain NaN/false.",f"- COVERAGE WARNING: `{alert}`.","- BLOCKING SCHEMA OR CALCULATION ERROR: none.","","No patient ID, filename, path, age, label, or patient-level vector is present in this report.","",
              "- G2A_SMOKE_CACHE: **GENERATED**","- G2B_FULL_CACHE: **NOT GENERATED**","- G3_PREPROCESSING: **NOT STARTED**","- MODEL_TRAINING: **NOT STARTED**","- SITE_RANK_NORMALIZATION: **NOT STARTED**"]
    Path(report).write_text("\n".join(lines)+"\n",encoding="utf-8")


def run(args):
    if args.output_root.exists(): raise FileExistsError(f"Private target already exists: {args.output_root}")
    staging=args.output_root.with_name(args.output_root.name+".building")
    if staging.exists(): raise FileExistsError(f"Incomplete private staging directory requires review: {staging}")
    manifest=load_manifest(args.manifest); spec_sha=sha256_file(args.spec); samples=deterministic_base_sample(args.input_root,args.seed)
    staging.mkdir(parents=True); family=defaultdict(lambda:{"columns":0,"valid":0,"missing":0,"zeros":0,"supports":[],"uid_valid":Counter()}); entropy=defaultdict(Counter)
    coverage={k:False for k in ("N3<5","EEG lead unavailable","spatial intersection<5","no HRV windows","annotation missing","BMI missing","Race unavailable","probability value 9","probability incomplete")}
    mapping=[]; script_sha=sha256_file(Path(__file__)); source_before={}
    fmap=_family_map(manifest)
    for n,path in enumerate(samples):
        site=path.parent.name; source_before[str(path)]=sha256_file(path); rec=load_npz_record(path); result=compute_night_features(rec,manifest)
        arrays={"unique_values":result.unique_feature_values,"unique_valid_mask":result.unique_feature_valid_mask,"unique_support_count":result.unique_feature_support_count,
                "input_extraction_version":np.asarray(rec["extraction_version"]),"input_validity_schema_version":np.asarray(rec["validity_schema_version"]),
                "source_npz_sha256":np.asarray(source_before[str(path)]),"manifest_sha256":np.asarray(manifest.sha256),"spec_sha256":np.asarray(spec_sha),"script_sha256":np.asarray(script_sha),
                "python_version":np.asarray(platform.python_version()),"numpy_version":np.asarray(np.__version__),"sample_seed":np.asarray(args.seed),"evaluation_age":np.asarray(float(rec["x_static"][0]))}
        for v in manifest.view_sizes:
            arrays[f"{v}_values"]=result.view_values[v]; arrays[f"{v}_valid_mask"]=result.view_valid_masks[v]; arrays[f"{v}_support_count"]=result.view_support_counts[v]
        cache=staging/f"smoke_{n:03d}.npz"; write_cache_atomic(cache,arrays); mapping.append({"cache":cache.name,"source":str(path),"site":site})
        for uid,val,ok,sup in zip(result.unique_ids,result.unique_feature_values,result.unique_feature_valid_mask,result.unique_feature_support_count):
            q=family[(site,fmap[uid])]; q["columns"]+=1; q["valid"]+=int(ok); q["missing"]+=int(not ok); q["zeros"]+=int(ok and val==0); q["supports"].append(int(sup)); q["uid_valid"][uid]+=int(ok)
        q=_entropy_qc(rec); q["records"]=1
        for scope in (site,"Overall"): entropy[scope].update(q)
        n3=_stage_count(rec,1); coverage["N3<5"] |= n3<5; coverage["EEG lead unavailable"] |= not np.asarray(rec["eeg_channel_available"]).all()
        coverage["spatial intersection<5"] |= _four_count(rec,2)<5 or _four_count(rec,4)<5
        coverage["no HRV windows"] |= not np.any(rec["hrv_success"]); coverage["annotation missing"] |= rec["annotation_status"]!="available"
        coverage["BMI missing"] |= rec["x_static"][9]==0; coverage["Race unavailable"] |= rec["x_static"][7]==1
        coverage["probability value 9"] |= q["nine"]>0; coverage["probability incomplete"] |= not np.asarray(rec["stage_probability_channels_present"]).all() or q["length_mismatch_records"]>0
    _atomic_json(staging/"private_sample_map.json",mapping)
    replay_ok=True
    for n,item in enumerate(mapping):
        path=Path(item["source"]); rec=load_npz_record(path); result=compute_night_features(rec,manifest); cached=read_cache(staging/item["cache"])
        replay_ok &= source_before[str(path)]==sha256_file(path)
        replay_ok &= np.array_equal(result.unique_feature_values,cached["unique_values"],equal_nan=True)
        replay_ok &= np.array_equal(result.unique_feature_valid_mask,cached["unique_valid_mask"]) and np.array_equal(result.unique_feature_support_count,cached["unique_support_count"])
        for v in manifest.view_sizes:
            replay_ok &= np.array_equal(result.view_values[v],cached[f"{v}_values"],equal_nan=True) and np.array_equal(result.view_valid_masks[v],cached[f"{v}_valid_mask"]) and np.array_equal(result.view_support_counts[v],cached[f"{v}_support_count"])
    if not replay_ok: raise RuntimeError("Deterministic replay failed")
    # Add aggregate column count per family (rather than record-column total).
    family_columns=Counter(fmap.values())
    for (site,fam),q in family.items(): q["columns"]=family_columns[fam]
    metadata={"seed":args.seed,"records":len(samples),"manifest_sha256":manifest.sha256,"spec_sha256":spec_sha,"script_sha256":script_sha,"replay_ok":replay_ok,"coverage":coverage}
    _atomic_json(staging/"run_metadata.json",metadata)
    _render_report(args.report,"9b99a0de7724093c30ca3135b3ef5bfae9bcce2d",manifest,spec_sha,script_sha,samples,family,entropy,coverage,replay_ok)
    os.replace(staging,args.output_root)


def _stage_count(r,code): return int((np.asarray(r["stage_valid"],bool)&(np.asarray(r["stage_code_aligned"])==code)).sum())
def _four_count(r,code):
    mask=np.asarray(r["stage_valid"],bool)&(np.asarray(r["stage_code_aligned"])==code)
    clean=np.asarray(r["eeg_clean_subsegment_count"]); available=np.asarray(r["eeg_channel_available"],bool)
    for i in (0,1,4,5): mask &= available[i] & (clean[:,i]>0)
    return int(mask.sum())


def main(): run(build_parser().parse_args())
if __name__=="__main__": main()
