"""Independent average-tie percentile and Rank256 verification mathematics."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RankReference:
    unique_values: tuple
    frequencies: tuple
    sorted_values: tuple
    counts: np.ndarray
    fingerprint: str


def _validate(values,mask):
    values=np.asarray(values,dtype=np.float64)
    mask=np.asarray(mask)
    if values.ndim!=2 or mask.shape!=values.shape or mask.dtype!=np.dtype(bool):
        raise ValueError("values/mask schema")
    if np.isinf(values).any() or not np.isfinite(values[mask]).all():
        raise ValueError("nonfinite valid input")
    if not np.isnan(values[~mask]).all():
        raise ValueError("invalid input must be NaN")
    return values,mask


def build_reference(values,mask):
    values,mask=_validate(values,mask)
    unique=[]; frequencies=[]; sorted_columns=[]; counts=[]; digest=hashlib.sha256()
    for j in range(values.shape[1]):
        column=np.sort(values[mask[:,j],j],kind="mergesort").astype(np.float64,copy=False)
        column=np.ascontiguousarray(column)
        u,c=np.unique(column,return_counts=True)
        u=np.ascontiguousarray(u,dtype=np.float64)
        c=np.ascontiguousarray(c,dtype=np.int64)
        unique.append(u); frequencies.append(c); sorted_columns.append(column); counts.append(len(column))
        digest.update(np.asarray([len(column)],dtype=np.int64).tobytes())
        digest.update(column.tobytes())
    return RankReference(tuple(unique),tuple(frequencies),tuple(sorted_columns),
                         np.asarray(counts,dtype=np.int64),digest.hexdigest())


def rank_values(values,mask,reference):
    values,mask=_validate(values,mask)
    if values.shape[1]!=len(reference.unique_values):
        raise ValueError("reference width")
    output=np.full(values.shape,np.nan,dtype=np.float64)
    rank_mask=np.zeros(values.shape,dtype=bool)
    for j,(unique,frequency) in enumerate(zip(reference.unique_values,reference.frequencies)):
        n=int(reference.counts[j])
        if n==0:
            continue
        prefix=np.concatenate((np.array([0],dtype=np.int64),np.cumsum(frequency,dtype=np.int64)))
        rows=np.flatnonzero(mask[:,j]); query=values[rows,j]
        insertion=np.searchsorted(unique,query,side="left")
        left=prefix[insertion]
        equal=np.zeros(len(query),dtype=bool)
        inside=insertion<len(unique)
        equal[inside]=unique[insertion[inside]]==query[inside]
        ties=np.zeros(len(query),dtype=np.int64)
        ties[equal]=frequency[insertion[equal]]
        output[rows,j]=(left+0.5*ties)/n
        rank_mask[rows,j]=True
    return output,rank_mask


def quantize_values(percentile,rank_mask):
    percentile=np.asarray(percentile,dtype=np.float64)
    rank_mask=np.asarray(rank_mask)
    if percentile.ndim!=2 or rank_mask.shape!=percentile.shape or rank_mask.dtype!=np.dtype(bool):
        raise ValueError("percentile/mask schema")
    if np.isinf(percentile).any() or not np.isfinite(percentile[rank_mask]).all():
        raise ValueError("invalid percentile")
    if not np.isnan(percentile[~rank_mask]).all():
        raise ValueError("invalid percentile must be NaN")
    if np.any((percentile[rank_mask]<0)|(percentile[rank_mask]>1)):
        raise ValueError("percentile outside [0,1]")
    levels=np.full(percentile.shape,-1,dtype=np.int16)
    levels[rank_mask]=np.floor(255.0*percentile[rank_mask]+0.5).astype(np.int16)
    quantized=np.full(percentile.shape,np.nan,dtype=np.float64)
    quantized[rank_mask]=levels[rank_mask].astype(np.float64)/255.0
    return quantized,levels


def _flatten(reference):
    offsets=[0]; parts=[]
    for column in reference.sorted_values:
        parts.append(column); offsets.append(offsets[-1]+len(column))
    values=np.concatenate(parts) if parts else np.array([],dtype=np.float64)
    return values,np.asarray(offsets,dtype=np.int64)


def _scalar_string(value,key):
    value=np.asarray(value)
    if value.shape!=():
        raise ValueError(f"{key} scalar")
    return str(value.item())


def check_reference(saved,reference,prefix):
    keys=(f"{prefix}_values",f"{prefix}_offsets",f"{prefix}_counts",f"{prefix}_fingerprint")
    if any(key not in saved for key in keys):
        raise ValueError(f"{prefix} reference keys")
    values=np.asarray(saved[keys[0]]); offsets=np.asarray(saved[keys[1]]); counts=np.asarray(saved[keys[2]])
    if values.dtype!=np.float64 or offsets.dtype!=np.int64 or counts.dtype!=np.int64:
        raise ValueError(f"{prefix} reference dtype")
    if offsets.shape!=(len(reference.counts)+1,) or counts.shape!=reference.counts.shape:
        raise ValueError(f"{prefix} reference shape")
    if offsets[0]!=0 or np.any(np.diff(offsets)<0) or offsets[-1]!=len(values):
        raise ValueError(f"{prefix} offsets")
    if not np.array_equal(np.diff(offsets),counts):
        raise ValueError(f"{prefix} offset counts")
    if np.isinf(values).any() or not np.isfinite(values).all():
        raise ValueError(f"{prefix} nonfinite reference")
    for start,end in zip(offsets[:-1],offsets[1:]):
        if np.any(np.diff(values[start:end])<0):
            raise ValueError(f"{prefix} unsorted reference")
    expected_values,expected_offsets=_flatten(reference)
    if not np.array_equal(values,expected_values):
        raise ValueError(f"{prefix} reference values")
    if not np.array_equal(offsets,expected_offsets) or not np.array_equal(counts,reference.counts):
        raise ValueError(f"{prefix} reference counts")
    if _scalar_string(saved[keys[3]],keys[3])!=reference.fingerprint:
        raise ValueError(f"{prefix} reference fingerprint")
    return True


def check_quantized(saved_levels,saved_quantized,percentile,rank_mask):
    expected_quantized,expected_levels=quantize_values(percentile,rank_mask)
    levels=np.asarray(saved_levels); quantized=np.asarray(saved_quantized)
    if levels.dtype!=np.int16 or quantized.dtype!=np.float64:
        raise ValueError("quantized dtype")
    if levels.shape!=expected_levels.shape or quantized.shape!=expected_quantized.shape:
        raise ValueError("quantized shape")
    if not np.array_equal(levels,expected_levels):
        raise ValueError("Rank256 level formula mismatch")
    if not np.array_equal(quantized,expected_quantized,equal_nan=True):
        raise ValueError("Rank256 float formula mismatch")
    if np.any(levels[~rank_mask]!=-1) or np.any((levels[rank_mask]<0)|(levels[rank_mask]>255)):
        raise ValueError("Rank256 level domain")
    if np.any(np.abs(quantized[rank_mask]-percentile[rank_mask])>1/510+1e-14):
        raise ValueError("Rank256 error bound")
    return True
