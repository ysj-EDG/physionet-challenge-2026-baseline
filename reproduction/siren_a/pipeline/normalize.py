"""Pure percentile-ranking and Rank256 mathematics for G3-A."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RankReference:
    sorted_values: tuple
    counts: np.ndarray
    fingerprint: str


def _validate(values, mask):
    values=np.asarray(values,dtype=np.float64); mask=np.asarray(mask,dtype=bool)
    if values.ndim!=2 or mask.shape!=values.shape: raise ValueError("values/mask shape")
    if np.isinf(values).any() or not np.isfinite(values[mask]).all(): raise ValueError("nonfinite valid input")
    return values,mask


def fit_rank_reference(train_values, train_mask):
    values,mask=_validate(train_values,train_mask); columns=[]; h=hashlib.sha256()
    for j in range(values.shape[1]):
        selected=np.sort(values[mask[:,j],j],kind="mergesort").astype(np.float64,copy=False)
        selected=np.ascontiguousarray(selected); columns.append(selected)
        h.update(np.asarray([len(selected)],dtype=np.int64).tobytes()); h.update(selected.tobytes())
    counts=np.asarray([len(x) for x in columns],dtype=np.int64)
    return RankReference(tuple(columns),counts,h.hexdigest())


def apply_rank_reference(values, mask, reference):
    values,mask=_validate(values,mask)
    if values.shape[1]!=len(reference.sorted_values): raise ValueError("reference width")
    output=np.full(values.shape,np.nan,dtype=np.float64); rank_mask=np.zeros(values.shape,dtype=bool)
    for j,ref in enumerate(reference.sorted_values):
        n=len(ref)
        if not n: continue
        rows=np.flatnonzero(mask[:,j]); x=values[rows,j]
        left=np.searchsorted(ref,x,side="left"); right=np.searchsorted(ref,x,side="right")
        output[rows,j]=(left+0.5*(right-left))/n; rank_mask[rows,j]=True
    return output,rank_mask


def rank_by_site(values, mask, sites):
    values,mask=_validate(values,mask); sites=np.asarray(sites)
    if sites.shape!=(len(values),): raise ValueError("site shape")
    output=np.full(values.shape,np.nan,dtype=np.float64); rank_mask=np.zeros(values.shape,dtype=bool)
    for site in sorted(set(sites.tolist())):
        rows=np.flatnonzero(sites==site); ref=fit_rank_reference(values[rows],mask[rows])
        output[rows],rank_mask[rows]=apply_rank_reference(values[rows],mask[rows],ref)
    return output,rank_mask


def quantize_rank(percentiles, rank_mask):
    values=np.asarray(percentiles,dtype=np.float64); mask=np.asarray(rank_mask,dtype=bool)
    if values.ndim!=2 or mask.shape!=values.shape: raise ValueError("percentile/mask shape")
    if not np.isfinite(values[mask]).all() or np.isinf(values).any(): raise ValueError("invalid percentile")
    if np.any((values[mask]<0)|(values[mask]>1)): raise ValueError("percentile outside [0,1]")
    if not np.isnan(values[~mask]).all(): raise ValueError("invalid percentile must be NaN")
    levels=np.full(values.shape,-1,dtype=np.int16)
    levels[mask]=np.floor(255.0*values[mask]+0.5).astype(np.int16)
    output=np.full(values.shape,np.nan,dtype=np.float64); output[mask]=levels[mask].astype(np.float64)/255.0
    return output,levels
