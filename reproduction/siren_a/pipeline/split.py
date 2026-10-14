"""Frozen three-site LOSO split construction."""
from dataclasses import dataclass
import numpy as np

SITES=("I0002","I0006","S0001")


@dataclass(frozen=True)
class LosoFold:
    holdout_site: str
    train_site: str
    unused_site: str
    train_indices: np.ndarray
    test_indices: np.ndarray
    unused_indices: np.ndarray


def select_train_site(holdout_site, sites):
    sites=np.asarray(sites)
    if holdout_site not in SITES or set(sites.tolist())!=set(SITES): raise ValueError("three frozen sites required")
    candidates=[s for s in SITES if s!=holdout_site]
    counts={s:int(np.sum(sites==s)) for s in candidates}
    ordered=sorted(candidates,key=lambda s:(-counts[s],s))
    return ordered[0],ordered[1]


def make_loso(sites):
    sites=np.asarray(sites)
    if sites.ndim!=1 or set(sites.tolist())!=set(SITES): raise ValueError("invalid LOSO sites")
    folds={}
    for holdout in SITES:
        train,unused=select_train_site(holdout,sites)
        fold=LosoFold(holdout,train,unused,np.flatnonzero(sites==train),np.flatnonzero(sites==holdout),np.flatnonzero(sites==unused))
        if np.intersect1d(fold.train_indices,fold.test_indices).size: raise ValueError("LOSO overlap")
        folds[holdout]=fold
    return folds
