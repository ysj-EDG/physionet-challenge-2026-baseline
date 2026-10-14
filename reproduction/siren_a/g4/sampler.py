"""Exact constrained age-by-label sampling for G4-A."""

from dataclasses import dataclass
import hashlib
from typing import Dict, Mapping, Tuple
import numpy as np


@dataclass(frozen=True)
class SampleResult:
    train_site: str
    seed: int
    target_n: int
    local_positions: np.ndarray
    global_indices: np.ndarray
    quotas: Mapping[Tuple[int, int], int]
    stratum_sizes: Mapping[Tuple[int, int], int]
    index_sha256: str


def _array_sha(array: np.ndarray) -> str:
    value = np.asarray(array, dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def target_size(n: int) -> int:
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("training size must be positive")
    return (7 * int(n) + 5) // 10


def allocate_quotas(stratum_sizes: Mapping[Tuple[int, int], int], target_n: int) -> Dict[Tuple[int, int], int]:
    if not stratum_sizes or any(not isinstance(key, tuple) or len(key) != 2 or int(size) <= 0
                                for key, size in stratum_sizes.items()):
        raise ValueError("strata must be nonempty with positive sizes")
    target = int(target_n)
    total = sum(int(size) for size in stratum_sizes.values())
    if target <= 0 or target > total:
        raise ValueError("target outside population")
    quotas = {key: max(1, (7 * int(size)) // 10) for key, size in stratum_sizes.items()}
    if sum(quotas.values()) > target:
        raise ValueError("minimum-one stratum constraint is infeasible")
    remaining = target - sum(quotas.values())
    order = sorted(stratum_sizes, key=lambda key: (-(7 * int(stratum_sizes[key]) % 10),
                                                   int(key[0]), int(key[1])))
    while remaining:
        progressed = False
        for key in order:
            if remaining == 0:
                break
            if quotas[key] < int(stratum_sizes[key]):
                quotas[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise ValueError("cannot allocate exact target without exceeding capacity")
    if sum(quotas.values()) != target or any(quotas[key] > int(stratum_sizes[key]) for key in quotas):
        raise ValueError("quota closure failed")
    return quotas


def _seed64(train_site: str, seed: int, group: int, label: int) -> int:
    material = f"siren-g4a-v1|{train_site}|{int(seed)}|{int(group)}|{int(label)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little", signed=False)


def sample_training_rows(train_site: str, seed: int, age_groups: np.ndarray, labels: np.ndarray,
                         global_indices: np.ndarray) -> SampleResult:
    groups = np.asarray(age_groups)
    y = np.asarray(labels)
    indices = np.asarray(global_indices)
    if groups.ndim != 1 or y.shape != groups.shape or indices.shape != groups.shape:
        raise ValueError("sampling inputs must be aligned one-dimensional arrays")
    if groups.dtype.kind not in "iu" or indices.dtype.kind not in "iu" or not np.all(np.isin(y, [0, 1])):
        raise ValueError("sampling groups/indices must be integer and labels binary")
    indices = indices.astype(np.int64, copy=False)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("global indices must be unique")
    keys = [(int(group), int(label)) for group, label in zip(groups, y)]
    sizes = {key: keys.count(key) for key in sorted(set(keys))}
    target = target_size(len(groups))
    quotas = allocate_quotas(sizes, target)
    chosen = []
    for key in sorted(sizes):
        candidates = np.flatnonzero((groups == key[0]) & (y == key[1])).astype(np.int64)
        rng = np.random.Generator(np.random.PCG64(_seed64(train_site, seed, *key)))
        chosen.extend(np.asarray(rng.choice(candidates, size=quotas[key], replace=False),
                                 dtype=np.int64).tolist())
    local = np.asarray(sorted(chosen), dtype=np.int64)
    selected_global = indices[local]
    if len(local) != target or len(np.unique(local)) != target:
        raise ValueError("sample size or uniqueness failure")
    return SampleResult(str(train_site), int(seed), target, local, selected_global,
                        quotas, sizes, _array_sha(selected_global))
