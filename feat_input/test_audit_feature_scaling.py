#!/usr/bin/env python
"""Small deterministic checks for the read-only feature scaling audit."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_feature_scaling import Aggregate, build_schema, select_rows, weighted_quantile


def main():
    schema = build_schema()
    assert sum(x["branch"] == "X_seq" for x in schema) == 483
    assert sum(x["branch"] == "X_ecg" for x in schema) == 12
    assert sum(x["branch"] == "x_static" for x in schema) == 196

    x = np.array([[-51.0, -50.0, np.nan, 0.0], [51.0, 50.0, np.inf, -1.0]])
    agg = Aggregate(4); agg.update(x)
    assert agg.lt_neg50.tolist() == [1, 0, 0, 0]
    assert agg.gt_pos50.tolist() == [1, 0, 0, 0]
    assert agg.eq_neg50.tolist() == [0, 1, 0, 0]
    assert agg.eq_pos50.tolist() == [0, 1, 0, 0]
    assert agg.nan.tolist() == [0, 0, 1, 0]
    assert agg.posinf.tolist() == [0, 0, 1, 0]
    assert agg.finite.tolist() == [2, 2, 0, 2]

    sampled1 = select_rows(np.arange(100)[:, None], 7)
    sampled2 = select_rows(np.arange(100)[:, None], 7)
    assert np.array_equal(sampled1[0], sampled2[0])
    assert len(sampled1[0]) == 7
    q = weighted_quantile(np.array([0.0, 1.0, 2.0]), np.ones(3))
    assert np.all(np.diff(q) >= 0)

    log_names = [x["name"] for x in schema if "logP" in x["name"]]
    assert log_names and all(not x["log1p_preview_eligible"] for x in schema if x["name"] in log_names)
    assert all(x["expected_domain"] == "[0,1]" for x in schema if x["family"] == "stage_event")
    print("all audit tests passed")


if __name__ == "__main__":
    main()
