# T1 I0002 matched comparison

This analysis uses the same 317 temporally usable I0002 records for all
comparisons (51 positive, 266 negative). The O, S, and R prediction files have
identical ordered record IDs, labels, ages, and patient IDs. The S and R arms
also have identical inner splits and initial model-weight hashes.

## Matched results

| Model | AC-AUROC | AUROC | AUPRC |
|---|---:|---:|---:|
| CAISR20, matched cohort | 0.758564 | 0.778269 | 0.519791 |
| T1-O, orderless LR | 0.721044 | 0.733304 | 0.361576 |
| T1-S, shuffled local TCN | 0.747145 | 0.756303 | 0.419648 |
| T1-R, real-order local TCN | 0.731376 | 0.743771 | 0.410866 |

## Paired comparisons

The bootstrap used 1,000 patient-level replicates, stratified by class, with
the same resampled patients used for both arms in each comparison. The random
seed was 20260918. The vectorized bootstrap AC implementation was checked
against `evaluate_model.compute_auroc_age(..., gap=2)` to absolute tolerance
1e-12.

| Comparison | Point delta AC | Bootstrap mean | 95% percentile CI |
|---|---:|---:|---:|
| T1-S minus T1-O | +0.026101 | +0.025170 | [-0.018063, +0.071402] |
| T1-R minus T1-S | -0.015769 | -0.015585 | [-0.039096, +0.004971] |
| T1-R minus T1-O | +0.010332 | — | — |

## T1-R training

- Best epoch: 14
- Best inner seen-site macro AC: 0.6825667090
- Training wall time: 12,567.73 seconds (3 h 29 min 27.73 s)
- Initial-weight hash matched T1-S:
  `4f3c0df9a81f1e0f75cd8aeff87bef83afec66509d5f6b1e52ce3702a4e3cec1`
- NaN, Inf, CUDA OOM, exception, or abnormal termination: none observed

## Interpretation

The shuffled local TCN has a positive point difference over the orderless LR,
but its paired confidence interval includes zero. This is suggestive local
representation gain on I0002, not statistically resolved evidence.

Real temporal order does not improve over the matched shuffled control: the
point estimate is lower and the paired confidence interval includes zero.
Therefore this experiment provides no evidence of a real local temporal-order
gain. No additional site, modality, fusion, architecture, or timegrid
experiment was started.
