# Physio Fusion V1 Experiment Plan

> **For agentic workers:** Implement this plan in phases. The current execution phase is **Gate 1 + Gate 2 only**. Do not start CI fusion training until `D_SELECTION.md` has been reviewed.

**Goal:** Build a physiology-only epoch representation pipeline for DATA2 that compresses only the high-dimensional EEG coherence block, preserves the other engineered physiological features, and later tests whether a cross-modal fusion MLP improves strict 3-site LOSO CI performance.

**Architecture:** The frozen `timegrid_v2.1.0 + validity_v0.2` NPZ cache is the only feature source. EEG spectral54, EEG BSR18, HRV11, EMG24, and Resp14 remain in their engineered dimensions. EEG coherence360 is reshaped to 15 pairs × 24 descriptors and compressed by one shared pair encoder `24 -> 16 -> d`, with `d` selected inside each outer LOSO fold from `{4, 8, 12, 16}` using CI-blind reconstruction. Later, the selected coherence latents are concatenated with the uncompressed physiological features and mapped by a cross-modal MLP to a 192-D epoch embedding.

**Tech Stack:** Python, NumPy, PyTorch, scikit-learn, existing `evaluate_model.compute_auroc_age`.

**Input contract:** `validity_v0_2_full_6600/FULL_VALIDITY_QC.md` (`VALIDITY_V0_2_FULL_QC = PASS`).

**Repository:** `/database/home/gaohaojie/workspace/python-example-2026/`

**Branch:** `official-submission`

**Expected starting HEAD:** `6c023bca85054f5477e976b53dc98aab483d9ef4`

**Frozen NPZ root:** `/database/home/gaohaojie/workspace/python-example-2026/validity_v0_2_full_6600/workspace/result_bundle/npz`

## Global Constraints

- The input cache must contain exactly 6,600 NPZs: I0002=319, I0006=1,142, S0001=5,139.
- Every NPZ must report `extraction_version=timegrid_v2.1.0` and `validity_schema_version=validity_v0.2`.
- Never modify the extractor or any NPZ.
- Never use held-out outer-site data to fit normalization, train the coherence autoencoder, select `d`, train the CI model, select checkpoints, or tune thresholds.
- Stage/Event13 (`X_seq[:,470:483]`) are excluded from Physio Fusion V1.
- Circadian (`X_ecg[:,11]`) is excluded.
- CAISR20 is excluded until the later residual experiment.
- Demographics, site ID, availability flags, and raw validity flags are not concatenated to the physiological model input.
- Natural-invalid values never participate in normalization fitting or reconstruction loss.
- A legitimate physical zero remains valid.
- After valid-only normalization, invalid inputs are replaced by neutral normalized zero.
- `d` selection is CI-blind.
- CI training later uses `site+class balanced` record sampling and `pos_weight=1`.
- Do not revive the legacy LSTM path.
- Keep new code isolated in one directory; do not create a large framework.

## Review Focus

1. **Coherence indexing:** `X_seq[:,54:414]` must reshape exactly to `[T,15,24]`; pair order must remain unchanged.
2. **Validity semantics:** coherence pair validity must use the frozen rule: pair endpoints available and `eeg_common_clean_subsegment_count[t] > 0`.
3. **HRV alignment:** ECG window row `j` maps to sequence epoch `e=j+10`; only HRV columns `0:11` are used.
4. **Leakage:** all fold-local scalers and `d` selection must be fitted without outer-test records.
5. **Site imbalance:** reconstruction training/selection must not be dominated by S0001 record count.

---

# 1. Scientific question

The main V1 question is:

> Can a physiology-only representation preserve interpretable engineered PSG information while reducing the disproportionate dimensionality of EEG coherence, and can a simple cross-modal MLP then learn useful CI-relevant interactions across EEG, ECG, EMG, and respiration?

The design intentionally does **not** reproduce ID79 sleep-stage/event supervision. Our input is already composed of interpretable engineered physiological quantities. Stage/Event and CAISR20 are reserved for later controlled experiments.

---

# 2. Frozen feature layout

Use only the following V1 features.

| Family | Source | Dimensions | V1 handling |
|---|---|---:|---|
| EEG spectral | `X_seq[:,0:54]` | 54 | Preserve |
| EEG coherence | `X_seq[:,54:414]` | 360 = 15×24 | Compress only this block |
| EEG BSR | `X_seq[:,414:432]` | 18 | Preserve |
| EMG | `X_seq[:,432:456]` | 24 | Preserve |
| Resp | `X_seq[:,456:470]` | 14 | Preserve |
| Stage/Event | `X_seq[:,470:483]` | 13 | Exclude |
| HRV | aligned `X_ecg[:,0:11]` | 11 | Preserve |
| Circadian | `X_ecg[:,11]` | 1 | Exclude |

For a selected coherence latent width `d`, the epoch physiological input dimension before fusion is:

`D_in = 54 + 18 + 11 + 24 + 14 + 15*d = 121 + 15*d`

Thus:

| d | Coherence latent | `D_in` |
|---:|---:|---:|
| 4 | 60 | 181 |
| 8 | 120 | 241 |
| 12 | 180 | 301 |
| 16 | 240 | 361 |

---

# 3. Coherence structure and validity

The 15 fixed EEG pairs are:

1. F3-F4
2. F3-C3
3. F3-C4
4. F3-O1
5. F3-O2
6. F4-C3
7. F4-C4
8. F4-O1
9. F4-O2
10. C3-C4
11. C3-O1
12. C3-O2
13. C4-O1
14. C4-O2
15. O1-O2

Each pair has 24 descriptors in the frozen order:

- 5 band means
- 5 band AUCs
- 5 band IQRs
- 4 ratios
- 3 full-spectrum descriptors
- 2 sigma descriptors

For epoch `t` and pair `(a,b)`:

`pair_valid[t,p] = eeg_channel_available[a] & eeg_channel_available[b] & (eeg_common_clean_subsegment_count[t] > 0)`

A valid coherence zero is still valid and must remain a training target.

---

# 4. Gate 1 — Dataset contract

## Objective

Create a minimal reusable loader that exposes the frozen V1 physiological tensors and exact validity masks without altering the cache.

## New code directory

```text
physio_fusion_v1/
├── dataset.py
├── model.py
├── train.py
└── run.py
```

Do not add more Python modules in V1 unless a concrete blocker is found.

## `dataset.py` responsibilities

Implement:

```python
SEQ_DIM = 483
ECG_DIM = 12

SPECTRAL = slice(0, 54)
COHERENCE = slice(54, 414)
BSR = slice(414, 432)
EMG = slice(432, 456)
RESP = slice(456, 470)
STAGE_EVENT = slice(470, 483)
```

Provide one record loader that returns, at minimum:

```python
{
    "record_id": str,
    "site": str,
    "y": int,
    "age": float,
    "spectral": float32[T,54],
    "spectral_valid": bool[T,54],
    "coherence": float32[T,15,24],
    "coherence_pair_valid": bool[T,15],
    "bsr": float32[T,18],
    "bsr_valid": bool[T,18],
    "emg": float32[T,24],
    "emg_valid": bool[T,24],
    "resp": float32[T,14],
    "resp_valid": bool[T,14],
    "hrv": float32[T,11],
    "hrv_valid": bool[T,11],
}
```

Derive validity from frozen metadata; do not infer validity from feature value.

Required rules:

- Spectral channel `c`: valid iff EEG channel available and `eeg_clean_subsegment_count[:,c] > 0`; repeat to its 9 spectral features.
- BSR channel `c`: valid iff EEG channel available; map threshold-major BSR `[c,6+c,12+c]`.
- Coherence pair: exact rule in Section 3.
- EMG: per-channel availability & preprocessing success & epoch success, repeated across the 8 features of that channel.
- Resp: use `resp_feature_valid[T,14]` directly.
- HRV: align row `j` to epoch `j+10`, use columns `0:11`, and combine `ecg_alignment_valid`, `hrv_success`, and `hrv_feature_valid`.
- Do not expose Stage/Event13 or circadian to V1 model assembly.

Add deterministic utility functions for:
- record/site discovery from the 6,600 NPZ filenames/metadata;
- outer LOSO split;
- per-site deterministic inner 80/20 record split with seed 7;
- stable hash-based sampling.

## Gate 1 tests

`run.py smoke` must:

1. Confirm exactly 6,600 NPZ files and expected site counts.
2. Confirm versions on all files.
3. Load at least one normal record and the known edge cases:
   - `sub-I0002150027361_ses-4`
   - `sub-S0001111343357_ses-1`
4. Assert all family shapes.
5. Assert coherence reshape is `[T,15,24]`.
6. Assert `X_seq[:,470:483]` is never returned as model input.
7. Assert `X_ecg[:,11]` is never returned as model input.
8. Assert the short-ECG record yields HRV-invalid epochs rather than fake zeros marked valid.
9. Assert the ECG-absent record yields HRV-invalid epochs.
10. Assert valid physical zeros remain valid where present.
11. Assert no outer-test record appears in an inner-train or inner-val split.

Gate 1 passes only when all assertions pass.

---

# 5. Gate 2 — Select coherence latent width `d`

## Scientific objective

Select the smallest shared pair bottleneck that preserves coherence information nearly as well as the best tested bottleneck, without using CI labels.

Candidates:

```text
d ∈ {4, 8, 12, 16}
```

## Model

In `model.py` implement only the required shared autoencoder:

```text
encoder: 24 -> 16 -> d
decoder: d -> 16 -> 24
activation: ReLU between linear layers
```

The same encoder and decoder weights are used for all 15 EEG pairs.

No pair-ID embedding.
No Stage/Event supervision.
No CI supervision.
No artificial masking.
No EMA teacher.
No attention.
No graph network.

## Fold-local normalization

For each outer held-out site:

1. Use only the other two sites.
2. Split each training site at record level into deterministic 80% inner-train / 20% inner-val using seed 7.
3. Fit one 24-dimensional coherence scaler on **inner-train valid pairs only**.
4. The 24 descriptor positions share their scaler across all 15 pair identities.
5. Use robust center/scale: median and IQR per descriptor.
6. If IQR is zero, use scale 1.
7. Never include invalid pairs in scaler fitting.

## Balanced pair sampling

Do not flatten every valid pair from all records and random-shuffle globally.

For each training site:
- allocate the same target sample budget per site;
- sample records approximately uniformly within site;
- within a selected record, sample only valid `(epoch,pair)` vectors;
- use a stable hash/seed so reruns are reproducible.

Use a target of **200,000 valid pair vectors per training site per outer fold** when available.

For validation:
- do not use training samples;
- use the deterministic inner-val records;
- evaluate at most **128 valid pair vectors per record**, selected stably;
- first compute reconstruction metrics per record;
- then average records within site;
- primary validation metric is the equal-weight mean of the two site means.

This prevents S0001 from dominating by record count or night length.

## Training configuration

For each `d` and each seed in:

```text
[7, 17, 27]
```

train with:

```text
optimizer = Adam
lr = 1e-3
weight_decay = 1e-5
batch_size = 4096
max_epochs = 50
early_stopping_patience = 5
loss = mean squared error on normalized valid 24-D vectors
```

Checkpoint on the lowest site-balanced validation MSE.

## Metrics

For each `(outer_fold, d, seed)` report:

- best epoch
- site-balanced validation MSE
- each inner-val site's record-balanced MSE
- overall 24-D reconstruction R²
- each of 24 descriptor R² values
- each of 24 descriptor Pearson correlations
- family metrics for:
  - band means `[0:5]`
  - AUC `[5:10]`
  - IQR `[10:15]`
  - ratios `[15:19]`
  - full-spectrum `[19:22]`
  - sigma `[22:24]`

## Selection rule

For each outer fold:

1. For each `d`, compute the mean site-balanced validation MSE over seeds 7/17/27.
2. Compute its standard error across the three seeds:
   `SE = sample_std / sqrt(3)`.
3. Find the `d_best` with the lowest mean MSE.
4. Set:
   `threshold = mean_MSE(d_best) + SE(d_best)`.
5. Select the **smallest** `d` whose mean MSE is `<= threshold`.
6. Family and per-feature metrics are diagnostic; they do not silently alter the rule.
7. If any candidate produces non-finite loss/metrics, mark that candidate failed and report it; do not substitute another rule.

Expected outputs:

```text
output/physio_fusion_v1/d_selection/
├── outer_I0002/
├── outer_I0006/
├── outer_S0001/
├── d_selection.csv
├── d_selection.json
└── D_SELECTION.md
```

`D_SELECTION.md` must state:

```text
held I0002 -> selected d = ?
held I0006 -> selected d = ?
held S0001 -> selected d = ?
```

and show the candidate table for all four `d` values.

**STOP after Gate 2 for human review.**

Do not start NoFusion or Fusion192 training in the current execution phase.

---

# 6. Gate 3 — NoFusion control (deferred until Gate 2 review)

After `d` is accepted, freeze the selected fold-specific coherence encoder and discard the decoder.

For each epoch:

```text
spectral54
+ BSR18
+ coherence(15*d)
+ HRV11
+ EMG24
+ Resp14
```

Only fold-local outer-training data may fit the downstream feature normalization.

For every invalid normalized input position, use neutral zero after normalization.

No raw validity mask is concatenated.

NoFusion control:

```text
D_in physiological epoch vector
-> whole-night masked mean
+ whole-night masked std
-> 2*D_in patient vector
-> low-capacity linear/logistic CI classifier
```

Evaluate strict 3-site LOSO.

This gate asks:

> How much CI information is already present after coherence compression without learned cross-modal fusion?

---

# 7. Gate 4 — Fusion192 main model (deferred until Gate 3)

Use the same fold-local selected `d` and same physiological inputs.

```text
D_in
-> Linear(D_in,256)
-> ReLU
-> Linear(256,192)
-> 192-D epoch embedding
-> whole-night mean + std
-> 384-D patient vector
-> Linear(384,1)
```

No LSTM.
No Transformer.
No attention.
No Stage/Event supervision.
No CAISR20.

Train patient-level CI model with:
- site + class balanced record sampling;
- `pos_weight=1`;
- strict outer LOSO;
- checkpoint selection on inner validation Age-conditioned AUROC.

Primary comparison:

```text
NoFusion vs Fusion192
```

Primary reporting:
- I0002 AC
- I0006 AC
- S0001 AC
- Macro AC
- Worst-site AC

Secondary:
- AUROC
- AUPRC

The purpose is to isolate whether the cross-modal MLP adds value beyond the compact physiological vector itself.

---

# 8. Later experiments, explicitly outside V1 Gate 1–4

Only after the physiology-only representation is characterized:

1. Add CAISR20 as a low-capacity residual on top of the physiology CI score.
2. Run a controlled Stage/Event13 direct-input ablation if needed.
3. Test temporal information with the predefined `Orderless vs Shuffled vs Real` gate.
4. Consider artificial masking / EMA consistency only if cross-site diagnostics show a remaining domain shortcut.

These are separate experiments, not hidden options inside the V1 implementation.

---

# 9. Code organization

Create only:

```text
physio_fusion_v1/
├── dataset.py
├── model.py
├── train.py
└── run.py
```

Responsibilities:

## `dataset.py`
- NPZ loading and schema checks
- fixed feature slicing
- exact validity derivation
- HRV-to-epoch alignment
- deterministic outer/inner split utilities
- valid-only robust scaler utilities
- deterministic balanced coherence pair sampling

## `model.py`
- `SharedPairAutoencoder`
- later: `PhysioFusion`
- later: `NightClassifier`

Do not put IO or split logic here.

## `train.py`
- `train_coherence_ae`
- `evaluate_coherence_ae`
- `select_d`
- later: CI train/evaluate functions

Do not put outer-fold orchestration here.

## `run.py`
Single experiment CLI:
- `smoke`
- `d-search`
- later: `nofusion`
- later: `fusion`

It owns the 3-site LOSO loop and writes CSV/JSON/Markdown summaries.

---

# 10. Reuse policy

Reuse:
- `evaluate_model.compute_auroc_age` later for CI metrics.
- scikit-learn metric implementations where appropriate.
- frozen `validity_v0.2` metadata and NPZ cache.
- prior site/class balancing lesson: `site+class balanced`, `pos_weight=1`.

Reference but do not import as a new framework:
- `feature_scaling.py` for feature semantics.
- `feat_input2/run_data2_lowcost.py` for prior LOSO/result-writing patterns.

Do not reuse:
- `train_lstm.py` training architecture.
- legacy clip/mask behavior.
- Stage/Event branches.
- CAISR20 in the V1 main representation.

Do not create model-specific duplicate NPZ caches.

---

# 11. Output and Git policy

Runtime artifacts belong under:

`output/physio_fusion_v1/`

Do not commit:
- checkpoints
- sampled pair arrays
- NPZ
- logs with large binary content
- derived tensors

Commit only when the phase is complete:
- `physio_fusion_v1/*.py`
- this experiment plan
- `D_SELECTION.md`
- compact CSV/JSON summaries if small and useful

Before any commit:
- `git diff --check`
- `python -m py_compile physio_fusion_v1/*.py`
- `python physio_fusion_v1/run.py smoke`
- confirm no extractor/NPZ changes

Suggested phase-1 commit message:

`experiment: add physiology fusion coherence bottleneck study`

---

# 12. Current execution boundary

The next Codex execution should implement and run only:

- **Gate 1 — Dataset contract**
- **Gate 2 — Coherence `d` selection**

Then STOP and return:
- exact modified/created files
- smoke result
- per-fold d-selection table
- `D_SELECTION.md`
- selected `d` for each outer fold
- runtime/environment summary
- git status

Do not start Gate 3 or Gate 4 until the Gate 2 result has been reviewed.
