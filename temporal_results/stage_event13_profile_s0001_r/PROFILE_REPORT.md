# T1 Stage/Event13 H100 runtime profile

## Status

The pre-existing formal Medex run (session `38440`) was not terminated. It progressed from shuffled-arm epoch 1 to epochs 2--6 during this audit, so it did not meet the requested stalled-at-epoch-1 termination condition. No control characters were sent to that session.

The independent benchmark used the full usable I0002 + I0006 training population, the frozen 80/20 inner split, holdout S0001, real-order arm, seed 7, and two epochs. It did not use a smoke subset and did not save or transfer a checkpoint.

## Remote worker

- Device: NVIDIA H100 PCIe (`CUDA_VISIBLE_DEVICES=0`)
- Logical CPUs: 32
- System RAM: 164,830,288 KiB total; 152,903,103--154,609,798 KiB available during the run
- Process RSS: 1,174,720 KiB at fold start; 2,147,980 KiB at fold end
- Process peak CUDA allocation: 286,342,144 bytes (273.1 MiB)
- Whole-device memory before launch: 67,408 / 81,559 MiB (82.6%); the GPU was shared and already heavily occupied

## Timings

- Direct load of 6,600 DATA2 NPZ records: 93.379 s
- Process load-only total: 93.425 s
- Block construction: inner train 0.0529 s; I0002 validation 0.0023 s; I0006 validation 0.0102 s; S0001 holdout 0.3760 s
- Epoch 1: train 149.413 s; validation 19.645 s; AC calculation 0.0060 s; total 169.065 s
- Epoch 2: train 151.313 s; validation 19.343 s; AC calculation 0.0055 s; total 170.662 s
- Mean training batch time: 7.910 s across 19 batches/epoch
- Mean epoch total: 169.864 s
- S0001 holdout inference: 333.547 s
- S0001 final metric bundle: 16.736 s
- Two-epoch fold total after loading: 690.435 s (11.51 min)
- Load plus two-epoch fold: 783.814 s (13.06 min)

## Bottleneck diagnosis

Block construction, DataLoader/collation overhead, AC calculation, and logging are negligible. GPU model execution dominates: training accounts for 43.6% of fold time and S0001 holdout inference for 48.3%. Validation inference adds 5.6%. The model itself used only 273 MiB peak allocation while the shared H100 already had 67.4 GiB allocated before launch, so resource contention and execution/kernel overhead are the leading runtime concerns. This profile does not justify changing the scientific architecture.

## Runtime projection

Using the observed epoch time and scaling by usable record counts, a minimum-eight-epoch TCN arm is approximately 28.5 min for holdout S0001, 97.7 min for holdout I0002, and 85.8 min for holdout I0006. Both S and R arms therefore require about 7.1 hours at minimum, plus the inexpensive O arms and one-time loading. The 30-epoch upper-bound projection is approximately 26 hours. These are shared-GPU estimates, not guarantees.

The 120-minute guard is not satisfied; T2 should not be started under the observed runtime. The next action should be execution-only optimization and an isolated/less-contended H100 re-profile, while preserving the frozen model mathematics and protocol.

## Benchmark completion note

All two-epoch computations, validation, holdout inference, metrics, and profile JSON were completed. The first benchmark process then failed only while writing the final root config because Medex snapshots omit `.git`. Commit `0103eac` added a non-failing provenance fallback; no scientific computation was affected.
