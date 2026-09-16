# DATA2-L0 reuse audit

- Branch: `official-submission`.
- Starting HEAD: `ee925dd317af950f35cb632da57a86400dc999be`.
- Reuses `train_lstm.LSTMModel`, `train_epoch`, `PSGDataset`, `collate_fn`, and `legacy_clip`.
- Reuses P5 `FrozenTruthDataset`, global class-balanced loader, natural loader, raw-logit collection, and official metric bundle.
- Reuses the frozen data2 low-cost manifest; no EDF, label reconstruction, feature extraction, or NPZ write occurred.
- Thin runner adds only per-seen-site stratified inner validation, macro seen-site AC selection, checkpointing, and report aggregation.
- Scheduler is historical `ReduceLROnPlateau(mode=max,factor=0.5,patience=8)` monitoring seen-site validation macro AC.
- Environment: Python `3.10.20`, PyTorch `2.5.1+cu121`, CUDA `12.1`, device `Quadro P6000`.
- Independent one-batch forward/backward smoke loss `113.552513` was finite.
