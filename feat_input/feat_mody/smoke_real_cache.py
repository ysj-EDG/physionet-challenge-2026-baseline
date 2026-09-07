#!/usr/bin/env python
"""Two-record real-cache smoke test; no optimizer step and no saved weights."""

from __future__ import annotations
import json
import tempfile
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from feature_scaling import FeatureScaler
from train_lstm import LSTMModel, PSGDataset, collate_fn
import team_code


def main():
    state = json.loads((ROOT / "feat_input/p0_fix/scaler_state_preview.json").read_text())
    scaler = FeatureScaler.from_state_dict(state)
    records = json.loads((ROOT / "split/train_records.json").read_text())[:2]
    dataset = PSGDataset(records, ROOT / "npz_new/train", None, scaler)
    items = [dataset[i] for i in range(2)]

    sample = team_code._sample(
        records[0], None, [ROOT / "npz_new/train"], scaler
    )
    np.testing.assert_allclose(items[0]["X_seq"].numpy(), sample["X_seq"])
    np.testing.assert_allclose(items[0]["X_ecg"].numpy(), sample["X_ecg"])
    np.testing.assert_allclose(items[0]["x_static"].numpy(), sample["x_static"])

    batch = collate_fn(items)
    X_seq, X_ecg, mask, x_static, ages, y, lengths = batch
    model = LSTMModel().cpu().train()
    logits = model(X_seq, X_ecg, mask, x_static, lengths)
    loss = torch.nn.BCEWithLogitsLoss()(logits, y.squeeze(-1))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits).all()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters() if parameter.grad is not None
    )

    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        torch.save({
            "state_dict": team_code.LSTMModel().state_dict(),
            "feature_dims": {"X_seq": 483, "X_ecg": 12, "x_static": 196},
            "input_preprocessing": state,
        }, folder / "lstm_model.pt")
        loaded = team_code.load_model(folder, False)
        assert loaded["preprocessor"].mode == "typed_v1"
        assert len(loaded["preprocessor"].rules) == 691

    print(
        "real_cache_smoke PASS",
        f"records={len(items)}",
        f"X_seq={tuple(X_seq.shape)}",
        f"X_ecg={tuple(X_ecg.shape)}",
        f"x_static={tuple(x_static.shape)}",
        f"raw_ages={ages.tolist()}",
        f"loss_finite={bool(torch.isfinite(loss))}",
        "typed_checkpoint_rules=691",
    )


if __name__ == "__main__":
    main()
