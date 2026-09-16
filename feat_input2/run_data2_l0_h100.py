#!/usr/bin/env python3
"""Sequential Medex/H100 launcher for the three frozen DATA2-L0 folds."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data2_l0_h100_results"
RUNNER = ROOT / "feat_input2/run_data2_l0_lstm.py"


def main() -> None:
    sites = ("I0002", "I0006", "S0001")
    print(f"DATA2-L0 H100 launcher root={ROOT} output={OUTPUT}", flush=True)
    print(f"python={sys.executable}", flush=True)
    required = [OUTPUT / "splits" / f"outer_{site}.csv" for site in sites]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen split manifests: {missing}")
    env = os.environ.copy()
    env.update({
        "DATA2_L0_OUTPUT_DIR": str(OUTPUT),
        "LSTM_SEED": "7", "PYTHONHASHSEED": "7",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })
    for site in sites:
        print(f"Starting DATA2-L0 outer holdout {site}", flush=True)
        fold_dir = OUTPUT / site
        if fold_dir.exists():
            raise FileExistsError(f"Refusing to overwrite: {fold_dir}")
        subprocess.run(
            [sys.executable, "-u", str(RUNNER), "--fold", site],
            cwd=ROOT, env=env, check=True,
        )
        print(f"Completed DATA2-L0 outer holdout {site}", flush=True)
    print("Finalizing DATA2-L0 aggregate results", flush=True)
    subprocess.run([sys.executable, "-u", str(RUNNER), "--finalize"], cwd=ROOT, env=env, check=True)
    print("DATA2-L0 H100 launcher completed", flush=True)


if __name__ == "__main__":
    main()
