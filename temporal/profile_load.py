#!/usr/bin/env python
"""Measure direct DATA2 temporal loading on the actual Medex worker."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from temporal.data import load_dataset
from temporal.run_loso import json_dump, runtime_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    records, _, load_seconds = load_dataset(args.data_root, 'stage_event13', False)
    payload = {
        'records': len(records),
        'unusable_no_complete_block': sum(record.n_blocks < 1 for record in records),
        'load_dataset_wall_time_sec': load_seconds,
        'process_total_wall_time_sec': time.perf_counter() - started,
        'runtime': runtime_snapshot(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    json_dump(output, payload)
    print(payload, flush=True)


if __name__ == '__main__':
    main()
