#!/usr/bin/env bash
set -euo pipefail

ROOT="/database/home/gaohaojie/workspace/python-example-2026"
PREFLIGHT_DIR="$ROOT/feat_input/p0_ab"
PREFLIGHT_PY="$ROOT/feat_input/feat_mody/p0_ab_preflight.py"
PYTHON_BIN="/database/home/gaohaojie/.conda/envs/sleepfm_env/bin/python"
CACHE_DIR="$ROOT/npz_new"
SPLITS_DIR="$ROOT/split"
RULES_FILE="$ROOT/feat_input/feature_rules_v1.json"
OUTPUT_A="$ROOT/output/p0_ab/seed7/A_legacy_clip"
OUTPUT_B="$ROOT/output/p0_ab/seed7/B_typed_v1"

export CUDA_VISIBLE_DEVICES=0
export LSTM_CACHE_DIR="$CACHE_DIR"
export LSTM_SPLITS_DIR="$SPLITS_DIR"
export LSTM_FEATURE_RULES="$RULES_FILE"
export LSTM_SEED=7
export PYTHONHASHSEED=7
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "BLOCKED: frozen sleepfm_env interpreter is unavailable: $PYTHON_BIN" >&2
    exit 2
fi

"$PYTHON_BIN" "$PREFLIGHT_PY" --verify --output-dir "$PREFLIGHT_DIR"

for result_dir in "$OUTPUT_A" "$OUTPUT_B"; do
    if [[ -e "$result_dir" ]]; then
        echo "BLOCKED: refusing to overwrite existing result path: $result_dir" >&2
        exit 2
    fi
done

run_arm() {
    local mode="$1"
    local model_dir="$2"
    mkdir -p "$model_dir"
    export LSTM_INPUT_PREPROCESSING="$mode"
    export LSTM_MODEL_DIR="$model_dir"
    "$PYTHON_BIN" "$PREFLIGHT_PY" \
        --launch-metadata --output-dir "$PREFLIGHT_DIR" \
        --mode "$mode" --model-dir "$model_dir"
    "$PYTHON_BIN" "$ROOT/train_lstm.py" 2>&1 | tee "$model_dir/train.log"
}

# Fresh independent processes, run sequentially on the same physical GPU.
# No resume/checkpoint input is accepted by train_lstm.py.
run_arm legacy_clip "$OUTPUT_A"
run_arm typed_v1 "$OUTPUT_B"
