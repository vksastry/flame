#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bench_hgt.sh

Notes:
  - Runs single-GPU benchmarks across model configs and input lengths
  - Uses default benchmark settings unless overridden via env vars

Optional env vars:
  BASE_DUMP_DIR (default: ./bench_out)
  BATCH_SIZE (default: 8)
  WARMUP_STEPS (default: 10)
  BENCH_STEPS (default: 50)
  CHANNELS (default: 1)
  HEIGHT (default: 73)
  WIDTH (default: 144)
  NUM_WORKERS (default: 0)
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

BASE_DUMP_DIR=${BASE_DUMP_DIR:-./bench_out}
BATCH_SIZE=${BATCH_SIZE:-1}
WARMUP_STEPS=${WARMUP_STEPS:-10}
BENCH_STEPS=${BENCH_STEPS:-50}
CHANNELS=${CHANNELS:-1}
HEIGHT=${HEIGHT:-64}
WIDTH=${WIDTH:-64}
NUM_WORKERS=${NUM_WORKERS:-0}
CSV_PATH=${CSV_PATH:-${BASE_DUMP_DIR}/hgt_benchmark_summary.csv}

CONFIGS=(
  "configs/gla_340M.json"
  "configs/gated_deltanet_340M.json"
  "configs/kda_340M.json"
  "configs/mamba2_340M.json"
  "configs/delta_net_340M.json"
  "configs/transformer_340M.json"
)

INPUT_LENS=(10 100 1000 10000) # 100000 1000000)

for MODEL_CONFIG in "${CONFIGS[@]}"; do
  MODEL_NAME=$(basename "${MODEL_CONFIG}" .json)
  for INPUT_LEN in "${INPUT_LENS[@]}"; do
    RUN_DIR="${BASE_DUMP_DIR}/${MODEL_NAME}/len_${INPUT_LEN}"
    python -m flame.train_hgt_benchmark \
      --model.config "${MODEL_CONFIG}" \
      --job.dump_folder "${RUN_DIR}" \
      --training.batch_size "${BATCH_SIZE}" \
      --training.num_workers "${NUM_WORKERS}" \
      --bench.input_len "${INPUT_LEN}" \
      --bench.channels "${CHANNELS}" \
      --bench.height "${HEIGHT}" \
      --bench.width "${WIDTH}" \
      --bench.warmup_steps "${WARMUP_STEPS}" \
      --bench.bench_steps "${BENCH_STEPS}"

    SUMMARY_PATH="${RUN_DIR}/hgt_benchmark_summary.json"
    python - "${SUMMARY_PATH}" "${CSV_PATH}" "${MODEL_NAME}" "${INPUT_LEN}" <<'PY'
import csv
import json
import os
import sys

summary_path = sys.argv[1]
csv_path = sys.argv[2]
model_name = sys.argv[3]
input_len = sys.argv[4]

with open(summary_path, "r", encoding="utf-8") as f:
    data = json.load(f)

row = {
    "model": model_name,
    "input_len": int(input_len),
    "steps": data.get("steps"),
    "elapsed_seconds": data.get("elapsed_seconds"),
    "steps_per_sec": data.get("steps_per_sec"),
    "tokens_per_sec": data.get("tokens_per_sec"),
    "tokens_per_step": data.get("tokens_per_step"),
    "global_batch_size": data.get("global_batch_size"),
    "channels": data.get("channels"),
    "height": data.get("height"),
    "width": data.get("width"),
    "model_params": data.get("model_params"),
}

fieldnames = list(row.keys())
write_header = not os.path.exists(csv_path)

with open(csv_path, "a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    writer.writerow(row)
PY
  done
done
