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
BATCH_SIZE=${BATCH_SIZE:-8}
WARMUP_STEPS=${WARMUP_STEPS:-10}
BENCH_STEPS=${BENCH_STEPS:-50}
CHANNELS=${CHANNELS:-1}
HEIGHT=${HEIGHT:-73}
WIDTH=${WIDTH:-144}
NUM_WORKERS=${NUM_WORKERS:-0}

CONFIGS=(
  "configs/gla_340M.json"
  "configs/gated_deltanet_340M.json"
  "configs/kda_340M.json"
  "configs/mamba2_340M.json"
  "configs/delta_net_340M.json"
  "configs/transformer_340M.json"
)

INPUT_LENS=(10 100 1000 10000 100000 1000000)

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
  done
done
