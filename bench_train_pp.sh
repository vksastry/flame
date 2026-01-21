#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bench_train_pp.sh

Notes:
  - Runs single-GPU random-token benchmark via flame.train_benchmark
  - Enables pipeline parallelism for input_len >= 10000

Optional env vars:
  BASE_DUMP_DIR (default: ./train_out)
  BATCH_SIZE (default: 8)
  TRAIN_STEPS (default: 50)
  NUM_WORKERS (default: 0)
  VOCAB_SIZE (default: model config vocab_size)
  PP_DEGREE (default: 4)
  PP_SPLIT_POINTS (default: empty)
  PP_MICROBATCHES (default: empty)
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

BASE_DUMP_DIR=${BASE_DUMP_DIR:-./train_out}
BATCH_SIZE=${BATCH_SIZE:-8}
TRAIN_STEPS=${TRAIN_STEPS:-50}
NUM_WORKERS=${NUM_WORKERS:-0}
VOCAB_SIZE=${VOCAB_SIZE:-}
PP_DEGREE=${PP_DEGREE:-4}
PP_SPLIT_POINTS=${PP_SPLIT_POINTS:-}
PP_MICROBATCHES=${PP_MICROBATCHES:-}

CONFIGS=(
  "configs/gla_340M.json"
  "configs/gated_deltanet_340M.json"
  "configs/kda_340M.json"
  "configs/mamba2_340M.json"
  "configs/delta_net_340M.json"
  "configs/transformer_340M.json"
)

INPUT_LENS=(10 100 1000 10000 100000 1000000)

python -m flame.train_benchmark \
  --model.config "configs/transformer_340M.json" \
  --job.dump_folder "${BASE_DUMP_DIR}/transformer_340M/len_100000_cp_tp" \
  --training.batch_size 1 \
  --training.num_workers "${NUM_WORKERS}" \
  --training.seq_len 100000 \
  --training.context_len 100000 \
  --training.steps 5 \
  --experimental.context_parallel_degree 4 \
  --training.tensor_parallel_degree 1 \
  --experimental.pipeline_parallel_degree 1

for MODEL_CONFIG in "${CONFIGS[@]}"; do
  MODEL_NAME=$(basename "${MODEL_CONFIG}" .json)
  for INPUT_LEN in "${INPUT_LENS[@]}"; do
    RUN_DIR="${BASE_DUMP_DIR}/${MODEL_NAME}/len_${INPUT_LEN}"

    EXTRA_PP_ARGS=()
    if [[ ${INPUT_LEN} -ge 10000 ]]; then
      EXTRA_PP_ARGS+=("--experimental.pipeline_parallel_degree" "${PP_DEGREE}")
      if [[ -n ${PP_SPLIT_POINTS} ]]; then
        EXTRA_PP_ARGS+=("--experimental.pipeline_parallel_split_points" "${PP_SPLIT_POINTS}")
      fi
      if [[ -n ${PP_MICROBATCHES} ]]; then
        EXTRA_PP_ARGS+=("--experimental.pipeline_parallel_microbatches" "${PP_MICROBATCHES}")
      fi
    fi

    BENCH_ARGS=()
    if [[ -n ${VOCAB_SIZE} ]]; then
      BENCH_ARGS+=("--bench.vocab_size" "${VOCAB_SIZE}")
    fi

    python -m flame.train_benchmark \
      --model.config "${MODEL_CONFIG}" \
      --job.dump_folder "${RUN_DIR}" \
      --training.batch_size "${BATCH_SIZE}" \
      --training.num_workers "${NUM_WORKERS}" \
      --training.seq_len "${INPUT_LEN}" \
      --training.context_len "${INPUT_LEN}" \
      --training.steps "${TRAIN_STEPS}" \
      "${BENCH_ARGS[@]}" \
      "${EXTRA_PP_ARGS[@]}"
  done
done
