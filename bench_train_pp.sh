#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bench_train_pp.sh

Notes:
  - Runs single-GPU training via flame.train across configs and input lengths
  - Enables pipeline parallelism for input_len >= 10000

Optional env vars:
  BASE_DUMP_DIR (default: ./train_out)
  BATCH_SIZE (default: 8)
  TRAIN_STEPS (default: 50)
  NUM_WORKERS (default: 0)
  DATASET (default: HuggingFaceFW/fineweb-edu)
  DATASET_NAME (default: empty)
  DATASET_SPLIT (default: empty)
  DATA_FILES (default: empty)
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
DATASET=${DATASET:-HuggingFaceFW/fineweb-edu}
DATASET_NAME=${DATASET_NAME:-}
DATASET_SPLIT=${DATASET_SPLIT:-}
DATA_FILES=${DATA_FILES:-}
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

    DATASET_ARGS=("--training.dataset" "${DATASET}")
    if [[ -n ${DATASET_NAME} ]]; then
      DATASET_ARGS+=("--training.dataset_name" "${DATASET_NAME}")
    fi
    if [[ -n ${DATASET_SPLIT} ]]; then
      DATASET_ARGS+=("--training.dataset_split" "${DATASET_SPLIT}")
    fi
    if [[ -n ${DATA_FILES} ]]; then
      DATASET_ARGS+=("--training.data_files" "${DATA_FILES}")
    fi

    python -m flame.train \
      --model.config "${MODEL_CONFIG}" \
      --job.dump_folder "${RUN_DIR}" \
      --training.batch_size "${BATCH_SIZE}" \
      --training.num_workers "${NUM_WORKERS}" \
      --training.seq_len "${INPUT_LEN}" \
      --training.context_len "${INPUT_LEN}" \
      --training.steps "${TRAIN_STEPS}" \
      "${DATASET_ARGS[@]}" \
      "${EXTRA_PP_ARGS[@]}"
  done
done
