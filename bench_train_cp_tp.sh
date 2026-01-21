#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bench_train_cp_tp.sh

Notes:
  - Runs a short transformer-only CP test with mpiexec
  - Intended for seq_len=100000 feasibility check

Optional env vars:
  DUMP_DIR (default: ./train_out/transformer_340M/len_100000_cp_tp)
  STEPS (default: 5)
  BATCH_SIZE (default: 1)
  NUM_WORKERS (default: 0)
  CP_DEGREE (default: 4)
  TP_DEGREE (default: 1)
  PP_DEGREE (default: 1)
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

DUMP_DIR=${DUMP_DIR:-./train_out/transformer_340M/len_100000_cp_tp}
STEPS=${STEPS:-5}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-0}
CP_DEGREE=${CP_DEGREE:-4}
TP_DEGREE=${TP_DEGREE:-1}
PP_DEGREE=${PP_DEGREE:-1}

mpiexec -np 4 python -m flame.train_benchmark \
  --model.config "configs/transformer_340M.json" \
  --job.dump_folder "${DUMP_DIR}" \
  --training.batch_size "${BATCH_SIZE}" \
  --training.num_workers "${NUM_WORKERS}" \
  --training.seq_len 100000 \
  --training.context_len 100000 \
  --training.steps "${STEPS}" \
  --experimental.context_parallel_degree "${CP_DEGREE}" \
  --training.tensor_parallel_degree "${TP_DEGREE}" \
  --experimental.pipeline_parallel_degree "${PP_DEGREE}"
