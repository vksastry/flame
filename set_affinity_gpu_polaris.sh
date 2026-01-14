#!/bin/bash
# ============================================================================
# GPU Affinity Script for Polaris
# ============================================================================
# This script assigns GPUs to MPI ranks in a round-robin fashion.
# Each rank gets exclusive access to one GPU based on PMI_LOCAL_RANK.
#
# Usage:
#   mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} ./set_affinity_gpu_polaris.sh ./app
# ============================================================================

# Get the number of GPUs available on the node
num_gpus=$(nvidia-smi -L | wc -l)

# Assign GPU based on local rank (round-robin)
gpu=$((${PMI_LOCAL_RANK} % ${num_gpus}))

# Set CUDA_VISIBLE_DEVICES to restrict this rank to its assigned GPU
export CUDA_VISIBLE_DEVICES=$gpu

# Also set LOCAL_RANK for PyTorch distributed training
export LOCAL_RANK=${PMI_LOCAL_RANK}

# Set RANK and WORLD_SIZE from PMI environment variables if not already set
if [ -z "${RANK}" ]; then
    export RANK=${PMI_RANK}
fi

# Log the assignment for debugging
echo "RANK=${PMI_RANK} LOCAL_RANK=${PMI_LOCAL_RANK} CUDA_VISIBLE_DEVICES=${gpu} on $(hostname)"

# Execute the command passed as arguments
exec "$@"
