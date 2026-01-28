#!/bin/bash

NHOSTS=$(wc -l < "${PBS_NODEFILE}")
NGPU_PER_HOST=$(nvidia-smi -L | wc -l)
NGPUS="$((${NHOSTS}*${NGPU_PER_HOST}))"
echo $NGPUS
cd /eagle/datascience/vsastry/projects/LinearAttention/new_repo/flame
module use /soft/modulefiles; module load conda; conda activate base  
source /eagle/datascience/vsastry/projects/LinearAttention/venvs/flame_env/bin/activate

# proxy settings
export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"
MODEL='transformer_340M' #'gla_340M'
DUMP_DIR=${DUMP_DIR:-./train_out/${MODEL}/len_100000_cp_tp}
STEPS=${STEPS:-5}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-0}
CP_DEGREE=${CP_DEGREE:-1}
TP_DEGREE=${TP_DEGREE:-16}
PP_DEGREE=${PP_DEGREE:-1}
SEQ_LEN=1000000
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
#NGPUS=8
echo "launching the framework"
mpiexec -np ${NGPUS} -ppn 4 python -m flame.train_benchmark \
  --model.config "configs/${MODEL}.json" \
  --job.dump_folder "${DUMP_DIR}" \
  --training.batch_size "${BATCH_SIZE}" \
  --training.num_workers "${NUM_WORKERS}" \
  --training.seq_len ${SEQ_LEN} \
  --training.context_len ${SEQ_LEN} \
  --training.steps "${STEPS}" \
  --training.mixed_precision_param bfloat16 \
  --metrics.log_freq 1 \
  --experimental.context_parallel_degree "${CP_DEGREE}" \
  --training.tensor_parallel_degree "${TP_DEGREE}" \
  --experimental.pipeline_parallel_degree "${PP_DEGREE}"
