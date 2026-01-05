#!/bin/bash
#steps_per_epoch = ceil(112491 / 128)
NHOSTS=$(wc -l < "${PBS_NODEFILE}")
NGPU_PER_HOST=$(nvidia-smi -L | wc -l)
NGPUS="$((${NHOSTS}*${NGPU_PER_HOST}))"
BS=32
SAMPLES=112491
GLOBAL_BATCH_SIZE="$((${NGPUS}*${BS}))"
echo "GLobal BS : $GLOBAL_BATCH_SIZE"
STEPS_PER_EPOCH=$(( (SAMPLES + GLOBAL_BATCH_SIZE - 1) / GLOBAL_BATCH_SIZE ))
echo "Steps per epoch: $STEPS_PER_EPOCH"
NEPOCHS=1
T_STEPS="$((${STEPS_PER_EPOCH}*${NEPOCHS}))"
echo "Steps number: $T_STEPS"

# Calculate the ceiling using pure bash integer arithmetic
# STEPS_PER_EPOCH=$(( (SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE ))
#
# echo "Steps per epoch: $STEPS_PER_EPOCH"
#

NNODE=$NHOSTS NGPU=$NGPU_PER_HOST LOG_RANK=0 bash train.sh \
	  --job.config_file flame/models/fla.toml \
	  --job.dump_folder exp/kda-340M-hgt/ \
	  --model.config configs/kda_340M.json \
	  --model.tokenizer_path fla-hub/delta_net-1.3B-100B \
	  --optimizer.name AdamW \
	  --optimizer.eps 1e-15 \
	  --optimizer.lr 3e-4 \
	  --lr_scheduler.warmup_steps 1024 \
	  --lr_scheduler.lr_min 0.1 \
	  --lr_scheduler.decay_type cosine \
	  --training.batch_size $BS \
	  --training.seq_len 2048 \
	  --training.gradient_accumulation_steps 1 \
	  --training.steps $T_STEPS \
	  --training.max_norm 1.0 \
	  --training.skip_nan_inf \
	  --training.data_files /eagle/datascience/vsastry/projects/LatentTwinShared/hgt_all_new.npy \
	  --training.input_len 8 \
          --training.target_len 2 \
          --training.stride 1 \
          --training.levels 7 \
	  --training.dataset_split train \
	  --training.num_workers 0 \
	  --training.prefetch_factor 1 \
	  --training.seed 42 \
	  --training.compile \
	  --training.tensor_parallel_degree 1 \
	  --training.disable_loss_parallel \
	  --checkpoint.interval 500 \
	  --checkpoint.load_step -1 \
	  --metrics.log_freq 1
	  #--training.streaming \
