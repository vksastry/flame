# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Flame is a minimal, efficient distributed training framework for Large Language Models (LLMs) built on PyTorch's `torchtitan`. It specializes in Flash Linear Attention (FLA) architectures and integrates with Hugging Face transformers.

**Key features:** Online tokenization, dataset shuffling, variable-length sequence packing, multi-dataset support, 4D parallelism.

## Virtual Environment

A virtual environment is set up at `venv/`. To activate:
```bash
source venv/bin/activate
```

## Common Commands

### Installation
```bash
pip install .
pip uninstall flash-linear-attention && pip install -U --no-use-pep517 git+https://github.com/fla-org/flash-linear-attention
pip install git+https://github.com/pytorch/torchtitan.git@0b44d4c
pip install mpi4py tyro  # Required for MPI-based training
```

### Training

The `train.sh` script uses MPI (`mpiexec`) for distributed training:
```bash
# Multi-GPU with MPI (default 4 GPUs)
mpiexec -np 4 python -m flame.train --job.config_file flame/models/fla.toml --model.config configs/gla_340M.json ...

# Or use train.sh wrapper
NGPU=4 bash train.sh --job.config_file flame/models/fla.toml --model.config configs/gla_340M.json ...
```

### Slurm Job Submission
```bash
# Single GPU job
sbatch flame_train.job

# Multi-GPU job (4 GPUs)
sbatch flame_multigpu.job
```

### Linting
```bash
pre-commit run --all-files
```
Uses isort (import sorting) and flake8 (max line length: 127).

### Alternative Training Scripts
- `flame.train` - Standard LLM training
- `flame.train_hgt` - HGT (Heterogeneous Graph Transformer) experiments

### Checkpoint Testing
```bash
# Run training with checkpoints (no MPI required)
python test_checkpoint.py --clean --steps 10 --checkpoint_interval 5 --model_config configs/gla_340M.json

# Resume from checkpoint
python test_checkpoint.py --resume --steps 15 --model_config configs/gla_340M.json
```

### Help
```bash
bash train.sh -h
```

## Architecture

### Core Training Pipeline (`flame/train.py`)
Entry point that orchestrates: config loading → model building (AutoModelForCausalLM) → tokenizer → dataset/dataloader → parallelization → training loop → checkpointing.

### Configuration (`flame/config_manager.py`)
`JobConfig` class manages hierarchical TOML config with CLI overrides. Sections: job, model, training, optimizer, lr_scheduler, checkpoint, profiling, metrics, experimental, float8.

### Data Pipeline (`flame/data.py`)
- `BufferShuffledIterableDataset`: Token-level shuffling with resumability
- `build_dataset()`: Multi-dataset support with interleaving and weighted sampling
- `build_dataloader()`: Stateful DataLoader for distributed training with checkpoint resumption
- Supports online tokenization and variable-length sequence packing

### Parallelization (`flame/models/parallelize_fla.py`)
- **Tensor Parallelism (TP):** ColwiseParallel/RowwiseParallel for linear layers
- **Data Parallelism:** FSDP (sharded) and DDP (replicated)
- **Hybrid HSDP:** Combined DP_replicate + DP_shard
- **Context Parallelism (CP):** For long sequences
- Activation checkpointing and torch.compile support

### Checkpoint Management (`flame/components/checkpoint.py`)
`TrainState` dataclass tracks step, token count, elapsed time, loss history. Supports DCP format with async checkpointing modes.

### Model Conversion Utilities
- `flame/utils/convert_dcp_to_hf.py`: DCP → HuggingFace format
- `flame/utils/convert_hf_to_dcp.py`: HuggingFace → DCP format (for continual training)

## Key Training Parameters

- `--training.varlen`: Variable-length sequence packing (requires `batch_size=1`)
- `--training.seq_len`: Total packed sequence length per device
- `--training.context_len`: Max length per individual sample (truncates longer docs)
- `--training.compile`: Enable torch.compile
- `--training.dataset`: HuggingFace dataset identifier (comma-separated for multiple)
- `--training.data_probs`: Sampling weights for multiple datasets

## Supported Model Architectures

Configs in `configs/`: Transformer, GLA, Mamba, DeltaNet, Gated DeltaNet, GSA, HGRN2, KDA, Samba. Launch scripts: `launch_*.sh`.

**Note:** FLA model types (gla, transformer, delta_net, etc.) must be registered with transformers before use. Import `fla` first, then register:
```python
import fla
from fla.models import GLAConfig, GLAForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register('gla', GLAConfig)
AutoModelForCausalLM.register(GLAConfig, GLAForCausalLM)
```
The `transformer` model type requires `flash-attn` package. GLA models work without it.

## Custom Models

Add models under `custom_models/` (see `custom_models/sba/` for example):
1. Create config class extending `PretrainedConfig`
2. Create model class extending `PreTrainedModel`
3. Register with `AutoModelForCausalLM`, `AutoModel`, `AutoConfig` in `__init__.py`
4. Create JSON config with matching `model_type`

## Code Style

- Max line length: 127 characters
- Import sorting: isort
- Linter: flake8

## Dependencies

- Python ≥3.10
- torch ≥2.5 (recommend ≥2.6 for torch.compile)
- triton ≥3.0
- transformers ≥4.45.0
- fla (flash-linear-attention)
- torchtitan (specific commit: 0b44d4c)
- mpi4py (for MPI-based distributed training)
- tyro (config parsing)
