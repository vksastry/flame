# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Flame is a minimal, efficient distributed training framework for Large Language Models (LLMs) built on PyTorch's `torchtitan`. It specializes in Flash Linear Attention (FLA) architectures and integrates with Hugging Face transformers.

**Key features:** Online tokenization, dataset shuffling, variable-length sequence packing, multi-dataset support, 4D parallelism.

## Common Commands

### Installation
```bash
pip install .
pip uninstall flash-linear-attention && pip install -U --no-use-pep517 git+https://github.com/fla-org/flash-linear-attention
pip install git+https://github.com/pytorch/torchtitan.git@0b44d4c
```

### Training
```bash
# Single GPU debugging
NGPU=1 bash train.sh --job.config_file flame/models/fla.toml --model.config configs/transformer_340M.json ...

# Multi-GPU (default 8)
bash train.sh --job.config_file flame/models/fla.toml --model.config configs/gla_340M.json ...

# Multi-node: set MASTER_ADDR and MASTER_PORT environment variables
```

### Linting
```bash
pre-commit run --all-files
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

## Custom Models

Add models under `custom_models/` (see `custom_models/sba/` for example):
1. Create config class extending `PretrainedConfig`
2. Create model class extending `PreTrainedModel`
3. Register with `AutoModelForCausalLM`, `AutoModel`, `AutoConfig` in `__init__.py`
4. Create JSON config with matching `model_type`

## Dependencies

- Python ≥3.10
- torch ≥2.5 (recommend ≥2.6 for torch.compile)
- triton ≥3.0
- transformers ≥4.45.0
- fla (flash-linear-attention)
- torchtitan (specific commit: 0b44d4c)
