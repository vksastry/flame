<div align="center">

# Flame

### Flash Linear Attention Made Easy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/pytorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Transformers-orange)](https://huggingface.co/)

A minimal and efficient distributed training framework for Large Language Models built on [torchtitan](https://github.com/pytorch/torchtitan).

[Installation](#installation) | [Quick Start](#quick-start) | [Documentation](#training-configuration) | [Custom Models](#custom-models) | [Citation](#citation)

</div>

---

## Highlights

- **Minimal & Extensible** — Clean, modular codebase designed for easy customization and experimentation
- **Seamless Integration** — Native support for [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) and Hugging Face Transformers
- **Zero-Cost Preprocessing** — Online tokenization, dynamic dataset shuffling, and multi-dataset support without preprocessing overhead
- **Efficient Training** — Variable-length sequence packing, `torch.compile` support, and distributed training across multiple nodes
- **4D Parallelism** — Data, Tensor, Pipeline, and Context parallelism (coming soon)

## Supported Architectures

| Architecture | Sizes | Config Files |
|-------------|-------|--------------|
| Transformer | 340M, 1B, 7B | `transformer_*.json` |
| GLA (Gated Linear Attention) | 340M, 7B | `gla_*.json` |
| Mamba / Mamba2 | 340M, 1B | `mamba_*.json`, `mamba2_*.json` |
| DeltaNet | 340M, 1B | `delta_net_*.json` |
| Gated DeltaNet | 340M, 1B | `gated_deltanet_*.json` |
| GSA | 340M, 1B | `gsa_*.json` |
| HGRN2 | 340M | `hgrn2_*.json` |
| Samba | 1B | `samba_*.json` |
| SBA (Stick-Breaking Attention) | 340M | `sba_*.json` |

## Installation

### Prerequisites

- Python >= 3.10
- CUDA-compatible GPU
- PyTorch >= 2.5 (>= 2.6 recommended for `torch.compile`)
- Triton >= 3.0

### Setup

```bash
# Clone the repository
git clone https://github.com/fla-org/flame.git
cd flame

# Install flame
pip install .

# Install flash-linear-attention (latest version)
pip uninstall flash-linear-attention
pip install -U --no-use-pep517 git+https://github.com/fla-org/flash-linear-attention

# Install torchtitan (specific version required)
pip install git+https://github.com/pytorch/torchtitan.git@0b44d4c
```

## Quick Start

### Dataset Preparation

Download datasets locally for stable training (recommended):

```python
from datasets import load_dataset

# Full dataset
dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="default",
    num_proc=64,
    cache_dir="/your/cache/path"
)

# 100B token subset (for smaller experiments)
dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-100BT",
    num_proc=64,
    cache_dir="/your/cache/path"
)
```

### Basic Training

Train a 340M Transformer model on FineWeb-Edu:

```bash
bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/transformer-340M \
  --model.config configs/transformer_340M.json \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --optimizer.name AdamW \
  --optimizer.lr 1e-3 \
  --lr_scheduler.warmup_steps 1024 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 1 \
  --training.seq_len 65536 \
  --training.context_len 4096 \
  --training.varlen \
  --training.steps 20480 \
  --training.dataset HuggingFaceFW/fineweb-edu \
  --training.dataset_name sample-100BT \
  --training.compile \
  --checkpoint.interval 2048
```

**GPU Configuration:**
- Default: 8 GPUs
- Single GPU debugging: `NGPU=1 bash train.sh ...`
- Custom count: `NGPU=4 bash train.sh ...`

## Training Configuration

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--training.steps` | Total number of training steps |
| `--training.batch_size` | Batch size per device (must be 1 with `--training.varlen`) |
| `--training.seq_len` | Packed sequence length fed to the model |
| `--training.context_len` | Maximum length per individual sample |
| `--training.varlen` | Enable variable-length sequence packing |
| `--training.compile` | Enable `torch.compile` acceleration |
| `--training.gradient_accumulation_steps` | Gradient accumulation steps |
| `--lr_scheduler.warmup_steps` | Learning rate warmup steps |
| `--lr_scheduler.decay_type` | Scheduler type: `cosine`, `wsd`, or `linear` |

> **Note:** Global batch size = `batch_size × gradient_accumulation_steps × num_gpus`
> Each step processes `global_batch_size × seq_len` tokens.

For all available parameters:

```bash
bash train.sh -h
```

### Variable-Length Training

Enable efficient training by packing multiple documents into single sequences:

```bash
--training.varlen \
--training.seq_len 65536 \    # Total packed sequence length
--training.context_len 4096   # Max tokens per document
```

- **`seq_len`**: Total length of the packed sequence per device
- **`context_len`**: Maximum tokens per document (longer documents are split)

### Multi-Dataset Training

Train on multiple datasets with custom sampling weights:

```bash
--training.dataset HuggingFaceFW/fineweb-edu,OpenCoder-LLM/opc-fineweb-code-corpus,math-ai/AutoMathText \
--training.data_probs 0.7,0.2,0.1
```

### Learning Rate Schedules

**Cosine Schedule:**
```bash
--lr_scheduler.decay_type cosine
--lr_scheduler.warmup_steps 1024
--lr_scheduler.lr_min 0.1  # Min LR as ratio of max
```

**Warmup-Stable-Decay (WSD):**
```bash
--lr_scheduler.decay_type wsd
--lr_scheduler.decay_ratio 0.2  # Last 20% of steps for decay
```

## Parallelism

### Data Parallelism

```bash
# FSDP (Fully Sharded Data Parallel)
--training.data_parallel_shard_degree 8

# DDP (Distributed Data Parallel)
--training.data_parallel_replicate_degree 8

# HSDP (Hybrid: replicate + shard)
--training.data_parallel_replicate_degree 2
--training.data_parallel_shard_degree 4
```

### Tensor Parallelism

```bash
--training.tensor_parallel_degree 2
```

### Context Parallelism

```bash
--experimental.context_parallel_degree 2
```

## Multi-Node Training

Set environment variables before running on each node:

```bash
export MASTER_ADDR=<master_ip>
export MASTER_PORT=<port>
export NNODE=<num_nodes>
bash train.sh ...
```

For Slurm clusters, see the [torchtitan Slurm script](https://github.com/pytorch/torchtitan/blob/main/multinode_trainer.slurm).

## Checkpoints

### Automatic Conversion

Checkpoints are automatically converted to Hugging Face format during training.

### Manual Conversion

**DCP to Hugging Face:**
```bash
python -m flame.utils.convert_dcp_to_hf \
  --path <checkpoint_dir> \
  --step <step_number> \
  --config <model_config.json> \
  --tokenizer <tokenizer_path>
```

**Hugging Face to DCP (for continual training):**
```bash
python -m flame.utils.convert_hf_to_dcp \
  --model <hf_model_path> \
  --checkpoint <output_dir>/checkpoint/step-0
```

## Custom Models

Add custom architectures with Hugging Face integration:

1. **Create model directory:**
   ```
   custom_models/
   └── your_model/
       ├── __init__.py
       ├── config_your_model.py
       └── modeling_your_model.py
   ```

2. **Implement configuration** (inherit from `PretrainedConfig`):
   ```python
   # config_your_model.py
   from transformers import PretrainedConfig

   class YourModelConfig(PretrainedConfig):
       model_type = "your_model"
       # ... define your config attributes
   ```

3. **Implement model** (inherit from `PreTrainedModel`):
   ```python
   # modeling_your_model.py
   from transformers import PreTrainedModel

   class YourModelForCausalLM(PreTrainedModel):
       # ... implement your model
   ```

4. **Register with AutoClasses:**
   ```python
   # __init__.py
   from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
   from .config_your_model import YourModelConfig
   from .modeling_your_model import YourModel, YourModelForCausalLM

   AutoConfig.register("your_model", YourModelConfig)
   AutoModel.register(YourModelConfig, YourModel)
   AutoModelForCausalLM.register(YourModelConfig, YourModelForCausalLM)
   ```

5. **Create config file** (`configs/your_model.json`):
   ```json
   {
     "model_type": "your_model",
     ...
   }
   ```

See `custom_models/sba/` for a complete example.

## Project Structure

```
flame/
├── flame/
│   ├── train.py              # Main training script
│   ├── data.py               # Data loading pipeline
│   ├── config_manager.py     # Configuration management
│   ├── models/
│   │   ├── parallelize_fla.py    # Parallelization strategies
│   │   └── fla.toml              # Default config template
│   ├── components/
│   │   └── checkpoint.py     # Checkpoint management
│   └── utils/
│       ├── convert_dcp_to_hf.py  # DCP → HF conversion
│       └── convert_hf_to_dcp.py  # HF → DCP conversion
├── configs/                  # Model configurations
├── custom_models/            # Custom model implementations
└── train.sh                  # Training launcher
```

## Troubleshooting

### torch.compile Issues

If experiencing compilation errors with FLA kernels:
- Ensure `torch >= 2.6` and `triton >= 3.0`
- Try disabling compile: remove `--training.compile`

### Dataset Streaming Instability

For stable training, download datasets locally rather than streaming:
```python
dataset = load_dataset("...", cache_dir="/local/path")
```

### Memory Issues

- Reduce `--training.seq_len`
- Enable CPU offloading: `--training.enable_cpu_offload`
- Use activation checkpointing (enabled by default)

## Citation

If you find Flame useful, please cite:

```bibtex
@software{yang2025flame,
  title  = {Flame: Flash Linear Attention Made Easy},
  author = {Zhang, Yu and Yang, Songlin},
  url    = {https://github.com/fla-org/flame},
  year   = {2025}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Flame is built on top of [torchtitan](https://github.com/pytorch/torchtitan) and integrates with [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) and [Hugging Face Transformers](https://github.com/huggingface/transformers).
