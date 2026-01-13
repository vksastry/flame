# Paper: Linear Attention Architectures for Geopotential Height Forecasting

This directory contains the LaTeX source and experiment configurations for the paper comparing linear attention architectures on geopotential height forecasting.

## Target Venue

**MLDADS 2026** (8th Workshop on Machine Learning and Data Assimilation for Dynamical Systems)
- Part of ICCS 2026
- Format: Springer LNCS
- Submission deadline: January 23, 2026

## Directory Structure

```
paper/
├── main.tex              # Main LaTeX document
├── references.bib        # Bibliography
├── README.md             # This file
├── run_experiments.sh    # Script to run all experiments
└── configs/              # Experiment configurations
    ├── hgt_base.toml         # Base config (template)
    ├── hgt_transformer.toml  # Transformer baseline
    ├── hgt_gla.toml          # Gated Linear Attention
    ├── hgt_gated_deltanet.toml # Gated DeltaNet
    ├── hgt_deltanet.toml     # DeltaNet
    ├── hgt_mamba2.toml       # Mamba2 (State Space Model)
    └── hgt_kda.toml          # KDA (Kimi Delta Attention)
```

## Architectures Compared

| Model | Type | Key Mechanism |
|-------|------|---------------|
| Transformer | Baseline | Softmax attention (O(n²)) |
| GLA | Linear Attention | Data-dependent decay + gating |
| Gated DeltaNet | Linear Attention | Delta rule + output gating |
| DeltaNet | Linear Attention | Delta rule update |
| Mamba2 | State Space | Selective SSM (SSD) |
| KDA | Linear Attention | Channel-wise gating (Kimi Linear) |

## Running Experiments

### Prerequisites

1. Install Flame and dependencies (see main README)
2. Download NCEP geopotential height data
3. Update data path in config files or pass via command line

### Quick Start

```bash
# Make script executable
chmod +x run_experiments.sh

# Run all experiments
./run_experiments.sh all /path/to/hgt_all_new.npy

# Run single model
./run_experiments.sh gla /path/to/hgt_all_new.npy
```

### Manual Execution

```bash
# From repo root
NGPU=4 bash train.sh \
    --job.config_file paper/configs/hgt_gla.toml \
    --training.data_files /path/to/hgt_all_new.npy
```

## Experiment Configuration

All experiments use consistent hyperparameters for fair comparison:

| Parameter | Value |
|-----------|-------|
| Model size | ~340M parameters |
| Input length | 6 timesteps (36 hours) |
| Target length | 4 timesteps (24 hours) |
| Pressure level | 500 hPa (index 7) |
| Batch size | 8 |
| Learning rate | 3e-4 |
| Warmup steps | 1,000 |
| Total steps | 50,000 |
| Optimizer | AdamW |

## Compiling the Paper

```bash
# Compile LaTeX
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use latexmk:
```bash
latexmk -pdf main.tex
```

## Key References

- **GLA**: Yang et al., "Gated Linear Attention Transformers with Hardware-Efficient Training" (ICML 2024)
- **Mamba2**: Dao & Gu, "Transformers are SSMs" (ICML 2024)
- **KDA**: Moonshot AI, "Kimi Linear: An Expressive, Efficient Attention Architecture" (2025)
- **NCEP Data**: Kalnay et al., "The NCEP/NCAR 40-year reanalysis project" (1996)

## TODO

- [ ] Run experiments for all 6 architectures
- [ ] Collect RMSE, MAE, ACC metrics
- [ ] Generate forecast horizon plots
- [ ] Add efficiency benchmarks (throughput, memory)
- [ ] Complete Results section
- [ ] Add architecture diagram figure
