#!/usr/bin/env python3
"""
Simple checkpoint test script for flame - tests save and resume functionality.
Does not require MPI.
"""

import os
import sys
import json
import time
import shutil
from datetime import timedelta

# Set up distributed environment for single GPU
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['LOCAL_RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
os.environ['WANDB_MODE'] = 'disabled'

import torch
import torch.distributed as dist
import fla
from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Register FLA models with transformers
from fla.models import (
    GLAConfig, GLAForCausalLM,
    DeltaNetConfig, DeltaNetForCausalLM,
    TransformerConfig, TransformerForCausalLM,
)

# Register model types
for cfg_cls, model_cls in [
    (GLAConfig, GLAForCausalLM),
    (DeltaNetConfig, DeltaNetForCausalLM),
    (TransformerConfig, TransformerForCausalLM),
]:
    try:
        AutoConfig.register(cfg_cls.model_type, cfg_cls)
        AutoModelForCausalLM.register(cfg_cls, model_cls)
    except ValueError:
        pass  # Already registered
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict
from torch.distributed.checkpoint.stateful import Stateful
import torch.distributed.checkpoint as dcp

import custom_models
from flame.config_manager import JobConfig
from flame.data import build_dataloader


class TrainState(Stateful):
    """Training state for checkpointing."""
    def __init__(self):
        self.step = 0
        self.global_avg_losses = []
        self.global_max_losses = []
        self.log_steps = []

    def state_dict(self):
        return {
            "step": self.step,
            "global_avg_losses": self.global_avg_losses,
            "global_max_losses": self.global_max_losses,
            "log_steps": self.log_steps,
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.global_avg_losses = state_dict.get("global_avg_losses", [])
        self.global_max_losses = state_dict.get("global_max_losses", [])
        self.log_steps = state_dict.get("log_steps", [])


def save_checkpoint(checkpoint_dir, model, optimizer, train_state, step):
    """Save checkpoint using DCP format."""
    checkpoint_path = os.path.join(checkpoint_dir, f"step-{step}")
    os.makedirs(checkpoint_path, exist_ok=True)

    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_state": train_state.state_dict(),
    }

    # Save as regular torch checkpoint for simplicity
    torch.save(state_dict, os.path.join(checkpoint_path, "checkpoint.pt"))
    print(f"Saved checkpoint at step {step} to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer, train_state):
    """Load checkpoint."""
    ckpt_file = os.path.join(checkpoint_path, "checkpoint.pt")
    if os.path.exists(ckpt_file):
        state_dict = torch.load(ckpt_file, map_location='cuda')
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        train_state.load_state_dict(state_dict["train_state"])
        print(f"Loaded checkpoint from {checkpoint_path}, resuming from step {train_state.step}")
        return True
    return False


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint step."""
    if not os.path.exists(checkpoint_dir):
        return None

    steps = []
    for d in os.listdir(checkpoint_dir):
        if d.startswith("step-"):
            try:
                step = int(d.split("-")[1])
                steps.append(step)
            except ValueError:
                pass

    if steps:
        return os.path.join(checkpoint_dir, f"step-{max(steps)}")
    return None


def run_training(config_file, dump_folder, model_config, tokenizer_path,
                 steps=10, checkpoint_interval=5, resume=False):
    """Run training with checkpointing."""

    # Initialize distributed (single GPU)
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=120))
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Total steps: {steps}")
    print(f"  Checkpoint interval: {checkpoint_interval}")
    print(f"  Resume: {resume}")
    print(f"  Dump folder: {dump_folder}")
    print(f"{'='*60}\n")

    # Load model config
    print("Loading model configuration...")
    model_cfg = AutoConfig.from_pretrained(model_config, trust_remote_code=True)

    # Build model
    print("Building model...")
    model = AutoModelForCausalLM.from_config(model_cfg, trust_remote_code=True)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Build optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Training state
    train_state = TrainState()

    # Checkpoint directory
    checkpoint_dir = os.path.join(dump_folder, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Try to resume from checkpoint
    if resume:
        latest_ckpt = find_latest_checkpoint(checkpoint_dir)
        if latest_ckpt:
            load_checkpoint(latest_ckpt, model, optimizer, train_state)
        else:
            print("No checkpoint found, starting from scratch")

    # Create dummy data for testing
    print("\nStarting training loop...")
    model.train()

    start_step = train_state.step
    for step in range(start_step, steps):
        train_state.step = step

        # Create dummy batch
        batch_size = 1
        seq_len = 512
        input_ids = torch.randint(0, model_cfg.vocab_size, (batch_size, seq_len), device=device)
        labels = input_ids.clone()

        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()
        optimizer.step()

        train_state.global_avg_losses.append(loss.item())
        train_state.log_steps.append(step)

        print(f"Step {step + 1}/{steps} | Loss: {loss.item():.4f}")

        # Save checkpoint
        if (step + 1) % checkpoint_interval == 0:
            save_checkpoint(checkpoint_dir, model, optimizer, train_state, step + 1)

    # Final checkpoint
    if train_state.step + 1 != steps or (steps % checkpoint_interval != 0):
        save_checkpoint(checkpoint_dir, model, optimizer, train_state, steps)

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Final step: {train_state.step + 1}")
    print(f"Loss history: {train_state.global_avg_losses}")
    print(f"{'='*60}\n")

    dist.destroy_process_group()
    return train_state


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Checkpoint test for flame")
    parser.add_argument("--dump_folder", type=str, default="exp/checkpoint-test")
    parser.add_argument("--model_config", type=str, default="configs/transformer_340M.json")
    parser.add_argument("--tokenizer", type=str, default="fla-hub/transformer-1.3B-100B")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--clean", action="store_true", help="Clean existing checkpoints")
    args = parser.parse_args()

    if args.clean and os.path.exists(args.dump_folder):
        print(f"Cleaning {args.dump_folder}...")
        shutil.rmtree(args.dump_folder)

    run_training(
        config_file="flame/models/fla.toml",
        dump_folder=args.dump_folder,
        model_config=args.model_config,
        tokenizer_path=args.tokenizer,
        steps=args.steps,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
