import argparse
import json
import os
import socket
import sys
import time

from mpi4py import MPI
import torch
import torch.distributed as dist

import fla  # noqa: F401
from torch.distributed.elastic.multiprocessing.errors import record
from torchtitan.components.ft import FTParallelDims, init_ft_manager
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.distributed import ParallelDims
from torchtitan.distributed import utils as dist_utils
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.protocols.train_spec import TrainSpec, get_train_spec, register_train_spec
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from transformers import AutoConfig, AutoModelForCausalLM, CONFIG_MAPPING
from fla.models import GLAConfig

from flame.config_manager import JobConfig
from flame.data_random_hgt import build_random_hgt_dataloader
from flame.models.hgt_model import HGTModelWrapper
from flame.models.parallelize_fla import parallelize_fla
from flame.models.pipeline_fla import pipeline_fla
from flame.tools.utils import get_nparams_and_flops


CONFIG_MAPPING["gla"] = GLAConfig

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
local_rank = rank % max(torch.cuda.device_count(), 1)
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(size)
os.environ["LOCAL_RANK"] = str(local_rank)
master_addr = socket.gethostname() if rank == 0 else None
master_addr = comm.bcast(master_addr, root=0)
master_port = "29500"
os.environ["MASTER_ADDR"] = master_addr
os.environ["MASTER_PORT"] = master_port


def parse_bench_args(args_list: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bench.input_len", dest="bench_input_len", type=int, default=16)
    parser.add_argument("--bench.target_len", dest="bench_target_len", type=int, default=1)
    parser.add_argument("--bench.channels", dest="bench_channels", type=int, default=1)
    parser.add_argument("--bench.height", dest="bench_height", type=int, default=73)
    parser.add_argument("--bench.width", dest="bench_width", type=int, default=144)
    parser.add_argument("--bench.num_samples", dest="bench_num_samples", type=int, default=10000)
    parser.add_argument("--bench.val_num_samples", dest="bench_val_num_samples", type=int, default=1000)
    parser.add_argument("--bench.warmup_steps", dest="bench_warmup_steps", type=int, default=10)
    parser.add_argument("--bench.bench_steps", dest="bench_bench_steps", type=int, default=50)
    parser.add_argument("--bench.seed", dest="bench_seed", type=int, default=0)
    bench_args, remaining = parser.parse_known_args(args_list)
    return bench_args, remaining


def _rank0() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


register_train_spec(
    TrainSpec(
        name="fla",
        cls=AutoModelForCausalLM,
        config=AutoConfig,
        parallelize_fn=parallelize_fla,
        pipelining_fn=pipeline_fla,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=None,
        build_tokenizer_fn=None,
        build_loss_fn=build_cross_entropy_loss,
    )
)


def _maybe_sync(device_type: str) -> None:
    if device_type == "cuda":
        torch.cuda.synchronize()


@record
def main(job_config: JobConfig, bench_args: argparse.Namespace) -> None:
    logger.info("Starting HGT random benchmark")

    device_module, device_type = utils.device_module, utils.device_type
    device = torch.device(f"{device_type}:{int(local_rank)}")
    device_module.set_device(device)
    ft_manager = init_ft_manager(job_config)

    world_size = int(size)
    if not ft_manager.enabled:
        parallel_dims = ParallelDims(
            dp_shard=job_config.training.data_parallel_shard_degree,
            dp_replicate=job_config.training.data_parallel_replicate_degree,
            cp=job_config.experimental.context_parallel_degree,
            tp=job_config.training.tensor_parallel_degree,
            pp=job_config.experimental.pipeline_parallel_degree,
            world_size=world_size,
            enable_loss_parallel=not job_config.training.disable_loss_parallel,
        )
    else:
        parallel_dims = FTParallelDims(
            dp_shard=job_config.training.data_parallel_shard_degree,
            dp_replicate=job_config.training.data_parallel_replicate_degree,
            cp=job_config.experimental.context_parallel_degree,
            tp=job_config.training.tensor_parallel_degree,
            pp=job_config.experimental.pipeline_parallel_degree,
            world_size=world_size,
            enable_loss_parallel=not job_config.training.disable_loss_parallel,
            ft_manager=ft_manager,
        )

    dist_utils.init_distributed(job_config)
    torch.distributed.barrier()

    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    if parallel_dims.pp_enabled:
        raise NotImplementedError("Pipeline parallelism is not supported in this benchmark")

    dist_utils.set_determinism(
        world_mesh, device, job_config.training.seed, job_config.training.deterministic
    )

    train_spec = get_train_spec(job_config.model.name)

    logger.info("Building random HGT dataloader...")
    train_loader, _, train_sampler, _ = build_random_hgt_dataloader(
        job_config=job_config,
        rank=dp_rank,
        world_size=dp_degree,
        input_len=bench_args.bench_input_len,
        target_len=bench_args.bench_target_len,
        channels=bench_args.bench_channels,
        height=bench_args.bench_height,
        width=bench_args.bench_width,
        num_samples=bench_args.bench_num_samples,
        val_num_samples=bench_args.bench_val_num_samples,
        seed=bench_args.bench_seed,
    )

    logger.info(f"Loading model config from {job_config.model.config}")
    model_config = AutoConfig.from_pretrained(job_config.model.config)
    if parallel_dims.tp_enabled and getattr(model_config, "fuse_norm", False):
        logger.warning("Fused norm is not compatible with tensor parallelism. Disabling it.")
        model_config.fuse_norm = False
    if parallel_dims.loss_parallel_enabled and getattr(model_config, "fuse_linear_cross_entropy", False):
        logger.warning("Loss parallel enabled. Disabling fused cross entropy.")
        model_config.fuse_linear_cross_entropy = False

    logger.info(f"Building model from the config\n{model_config}")
    with torch.device("meta"):
        base_model = AutoModelForCausalLM.from_config(model_config)
        model = HGTModelWrapper(
            base_model,
            T_out=bench_args.bench_target_len,
            C=bench_args.bench_channels,
            H=bench_args.bench_height,
            W=bench_args.bench_width,
        )
        model.apply(lambda m: setattr(m, "_is_hf_initialized", False))

    model_converters = build_model_converters(job_config, parallel_dims)
    model_converters.convert(model)

    model_param_count, _ = get_nparams_and_flops(
        model, model_config, job_config.training.context_len
    )

    if job_config.checkpoint.create_seed_checkpoint:
        raise RuntimeError("Seed checkpointing is not supported in benchmark mode.")

    train_spec.parallelize_fn(model, world_mesh, parallel_dims, job_config)
    model.to_empty(device=device_type)
    with torch.no_grad():
        model.post_init()
    model.train()

    optimizers = train_spec.build_optimizers_fn([model], job_config, ft_manager)
    lr_schedulers = train_spec.build_lr_schedulers_fn(optimizers, job_config)

    train_data_iterator = iter(train_loader)
    train_context = dist_utils.get_train_context(
        parallel_dims.loss_parallel_enabled,
        job_config.experimental.enable_compiled_autograd,
    )
    maybe_enable_amp = dist_utils.maybe_enable_amp(
        parallel_dims,
        job_config.training.mixed_precision_param,
        device_type,
    )

    global_batch_size = (
        job_config.training.batch_size
        * dp_degree
        * job_config.training.gradient_accumulation_steps
    )
    tokens_per_step = global_batch_size * bench_args.bench_input_len

    if _rank0():
        logger.info(
            "Benchmark settings: "
            f"input_len={bench_args.bench_input_len}, target_len={bench_args.bench_target_len}, "
            f"C={bench_args.bench_channels}, H={bench_args.bench_height}, W={bench_args.bench_width}, "
            f"batch_size={job_config.training.batch_size}, grad_accum={job_config.training.gradient_accumulation_steps}"
        )
        logger.info(f"Parameters: {model_param_count:,}")

    def run_steps(num_steps: int) -> None:
        nonlocal train_data_iterator
        epoch = 0
        for _ in range(num_steps):
            optimizers.zero_grad()
            losses = []
            for _ in range(job_config.training.gradient_accumulation_steps):
                try:
                    batch = next(train_data_iterator)
                except StopIteration:
                    epoch += 1
                    if train_sampler is not None:
                        train_sampler.set_epoch(epoch)
                    train_data_iterator = iter(train_loader)
                    batch = next(train_data_iterator)

                inputs = batch["inputs"].to(device_type)
                targets = batch["targets"].to(device_type)
                position_ids = (
                    torch.arange(0, inputs.shape[1], device=device_type)
                    .repeat(inputs.shape[0], 1)
                    .to(torch.int32)
                )
                with train_context(None):
                    with maybe_enable_amp:
                        output = model(
                            inputs=inputs,
                            labels=targets,
                            position_ids=position_ids,
                            cu_seqlens=None,
                        )
                        loss = (
                            output.loss
                            / job_config.training.gradient_accumulation_steps
                        )
                    loss.backward()
                losses.append(loss)

            loss = sum(losses)
            grad_norm = dist_utils.clip_grad_norm_(
                [p for p in model.parameters()],
                job_config.training.max_norm,
                foreach=True,
            )
            if job_config.training.skip_nan_inf and (
                grad_norm.isnan() or grad_norm.isinf()
            ):
                logger.warning(
                    f"Skipping optimizer step - invalid gradient norm: {grad_norm:.4f}"
                )
                optimizers.zero_grad()
            else:
                optimizers.step()
            lr_schedulers.step()

    if bench_args.bench_warmup_steps > 0:
        if _rank0():
            logger.info(f"Warmup: {bench_args.bench_warmup_steps} steps")
        run_steps(bench_args.bench_warmup_steps)

    torch.distributed.barrier()
    _maybe_sync(device_type)
    start = time.perf_counter()
    run_steps(bench_args.bench_bench_steps)
    _maybe_sync(device_type)
    torch.distributed.barrier()
    elapsed = time.perf_counter() - start

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        elapsed_tensor = torch.tensor(elapsed, device=device)
        torch.distributed.all_reduce(elapsed_tensor, op=torch.distributed.ReduceOp.MAX)
        elapsed = float(elapsed_tensor.item())

    steps_per_sec = bench_args.bench_bench_steps / max(elapsed, 1e-9)
    tokens_per_sec = steps_per_sec * tokens_per_step

    if _rank0():
        logger.info(f"Benchmark elapsed: {elapsed:.4f}s")
        logger.info(f"Steps/sec: {steps_per_sec:.2f}")
        logger.info(f"Tokens/sec: {tokens_per_sec:.2f}")
        summary = {
            "steps": bench_args.bench_bench_steps,
            "elapsed_seconds": elapsed,
            "steps_per_sec": steps_per_sec,
            "tokens_per_sec": tokens_per_sec,
            "tokens_per_step": tokens_per_step,
            "global_batch_size": global_batch_size,
            "input_len": bench_args.bench_input_len,
            "target_len": bench_args.bench_target_len,
            "channels": bench_args.bench_channels,
            "height": bench_args.bench_height,
            "width": bench_args.bench_width,
            "model_params": model_param_count,
        }
        os.makedirs(job_config.job.dump_folder, exist_ok=True)
        summary_path = os.path.join(job_config.job.dump_folder, "hgt_benchmark_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Wrote benchmark summary to {summary_path}")


if __name__ == "__main__":
    init_logger()
    bench_args, remaining_args = parse_bench_args(sys.argv[1:])
    config = JobConfig()
    config.parse_args(remaining_args)
    config.training.seq_len = bench_args.bench_input_len
    config.training.context_len = bench_args.bench_input_len
    config.training.steps = bench_args.bench_warmup_steps + bench_args.bench_bench_steps
    setattr(config.training, "input_len", bench_args.bench_input_len)
    setattr(config.training, "target_len", bench_args.bench_target_len)
    main(config, bench_args)
    torch.distributed.destroy_process_group()
