import argparse
import contextlib
import json
import os
import socket
import sys
import time
from datetime import timedelta

from mpi4py import MPI
import torch

import fla  # noqa: F401
from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from fla.ops.utils import prepare_position_ids
from torch.distributed.elastic.multiprocessing.errors import record
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.ft import FTParallelDims, init_ft_manager
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.metrics import (
    build_device_memory_monitor,
    build_metrics_processor,
    ensure_pp_loss_visible,
)
from torchtitan.components.optimizer import build_optimizers
from torchtitan.distributed import ParallelDims
from torchtitan.distributed import utils as dist_utils
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.protocols.train_spec import TrainSpec, get_train_spec, register_train_spec
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from transformers import AutoConfig, AutoModelForCausalLM, CONFIG_MAPPING
from fla.models import GLAConfig

from flame.components.checkpoint import TrainState
from flame.config_manager import JobConfig, TORCH_DTYPE_MAP
from flame.data_random_tokens import build_random_token_dataloader
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
master_addr = socket.gethostname() if rank == 0 else None
master_addr = comm.bcast(master_addr, root=0)
master_port = "29500"
os.environ["LOCAL_RANK"] = str(local_rank)
os.environ["MASTER_ADDR"] = master_addr
os.environ["MASTER_PORT"] = master_port


def parse_bench_args(args_list: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bench.vocab_size", dest="bench_vocab_size", type=int, default=None)
    parser.add_argument("--bench.num_samples", dest="bench_num_samples", type=int, default=None)
    parser.add_argument("--bench.seed", dest="bench_seed", type=int, default=0)
    bench_args, remaining = parser.parse_known_args(args_list)
    return bench_args, remaining


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


@record
def main(job_config: JobConfig, bench_args: argparse.Namespace) -> None:
    logger.info(f"Starting benchmark job: {job_config.job.description}")

    if job_config.experimental.custom_model_path:
        utils.import_module_from_path(job_config.experimental.custom_model_path)

    color = utils.NoColor if job_config.metrics.disable_color_printing else utils.Color

    if job_config.job.print_args:
        logger.info(
            f"{color.green}{json.dumps(job_config.to_dict(), indent=2, sort_keys=True)}{color.reset}"
        )

    gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

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

    device_memory_monitor = build_device_memory_monitor()
    gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")

    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    pp_mesh = world_mesh["pp"] if parallel_dims.pp_enabled else None

    dist_utils.set_determinism(
        world_mesh, device, job_config.training.seed, job_config.training.deterministic
    )

    train_spec = get_train_spec(job_config.model.name)

    logger.info(f"Loading model config from {job_config.model.config}")
    model_config = AutoConfig.from_pretrained(job_config.model.config)
    if parallel_dims.tp_enabled and getattr(model_config, "fuse_norm", False):
        logger.warning(
            f"{color.red}Fused norm is not compatible with tensor parallelism. Disabling it.{color.reset}"
        )
        model_config.fuse_norm = False
    if parallel_dims.loss_parallel_enabled and getattr(model_config, "fuse_linear_cross_entropy", False):
        logger.warning(
            f"{color.red}Loss parallel enabled. Disabling fused cross entropy for now.{color.reset}"
        )
        model_config.fuse_linear_cross_entropy = False

    vocab_size = bench_args.bench_vocab_size or model_config.vocab_size
    if vocab_size is None or vocab_size <= 0:
        raise ValueError("vocab_size must be > 0 for random token benchmarking")
    model_config.vocab_size = max(model_config.vocab_size or 0, vocab_size)

    logger.info(
        f"Building model from the config\n{color.green}{model_config}{color.reset}"
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_config)
        if (
            getattr(model_config, "fuse_linear_cross_entropy", False)
            and FusedLinearCrossEntropyLoss is not None
        ):
            model.criterion = FusedLinearCrossEntropyLoss(num_chunks=8 // parallel_dims.tp)
        model.apply(lambda m: setattr(m, "_is_hf_initialized", False))
    logger.info(f"{color.blue}\n{model}{color.reset}\n")

    model_converters = build_model_converters(job_config, parallel_dims)
    model_converters.convert(model)

    model_param_count, num_flops_per_token = get_nparams_and_flops(
        model, model_config, job_config.training.context_len
    )

    if job_config.checkpoint.create_seed_checkpoint:
        init_device = "cpu"
    elif job_config.training.enable_cpu_offload:
        init_device = "cpu"
    else:
        init_device = device_type

    if parallel_dims.pp_enabled:
        (
            pp_schedule,
            model_parts,
            has_first_stage,
            has_last_stage,
        ) = train_spec.pipelining_fn(
            model,
            pp_mesh,
            parallel_dims,
            job_config,
            device,
            model_config,
            train_spec.loss_fn,
        )
        del model

        for m in model_parts:
            train_spec.parallelize_fn(m, world_mesh, parallel_dims, job_config)
            m.to_empty(device=init_device)
            with torch.no_grad():
                m.post_init()
            m.train()

        ensure_pp_loss_visible(parallel_dims, job_config, color)
    else:
        train_spec.parallelize_fn(model, world_mesh, parallel_dims, job_config)
        model.to_empty(device=init_device)
        with torch.no_grad():
            model.post_init()
        model.train()
        model_parts = [model]

    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

    optimizers = train_spec.build_optimizers_fn(model_parts, job_config, ft_manager)
    lr_schedulers = train_spec.build_lr_schedulers_fn(optimizers, job_config)
    optimizers.register_step_post_hook(
        lambda *args, **kwargs: model_converters.post_optimizer_hook(model_parts)
    )

    train_state = TrainState()

    per_rank_samples = (
        job_config.training.steps
        * job_config.training.batch_size
        * job_config.training.gradient_accumulation_steps
    )
    num_samples = bench_args.bench_num_samples
    if num_samples is None:
        num_samples = per_rank_samples * dp_degree

    dataloader, train_sampler = build_random_token_dataloader(
        job_config=job_config,
        rank=dp_rank,
        world_size=dp_degree,
        seq_len=job_config.training.seq_len,
        vocab_size=vocab_size,
        num_samples=num_samples,
        seed=bench_args.bench_seed,
    )

    checkpoint = CheckpointManager(
        dataloader=dataloader,
        model_parts=model_parts,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"train_state": train_state},
        job_config=job_config,
        ft_manager=ft_manager,
    )

    if job_config.checkpoint.create_seed_checkpoint:
        assert world_size == 1, (
            "Must create seed checkpoint using a single device, to disable sharding"
        )
        assert job_config.checkpoint.enable_checkpoint, (
            "Must enable checkpointing when creating a seed checkpoint"
        )
        checkpoint.save(curr_step=0, force=True)
        logger.info("Created seed checkpoint")
        return

    checkpoint.load(step=job_config.checkpoint.load_step)
    metric_logger = build_metrics_processor(job_config, parallel_dims)
    metric_logger.num_flops_per_token = num_flops_per_token
    metric_logger.optimizers = optimizers
    metric_logger.lr_schedulers = lr_schedulers

    if train_state.step > 0 and len(metric_logger.data_loading_times) > 0:
        for idx, step in enumerate(train_state.log_steps):
            metric_logger.log(
                step,
                global_avg_loss=train_state.global_avg_losses[idx],
                global_max_loss=train_state.global_max_losses[idx],
            )

    data_iterator = iter(dataloader)

    train_context = dist_utils.get_train_context(
        parallel_dims.loss_parallel_enabled,
        job_config.experimental.enable_compiled_autograd,
    )
    if parallel_dims.tp_enabled or parallel_dims.pp_enabled:
        if job_config.training.mixed_precision_param == "float32":
            maybe_enable_amp = contextlib.nullcontext()
        else:
            maybe_enable_amp = torch.autocast(
                device_type,
                dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            )
    else:
        maybe_enable_amp = dist_utils.maybe_enable_amp(
            parallel_dims,
            job_config.training.mixed_precision_param,
            device_type,
        )

    device_memory_monitor.reset_peak_stats()

    global_batch_size = (
        job_config.training.batch_size
        * dp_degree
        * job_config.training.gradient_accumulation_steps
    )
    num_tokens_per_step = global_batch_size * job_config.training.seq_len
    logger.info(f"{color.red}***** Running training *****{color.reset}")
    logger.info(f"{color.green}  Training starts at step {train_state.step + 1}")
    logger.info(
        f"{color.green}  Number of tokens per sequence = {job_config.training.seq_len:,}"
    )
    logger.info(
        f"{color.green}  Gradient Accumulation steps = {job_config.training.gradient_accumulation_steps}"
    )
    logger.info(
        f"{color.green}  Instantaneous batch size (per device) = {job_config.training.batch_size:,}"
    )
    logger.info(
        f"{color.green}  Global batch size (w. parallel, distributed & accumulation) = {global_batch_size:,}"
        f" ({num_tokens_per_step:,} tokens)"
    )
    logger.info(
        f"{color.green}  Total optimization steps = {job_config.training.steps:,} "
        f"({job_config.training.steps * num_tokens_per_step:,} tokens)"
    )
    logger.info(
        f"{color.green}  Warmup steps = {job_config.lr_scheduler.warmup_steps:,}"
        f" ({job_config.lr_scheduler.warmup_steps * num_tokens_per_step:,} tokens)"
    )
    logger.info(
        f"{color.green}  Number of parameters = {model_param_count:,} {color.reset}"
    )

    with (
        maybe_enable_profiling(job_config, global_step=train_state.step) as torch_profiler,
        maybe_enable_memory_snapshot(
            job_config, global_step=train_state.step
        ) as memory_profiler,
    ):
        while train_state.step < job_config.training.steps:
            train_state.step += 1
            gc_handler.run(train_state.step)

            optimizers.zero_grad()

            losses = []
            for _ in range(job_config.training.gradient_accumulation_steps):
                data_load_start = time.perf_counter()
                batch = next(data_iterator)
                input_ids, labels = batch["input_ids"], batch["labels"]

                metric_logger.ntokens_since_last_log += labels.numel()
                metric_logger.data_loading_times.append(
                    time.perf_counter() - data_load_start
                )

                input_ids = input_ids.to(device_type)
                labels = labels.to(device_type)
                cu_seqlens = (
                    batch["cu_seqlens"].to(device_type)
                    if "cu_seqlens" in batch
                    else None
                )
                if cu_seqlens is not None:
                    position_ids = prepare_position_ids(cu_seqlens).to(torch.int32)
                else:
                    position_ids = (
                        torch.arange(0, input_ids.shape[1], device=device_type)
                        .repeat(input_ids.shape[0], 1)
                        .to(torch.int32)
                    )

                optional_context_parallel_ctx = (
                    dist_utils.create_context_parallel_ctx(
                        cp_mesh=world_mesh["cp"],
                        cp_buffers=[input_ids, labels, position_ids],
                        cp_seq_dims=[1, 1, 1],
                        cp_no_restore_buffers={input_ids, labels, position_ids},
                        cp_rotate_method=job_config.experimental.context_parallel_rotate_method,
                    )
                    if parallel_dims.cp_enabled
                    else None
                )

                if parallel_dims.pp_enabled:
                    with train_context(optional_context_parallel_ctx):
                        targets, pp_losses = (
                            (labels, []) if has_last_stage else (None, None)
                        )

                        if has_first_stage:
                            pp_schedule.step(input_ids, target=targets, losses=pp_losses)
                        else:
                            pp_schedule.step(target=targets, losses=pp_losses)

                    loss = (
                        torch.mean(torch.stack(pp_losses)).to(device)
                        if has_last_stage
                        else torch.tensor([-1.0], device=device)
                    )
                else:
                    with train_context(optional_context_parallel_ctx):
                        with maybe_enable_amp:
                            output = model(
                                input_ids=input_ids,
                                labels=labels,
                                position_ids=position_ids,
                                cu_seqlens=cu_seqlens,
                            )
                        loss = (
                            output.loss
                            / job_config.training.gradient_accumulation_steps
                        )
                        loss.backward()

                losses.append(loss)
            loss = sum(losses)

            grad_norm = dist_utils.clip_grad_norm_(
                [p for m in model_parts for p in m.parameters()],
                job_config.training.max_norm,
                foreach=True,
                pp_mesh=pp_mesh if parallel_dims.pp_enabled else None,
            )

            checkpoint.maybe_wait_for_staging()
            if job_config.training.skip_nan_inf and (
                grad_norm.isnan() or grad_norm.isinf()
            ):
                logger.warning(
                    f"Skipping optimizer step - detected invalid gradient norm: {grad_norm:.4f}"
                )
                optimizers.zero_grad()
                train_state.skipped_step += 1
            else:
                optimizers.step()
            lr_schedulers.step()

            if metric_logger.should_log(train_state.step):
                if (
                    parallel_dims.dp_replicate_enabled
                    or parallel_dims.dp_shard_enabled
                    or parallel_dims.cp_enabled
                ):
                    loss = loss.detach()
                    global_avg_loss, global_max_loss = (
                        dist_utils.dist_mean(
                            loss,
                            world_mesh["dp_cp"],
                        ),
                        dist_utils.dist_max(
                            loss,
                            world_mesh["dp_cp"],
                        ),
                    )
                else:
                    global_avg_loss = global_max_loss = loss.item()

                time_now = time.perf_counter()
                time_delta = time_now - metric_logger.time_last_log
                train_state.token += (
                    metric_logger.ntokens_since_last_log
                    * parallel_dims.world_size
                    / parallel_dims.non_data_parallel_size
                )
                train_state.elapsed += timedelta(seconds=time_delta)
                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                last_lr = lr_schedulers.schedulers[0].get_last_lr()[0]
                eta = (
                    train_state.elapsed
                    * (job_config.training.steps - train_state.step)
                    / train_state.step
                )
                metric_logger.log(
                    train_state.step,
                    global_avg_loss,
                    global_max_loss,
                    extra_metrics={
                        "optimizer/lr": last_lr,
                        "optimizer/grad_norm": grad_norm.item(),
                        "optimizer/skipped_step": train_state.skipped_step,
                    },
                )

                logger.info(
                    f"{color.blue}lr: {last_lr:.4e} gnorm: {grad_norm:5.2f} "
                    f"{color.magenta}[{str(train_state.elapsed).split('.')[0]:>8}<{str(eta).split('.')[0]:>8}]{color.reset}"
                )

            checkpoint.save(
                train_state.step, force=(train_state.step == job_config.training.steps)
            )

            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

            if train_state.step == 1:
                dist_utils.set_pg_timeouts(
                    timeout=timedelta(seconds=job_config.comm.train_timeout_seconds),
                    world_mesh=world_mesh,
                )

    if torch.distributed.get_rank() == 0:
        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)

    metric_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    init_logger()
    bench_args, remaining_args = parse_bench_args(sys.argv[1:])
    config = JobConfig()
    config.parse_args(remaining_args)
    main(config, bench_args)
    torch.distributed.destroy_process_group()
