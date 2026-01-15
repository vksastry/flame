# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pdb
import json
import os
import time
from datetime import timedelta
from mpi4py import MPI
import sys
import socket
import torch
import random
import hashlib
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
local_rank = rank % torch.cuda.device_count() 
os.environ['RANK']=str(rank)
os.environ['WORLD_SIZE']=str(size)
#master_addr = "localhost"
master_addr = socket.gethostname() if rank == 0 else None
master_addr = comm.bcast(master_addr, root=0)
master_port = "29500"
os.environ["LOCAL_RANK"] = str(local_rank)
os.environ["MASTER_ADDR"] = master_addr
os.environ["MASTER_PORT"] = master_port

print("Hello World from rank {} of {} and localrank {} on {}".format(rank, size, local_rank, socket.gethostname()))
#os.environ['WANDB_MODE'] = 'disabled'
import fla  # noqa
from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from fla.ops.utils import prepare_position_ids
from torch.distributed.elastic.multiprocessing.errors import record
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.ft import FTParallelDims, init_ft_manager
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.metrics import build_device_memory_monitor, build_metrics_processor, ensure_pp_loss_visible
from torchtitan.components.optimizer import build_optimizers
from torchtitan.distributed import ParallelDims
from torchtitan.distributed import utils as dist_utils
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.protocols.train_spec import TrainSpec, get_train_spec, register_train_spec
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from flame.data_hgt import build_hgt_dataloader
import custom_models
import logging
from flame.components.checkpoint import TrainState
from flame.config_manager import JobConfig
from flame.data import build_dataloader, build_dataset
from flame.models.parallelize_fla import parallelize_fla
from flame.models.pipeline_fla import pipeline_fla
from flame.tools.utils import get_nparams_and_flops
from flame.models.hgt_model import HGTModelWrapper
from transformers import CONFIG_MAPPING
from fla.models import GLAConfig
CONFIG_MAPPING["gla"] = GLAConfig
from utils.plotting_function import plot_geopotential_comparison 
import tempfile, os, multiprocessing as mp
import torch.distributed as dist
print("TEMP DIR:", tempfile.gettempdir())
print("PYTHON TMPDIR:", os.environ.get("TMPDIR"))
print("CWD:", os.getcwd())
print("MP start method:", mp.get_start_method(allow_none=True))
try:
    import wandb
except ImportError:
    wandb = None

def build_tokenizer(job_config: JobConfig) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(job_config.model.tokenizer_path)

def _rank0():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

try:
    from torch.distributed.tensor import DTensor
except Exception:
    DTensor = None

def checksum_named_param_local(model, name_substr: str, n: int = 1024):
    matches = []
    for nme, p in model.named_parameters():
        if name_substr in nme:
            matches.append((nme, p))
    if len(matches) == 0:
        return None, None
    if len(matches) > 1:
        # pick the shortest/most specific match (usually correct)
        matches.sort(key=lambda x: len(x[0]))
    nme, p = matches[0]

    t = p.detach()
    if DTensor is not None and isinstance(t, DTensor):
        t = t.to_local()
    t = t.float().reshape(-1)[:n].contiguous().cpu()
    h = hashlib.sha256(t.numpy().tobytes()).hexdigest()
    return nme, h

def debug_embeddings_optimizer_state(model, optimizers, tag="", max_print=40):
    """
    Prints:
      - whether embeddings/lm_head are present
      - whether they are in optimizer param groups
      - whether they have grad
      - whether optimizer has state for them
    """
    if not _rank0():
        return

    # pick the first optimizer (torchtitan returns a list)
    opt = optimizers[0] if isinstance(optimizers, (list, tuple)) else optimizers

    name_to_param = dict(model.named_parameters())

    def find_name(substr):
        for n in name_to_param:
            if substr in n:
                return n
        return None

    emb_name = find_name("embeddings.weight")
    lm_name  = find_name("lm_head.weight")

    print(f"\n[DEBUG:{tag}] ----- model/optimizer debug -----")
    print(f"[DEBUG:{tag}] found embeddings.weight name = {emb_name}")
    print(f"[DEBUG:{tag}] found lm_head.weight name     = {lm_name}")

    # Build a quick set of params in optimizer
    opt_params = set()
    for g in opt.param_groups:
        for p in g["params"]:
            opt_params.add(p)

    def report(pname):
        if pname is None:
            return
        p = name_to_param[pname]
        in_opt = p in opt_params
        has_grad = (p.grad is not None)
        has_state = (p in opt.state) and (len(opt.state[p]) > 0)
        # Adam-like state often has keys: step, exp_avg, exp_avg_sq
        state_keys = list(opt.state[p].keys())[:10] if has_state else []
        print(f"[DEBUG:{tag}] {pname}")
        print(f"            requires_grad={p.requires_grad}  in_optimizer={in_opt}")
        print(f"            grad_present={has_grad}  grad_norm={(p.grad.norm().item() if has_grad else None)}")
        print(f"            opt_has_state={has_state}  opt_state_keys={state_keys}")

    report(emb_name)
    report(lm_name)

    # Also: how many parameters have grad at all (useful)
    n_total = 0
    n_grad = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            n_total += 1
            if p.grad is not None:
                n_grad += 1
    print(f"[DEBUG:{tag}] trainable params with grad: {n_grad}/{n_total}")
    print(f"[DEBUG:{tag}] --------------------------------\n")


@torch.no_grad()
def evaluate(
    model,
    val_loader,
    plot,
    plot_path,
    device_type,
    lat, 
    lon, 
    vmin, 
    vmax,
    max_batches: int | None = None,
):
    """
    Evaluate regression loss on validation data.

    Args:
        max_batches: None => evaluate entire val_loader
                    int  => evaluate only this many batches

    Returns:
        mean validation loss (float), averaged across all ranks
    """
    model.eval()

    total_loss = 0.0
    total_batches = 0

    # choose one of the batch bidx to plot
    if plot and max_batches is not None:
        rng = random.Random(0)
        plot_bidx = rng.randint(0, max_batches - 1)

    for bidx, batch in enumerate(val_loader):
        if max_batches is not None and bidx >= max_batches:
            break

        inputs = batch["inputs"].to(device_type, non_blocking=True)
        targets = batch["targets"].to(device_type, non_blocking=True)

        cu_seqlens = batch["cu_seqlens"].to(device_type) if "cu_seqlens" in batch else None

        T_in = inputs.shape[1]
        if cu_seqlens is not None:
            position_ids = prepare_position_ids(cu_seqlens).to(torch.int32)
        else:
            position_ids = (
                torch.arange(T_in, device=inputs.device)
                .unsqueeze(0)
                .expand(inputs.shape[0], T_in)
                .to(torch.int32)
            )

        output = model(
            inputs=inputs,
            labels=targets,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
        )

        loss = output.loss
        if loss is None:
            raise RuntimeError("evaluate(): output.loss is None — did you pass labels?")

        total_loss += float(loss.detach().float().item())
        total_batches += 1
        # plotting 
        if plot and plot_path is not None and bidx == plot_bidx :
            dataset = val_loader.dataset
            pred_phys = dataset.denormalize(output.logits)
            truth_phys = dataset.denormalize(targets)
                    
            pred_phys = pred_phys.detach().float().cpu()[:,0,0,:,:]
            truth_phys = truth_phys.detach().float().cpu()[:,0,0,:,:]
            plot_geopotential_comparison(model_output=pred_phys,truth=truth_phys,lat=lat,lon=lon,timesteps=[0, 4, 8, 12],  # Plot specific timesteps
                                                         save_path=plot_path,  # Set to filename to save
                                                         vmin=vmin,
                                                         vmax=vmax)


    # Reduce across ranks (sum loss, sum batches)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        t = torch.tensor([total_loss, total_batches], device=inputs.device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        total_loss = float(t[0].item())
        total_batches = int(t[1].item())

    mean_loss = total_loss / max(total_batches, 1)
    model.train()
    return mean_loss

register_train_spec(
    TrainSpec(
        name="fla",
        cls=AutoModelForCausalLM,
        config=AutoConfig,
        parallelize_fn=parallelize_fla,
        pipelining_fn=pipeline_fla,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)


# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(job_config: JobConfig):
    logger.info(f"Starting weatherbench job: {job_config.job.description}")

    if job_config.experimental.custom_model_path:
        utils.import_module_from_path(job_config.experimental.custom_model_path)

    # used for colorful printing
    color = utils.NoColor if job_config.metrics.disable_color_printing else utils.Color

    if job_config.job.print_args:
        logger.info(
            f"{color.green}{json.dumps(job_config.to_dict(), indent=2, sort_keys=True)}{color.reset}"
        )

    # take control of garbage collection to avoid stragglers
    gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

    device_module, device_type = utils.device_module, utils.device_type
    device = torch.device(f"{device_type}:{int(local_rank)}")
    # Device has to be set before creating TorchFT manager.
    device_module.set_device(device)
    ft_manager = init_ft_manager(job_config)

    # init distributed
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
    # initialize device memory monitor and get peak flops for MFU calculation
    device_memory_monitor = build_device_memory_monitor()
    gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    if parallel_dims.pp_enabled:
        raise NotImplementedError(
            "Pipeline parallelism is not supported in this version"
        )
        """
        ! TODO[flame]: We need to fix the pipeline parallelism for flame
        [x] Match the key of models' components with the actual naming
        [ ] Fix the post-init and tie-embedding for pipeline parallelism, HF's transformer automatically
            forces to tie if head is None, we need to handle this case
        [ ]
        """
        pp_mesh = world_mesh["pp"]

    # Set random seed, and maybe enable deterministic mode (mainly for debugging, expect perf loss)
    dist_utils.set_determinism(
        world_mesh, device, job_config.training.seed, job_config.training.deterministic
    )
    train_spec = get_train_spec(job_config.model.name)
    
    # ----------------- Hgt dataloader -----------------
    logger.info("Building Hgt dataloader...")
    train_loader, val_loader, train_sampler, val_sampler, lat, lon, vmin, vmax = build_hgt_dataloader(
        job_config=job_config,
        rank=dp_rank,
        world_size=dp_degree,
    )

    logger.info(f"Loading model config from {job_config.model.config}")
    model_config = AutoConfig.from_pretrained(job_config.model.config)
    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. disable fused norm if TP is enabled
    # 3. vocab size from tokenizer
    # 4. context_len base on inputs
    if parallel_dims.tp_enabled:
        if model_config.fuse_norm:
            logger.warning(
                f"{color.red}"
                f"Fused norm is not compatible with tensor parallelism. "
                f"Disabling it for now."
                f"{color.reset}"
            )
            model_config.fuse_norm = False
    if parallel_dims.loss_parallel_enabled:
        if model_config.fuse_linear_cross_entropy:
            logger.warning(
                f"{color.red}"
                f"Loss parallel enabled. Disabling fused cross entropy for now."
                f"{color.reset}"
            )
            model_config.fuse_linear_cross_entropy = False
    #model_config.vocab_size = max(tokenizer.vocab_size, model_config.vocab_size)

    logger.info(
        f"Building model from the config\n{color.green}{model_config}{color.reset}"
    )
    with torch.device("meta"):
        base_model = AutoModelForCausalLM.from_config(model_config)
        C, H, W = 1, 73, 144 # need to get from the data 
        T_out = job_config.training.target_len 
        model = HGTModelWrapper(base_model, T_out=T_out, C=C, H=H, W=W)
        if (
            getattr(model_config, "fuse_linear_cross_entropy", False)
            and FusedLinearCrossEntropyLoss is not None
        ):
            model.criterion = FusedLinearCrossEntropyLoss(
                num_chunks=8 // parallel_dims.tp
            )
        # defer weight initialization until after parallelisms are applied
        model.apply(lambda m: setattr(m, "_is_hf_initialized", False))
    logger.info(f"{color.blue}\n{model}{color.reset}\n")

    # Build the collection of model converters. No-op if `model.converters` empty
    model_converters = build_model_converters(job_config, parallel_dims)
    model_converters.convert(model)

    # calculate model size and flops per token
    model_param_count, num_flops_per_token = get_nparams_and_flops(
        model, model_config, job_config.training.context_len
    )

    # move sharded model to CPU/GPU and initialize weights via DTensor
    if job_config.checkpoint.create_seed_checkpoint:
        init_device = "cpu"
    elif job_config.training.enable_cpu_offload:
        init_device = "cpu"
    else:
        init_device = device_type
    
    # apply parallelisms and initialization
    if parallel_dims.pp_enabled:
        # apply PT-D Pipeline Parallel
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
        # when PP is enabled, `model` obj is no longer used after this point, model_parts is used instead
        del model

        # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
        # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
        # optimizer, and checkpointing
        for m in model_parts:
            # apply SPMD-style PT-D techniques
            train_spec.parallelize_fn(m, world_mesh, parallel_dims, job_config)
            m.to_empty(device=init_device)
            with torch.no_grad():
                m.post_init()
            m.train()

        # confirm that user will be able to view loss metrics on the console
        ensure_pp_loss_visible(parallel_dims, job_config, color)
    else:
        # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
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

    if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "embeddings"):
        model.model.model.embeddings.weight.requires_grad_(False)

    if hasattr(model, "model") and hasattr(model.model, "lm_head"):
        for p in model.model.lm_head.parameters():
            p.requires_grad_(False)

    # If your wrapper exposes lm_head directly:
    if hasattr(model, "lm_head"):
        for p in model.lm_head.parameters():
            p.requires_grad_(False)
    # build optimizer after applying parallelisms to the model
    optimizers = train_spec.build_optimizers_fn(model_parts, job_config, ft_manager)
    lr_schedulers = train_spec.build_lr_schedulers_fn(optimizers, job_config)

    #debug_embeddings_optimizer_state(model, optimizers, tag="after_optimizer_build")
    #print(ttttt)
    # Post optimizer step model converters hook.
    # e.g. calculate float8 dynamic amax/scale for all-parameter for FSDP2
    # where it issues a single all-reduce for all parameters at once for better performance
    optimizers.register_step_post_hook(
        lambda *args, **kwargs: model_converters.post_optimizer_hook(model_parts)
    )

    train_state = TrainState()

    # load initial checkpoint
    checkpoint = CheckpointManager(
        dataloader=train_loader,
        model_parts=model_parts,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"train_state": train_state},
        job_config=job_config,
        ft_manager=ft_manager,
    )
    #print(f"checkpoint load step ----------------- : {job_config.checkpoint.load_step}, {job_config.checkpoint.enable_checkpoint}")
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

    # Only reinitialize if NOT loading from a checkpoint
    #if job_config.checkpoint.load_step == -1:
    with torch.no_grad():
        # Reinit proj
        torch.nn.init.xavier_uniform_(model.proj.weight)
        torch.nn.init.zeros_(model.proj.bias)

        # Reinit forecast_head too (good idea)
        torch.nn.init.xavier_uniform_(model.forecast_head.weight)
        torch.nn.init.zeros_(model.forecast_head.bias)
    
    keys = ["proj.weight", "forecast_head.weight", "model.layers.0.attn.q_proj.weight"]
    pre = {}
    for k in keys:
        pre[k] = checksum_named_param_local(model, k)

    dist.barrier()  # keep ranks aligned
    if dist.get_rank() == 0:
        for k, (nme, h) in pre.items():
            print(f"[DEBUG:pre_load] {k} -> {nme} hash={h}", flush=True)
    
    checkpoint.load(step=job_config.checkpoint.load_step)
    
    dist.barrier()

    p = model.forecast_head.weight
    has_state = (p in optimizers.state) and (len(optimizers.state[p]) > 0)
    if dist.get_rank()==0:
        print("optimizer has forecast_head state:", has_state, optimizers.state.get(p, {}).keys())

    # --- POST ---
    post = {}
    for k in keys:
        post[k] = checksum_named_param_local(model, k)

    dist.barrier()
    if dist.get_rank() == 0:
        for k, (nme, h) in post.items():
            print(f"[DEBUG:post_load] {k} -> {nme} hash={h}", flush=True)
    
    metric_logger = build_metrics_processor(job_config, parallel_dims)
    # Set dependent attributes for metric_logger
    metric_logger.num_flops_per_token = num_flops_per_token
    metric_logger.optimizers = optimizers  # Pass optimizers if needed by logger logic
    metric_logger.lr_schedulers = (
        lr_schedulers  # Pass schedulers if needed by logger logic
    )

    # plot losses loaded from checkpoint (if any) to TensorBoard
    # NOTE: Loss info after the last log step before checkpoint saving will not be ploted.
    #       This can be avoided by setting checkpoint.interval to be a multiple of metrics.log_freq
    if train_state.step > 0 and len(metric_logger.data_loading_times) > 0:
        for idx, step in enumerate(train_state.log_steps):
            metric_logger.log(
                step,
                global_avg_loss=train_state.global_avg_losses[idx],
                global_max_loss=train_state.global_max_losses[idx],
            )

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

    # variables used to keep info for metrics logging
    device_memory_monitor.reset_peak_stats()

    global_batch_size = (
        job_config.training.batch_size
        * dp_degree
        * job_config.training.gradient_accumulation_steps
    )
    num_tokens_per_step = global_batch_size * job_config.training.seq_len
    # train loop
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
   
    """
    w = model.proj.weight
    b = model.proj.bias
    print("proj weight device:", w.device, "is_meta:", w.is_meta)
    print("proj bias  device:", b.device, "is_meta:", b.is_meta)

    if not w.is_meta:
        print("proj weight stats:", w.mean().item(), w.std().item())
        print("proj bias  stats:", b.mean().item(), b.std().item())
    
    #for name, p in model.named_parameters():
    #    if "proj" in name or "forecast_head" in name:
    #        print("param", name, "requires_grad:", p.requires_grad)
    """
    epoch = 0
    with (
        maybe_enable_profiling(
            job_config, global_step=train_state.step
        ) as torch_profiler,
        maybe_enable_memory_snapshot(
            job_config, global_step=train_state.step
        ) as memory_profiler,
    ):
        while train_state.step < job_config.training.steps:
            train_state.step += 1
            gc_handler.run(train_state.step)

            optimizers.zero_grad()

            losses = []
            # do gradient accumulation if enabled
            for _ in range(job_config.training.gradient_accumulation_steps):
                # get batch
                data_load_start = time.perf_counter()
                try:
                    batch = next(train_data_iterator)
                except StopIteration:
                    epoch += 1
                    if train_sampler is not None:
                        train_sampler.set_epoch(epoch)  # reshuffle each epoch across ranks
                    train_data_iterator = iter(train_loader)
                    batch = next(train_data_iterator)
                inputs, targets = batch["inputs"], batch["targets"]

                # Update metrics processor state before forward/backward
                metric_logger.ntokens_since_last_log += targets.numel()
                metric_logger.data_loading_times.append(
                    time.perf_counter() - data_load_start
                )

                inputs = inputs.to(device_type)

                """
                TODO[flame]: We need to carefully handle the position_ids for TP/CP
                Depending on the Models'PE, the position_ids might be different.

                e.g. for TP
                    For RoPE, all ranks have the same position_ids. [FOR HF model]
                    For sinusoidal, each rank has the coresponding chunked  position_ids. [FOR HF model]

                e.g. for CP, [optional_context_parallel_ctx shoudl automatically distbute the position_ids]
                    Each rank has the coresponding chunked position_ids. [FOR All model]

                """
                targets = targets.to(device_type)
                cu_seqlens = (
                    batch["cu_seqlens"].to(device_type)
                    if "cu_seqlens" in batch
                    else None
                )
                if cu_seqlens is not None:
                    position_ids = prepare_position_ids(cu_seqlens).to(torch.int32)
                else:
                    position_ids = (
                        torch.arange(0, inputs.shape[1], device=device_type)
                        .repeat(inputs.shape[0], 1)
                        .to(torch.int32)
                    )
                # apply context parallelism if cp is enabled
                # ensure CP handles the separate freqs_cis buffer for each pp stage
                optional_context_parallel_ctx = (
                    dist_utils.create_context_parallel_ctx(
                        cp_mesh=world_mesh["cp"],
                        cp_buffers=[inputs, targets, position_ids],
                        cp_seq_dims=[1, 1, 1],
                        cp_no_restore_buffers={inputs, targets, position_ids},
                        cp_rotate_method=job_config.experimental.context_parallel_rotate_method,
                    )
                    if parallel_dims.cp_enabled
                    else None
                )

                # #! TODO[flame], we should distribute the position_ids as well with CP
                if parallel_dims.pp_enabled:
                    raise NotImplementedError(
                        "Pipeline parallelism is not supported in this version"
                    )
                    # Pipeline Parallel forward / backward inside step() call
                    with train_context(optional_context_parallel_ctx):
                        targets, losses = (
                            (labels, []) if has_last_stage else (None, None)
                        )

                        if has_first_stage:
                            pp_schedule.step(inputs, target=targets, losses=losses)
                        else:
                            pp_schedule.step(target=targets, losses=losses)

                    # accumulate losses across pipeline microbatches
                    # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
                    loss = (
                        torch.mean(torch.stack(losses)).to(device)
                        if has_last_stage
                        else torch.tensor([-1.0], device=device)
                    )
                else:
                    # Non-PP forward / backward
                    with train_context(optional_context_parallel_ctx):
                        with maybe_enable_amp:
                            #logger.info(f"{color.red}  get the o/p ******** {color.reset}")
                            output = model(
                                inputs=inputs,
                                labels=targets,
                                position_ids=position_ids,
                                cu_seqlens=cu_seqlens,
                        )
                        #logger.info(f"{color.red}  got the o/p ******** {color.reset}")
                        loss = (
                            output.loss
                            / job_config.training.gradient_accumulation_steps
                        )
                        loss.backward()

                losses.append(loss)
            loss = sum(losses) # this is for the summing of losses over the gradient accum - but this does not matter 
            with torch.no_grad():
                if train_state.step % 500 == 0: 
                    print("labels stats:",
                        targets.mean().item(),
                        targets.std().item(),
                        targets.min().item(),
                        targets.max().item(),
                        )
                    print("pred stats:",
                        output.logits.mean().item(),
                        output.logits.std().item(),
                        output.logits.min().item(),
                        output.logits.max().item(),
                        )
                    diff = output.logits - targets
                    const = targets.mean()
                    baseline_mse = ((targets - const) ** 2).mean().item()
                    baseline_rmse = baseline_mse ** 0.5
                    logger.info(f"{color.red} rmse: {torch.sqrt((diff ** 2).mean()).item()} at trainstep : {train_state.step} ")
                    logger.info(f"{color.red} mean-baseline RMSE: {baseline_rmse} at trainstep : {train_state.step} ")
            
                if train_state.step % 250 == 0:
                    f_path = job_config.job.dump_folder + "/plots/train_step_" + str(train_state.step) + ".png"
                    with torch.no_grad():
                        dataset = train_loader.dataset
                        pred_phys = dataset.denormalize(output.logits)
                        truth_phys = dataset.denormalize(targets)
                    
                        pred_phys = pred_phys.detach().float().cpu()[:,0,0,:,:]
                        truth_phys = truth_phys.detach().float().cpu()[:,0,0,:,:]
                        plot_geopotential_comparison(model_output=pred_phys,truth=truth_phys,lat=lat,lon=lon,timesteps=[0, 4, 8, 12],  # Plot specific timesteps
                                                         save_path=f_path,  # Set to filename to save
                                                         vmin=vmin,
                                                         vmax=vmax)

            # clip gradients
            grad_norm = dist_utils.clip_grad_norm_(
                [p for m in model_parts for p in m.parameters()],
                job_config.training.max_norm,
                foreach=True,
                pp_mesh=pp_mesh if parallel_dims.pp_enabled else None,
            )

            # optimizer step
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

            # log metrics - Use MetricsProcessor
            if metric_logger.should_log(train_state.step):
                #print(f"enabled : { parallel_dims.dp_replicate_enabled}, {parallel_dims.dp_shard_enabled,} {parallel_dims.cp_enabled}")
                if (
                    parallel_dims.dp_replicate_enabled
                    or parallel_dims.dp_shard_enabled
                    or parallel_dims.cp_enabled
                ):
                    loss = loss.detach()
                    #print(f"WM:{world_mesh["dp_cp"]}, loss:{loss}")
                    
                    # Use dist_mean/max on the accumulated loss for the step
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
                    #print(f"global_avg_loss:{global_avg_loss}, global_max_loss:{global_max_loss}")
                else:
                    # Scale back the loss before logging
                    global_avg_loss = global_max_loss = loss.item()
                # Update train state tokens and elapsed time
                time_now = time.perf_counter()
                time_delta = (
                    time_now - metric_logger.time_last_log
                )  # Use metric_logger's time
                train_state.token += (
                    metric_logger.ntokens_since_last_log  # Use tokens tracked by metric_logger
                    * parallel_dims.world_size
                    / parallel_dims.non_data_parallel_size
                )
                train_state.elapsed += timedelta(seconds=time_delta)
                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                # Log using the metric processor
                last_lr = lr_schedulers.schedulers[0].get_last_lr()[0]
                eta = (
                    train_state.elapsed
                    * (job_config.training.steps - train_state.step)
                    / train_state.step
                )
                extra_metrics={
                        "optimizer/lr": last_lr,
                        "optimizer/grad_norm": grad_norm.item(),
                        "optimizer/skipped_step": train_state.skipped_step,
                    }
                if train_state.step % 10 == 0: # add job_config.training.eval_interval == 0: and job_config.training.eval_max_batches
                    if train_state.step % 500 == 0:
                        plot=True,
                        plot_path=job_config.job.dump_folder + "/plots/val_step_" + str(train_state.step) + ".png"
                    else:
                        plot=False
                        plot_path = None
                    val_loss = evaluate(model, val_loader, plot, plot_path, device,lat, lon, vmin, vmax, max_batches=2)
                    extra_metrics["loss_metrics/val_avg_loss"] = val_loss
                    if int(rank) == 0:
                        logger.info(f"[eval] step {train_state.step} val_loss={val_loss:.6f}")
                
                metric_logger.log(
                    train_state.step,
                    global_avg_loss,
                    global_max_loss,
                    extra_metrics=extra_metrics,
                )

                logger.info(
                    f"{color.blue}lr: {last_lr:.4e} gnorm: {grad_norm:5.2f} loss:{loss.item()}"
                    f"{color.magenta}[{str(train_state.elapsed).split('.')[0]:>8}<{str(eta).split('.')[0]:>8}]{color.reset}"
                )
            #logger.info(f"calling checkpoint save at train_state.step:{train_state.step},steps: {job_config.training.steps}")
            #pdb.set_trace()
            checkpoint.save(
                train_state.step, force=(train_state.step == job_config.training.steps)
            )
            # if _rank0():
            #     print("[DEBUG] model checksum before load:", model_checksum(model), flush=True)
    

            # signal the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

            # reduce timeout after first train step for faster signal
            # (assuming lazy init and compilation are finished)
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
    if rank > 0:
        logger.setLevel(logging.WARNING)
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
    config = JobConfig()
    config.parse_args()
    main(config)
    torch.distributed.destroy_process_group()
