# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.distributed.tensor.parallel import (
    parallelize_module,
    PrepareModuleInput,
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.config_manager import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims

from torchtitan.models.llama3.parallelize_llama import (
    apply_ac,
    apply_compile,
    apply_ddp,
    apply_fsdp,
)
from torchtitan.tools.logging import logger

from ..model import MoE


def parallelize_deepseek(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """

    if parallel_dims.tp_enabled:
        if (job_config.parallelism.enable_async_tensor_parallel
                and not job_config.training.compile):
            raise RuntimeError("Async TP requires --training.compile")

        enable_float8_linear = "float8" in job_config.model.converters and job_config.float8.enable_fp8_linear
        float8_is_rowwise = job_config.float8.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )
        float8_is_blockwise = job_config.float8.recipe_name in (
            "blockwise",
        )

        # For now, float8 all-gather with TP is only supported for tensorwise
        # float8 scaling recipes. For rowwise recipes, we use regular TP and
        # all-gather happens in high precision.
        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise and not float8_is_blockwise

        apply_tp(
            model,
            world_mesh["tp"],
            loss_parallel=parallel_dims.loss_parallel_enabled,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
            enable_async_tp=job_config.parallelism.
            enable_async_tensor_parallel,
            enable_tp2ep=job_config.parallelism.enable_tp2ep,
        )

    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if job_config.training.compile:
        if job_config.parallelism.enable_tp2ep:
            # TODO: enable fullgraph after https://github.com/pytorch/pytorch/issues/155205 resolved.
            torch._dynamo.disallow_in_graph(
                torch.ops._c10d_functional_autograd.all_to_all_single)
            apply_compile(model, fullgraph=False)
        else:
            apply_compile(model)

        # NOTE: needed for torch.compile to work with dynamic shapes in token-choice MoE
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True

    dp_mesh: DeviceMesh | None = None
    if (parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled
        ):  # apply FSDP or HSDP, potentially with Context Parallel
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp", )
        dp_mesh = world_mesh[tuple(dp_mesh_dim_names)]

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[
                job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[
                job_config.training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training.enable_cpu_offload,
            reshard_after_forward_policy=job_config.parallelism.
            fsdp_reshard_after_forward,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if parallel_dims.cp_enabled:
            logger.info("Applied Context Parallel to the model")

        if job_config.training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")
    elif parallel_dims.dp_replicate_enabled:
        if world_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        dp_mesh = world_mesh
        apply_ddp(
            model,
            dp_mesh,
            enable_compile=job_config.training.compile,
            enable_compiled_autograd=job_config.parallelism.
            enable_compiled_autograd,
        )

    # for MoE auxiliary-loss-free load balancing
    if dp_mesh is not None:
        # NOTE: Currently this sync is blocking (thus exposed) and happens on the
        # default compute stream. Need to assess if this is OK performance-wise.
        def _sync_tokens_per_expert(module, *_):
            assert isinstance(module, MoE)
            torch.distributed.all_reduce(module.tokens_per_expert,
                                         group=dp_mesh.get_group())

        for transformer_block in model.model.layers.values():
            if transformer_block.moe is not None:
                load_balance_coeff = transformer_block.moe.load_balance_coeff
                if load_balance_coeff is not None and load_balance_coeff > 0:
                    # prepend=True so that the sync runs before
                    # the _update_expert_bias hook in MoE
                    transformer_block.moe.register_full_backward_hook(
                        _sync_tokens_per_expert, prepend=True)
                else:
                    break

    return model


def apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
    enable_async_tp: bool,
    enable_tp2ep: bool = True,
):
    """Apply tensor parallelism."""
    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer

    from torchtitan.experiments.llama4.infra.expert_parallel import (
        NoParallel, TensorParallel, ExpertParallel,
        PrepareModuleInputOutputWithParams)

    parallelize_module(
        model,
        tp_mesh,
        {
            "model.tok_embeddings":
            RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "model.norm":
            SequenceParallel(),
            "output":
            ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )

    # Parallel styles used for transformer block linear weights and their
    # inputs may be different for float8 linears with tensorwise scaling.
    if enable_float8_tensorwise_tp:
        # TODO(vkuzo): add the items below to __init__.py of torchao.float8 and import from there
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )

        rowwise_parallel, colwise_parallel, prepare_module_input = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
            PrepareFloat8ModuleInput,
        )
    else:
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )

    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
    layers = model.layers if hasattr(model, "layers") else model.model.layers
    for transformer_block in layers.values():
        layer_plan = {
            "input_layernorm":
            SequenceParallel(),
            "post_attention_layernorm":
            SequenceParallel(),
            "self_attn":
            prepare_module_input(
                input_layouts=(Shard(1), None, None),
                desired_input_layouts=(Replicate(), None, None),
            ),
            "self_attn.q_proj":
            colwise_parallel(),
            "self_attn.q_a_proj":
            NoParallel(),
            "self_attn.q_a_layernorm":
            NoParallel(),
            "self_attn.q_b_proj":
            ColwiseParallel(),
            "self_attn.kv_a_proj_with_mqa":
            NoParallel(),
            "self_attn.kv_a_layernorm":
            NoParallel(),
            "self_attn.kv_b_proj":
            ColwiseParallel(),
            "self_attn.o_proj":
            rowwise_parallel(output_layouts=Shard(1)),
            "feed_forward":
            prepare_module_input(
                input_layouts=(Shard(1), ),
                desired_input_layouts=(Replicate(), ),
            ),
            "feed_forward.gate_proj":
            colwise_parallel(),
            "feed_forward.down_proj":
            rowwise_parallel(output_layouts=Shard(1)),
            "feed_forward.up_proj":
            colwise_parallel(),
            # input / output sharding on the seqlen dim
            # all-gather for input, reduce-scatter for output
            "moe":
            PrepareModuleInputOutputWithParams(
                input_layouts=(Shard(1), ),
                desired_input_layouts=(Replicate(), ),
                use_local_input=True,
                output_layouts=(Partial(), ),
                desired_output_layouts=(Shard(1), ),
            ),
            # replicate computation for the router
            "moe.router.gate":
            NoParallel(),
            "moe.experts":
            TensorParallel(output_layout=Partial()),
            "moe.shared_expert.gate_proj":
            colwise_parallel(),
            "moe.shared_expert.down_proj":
            rowwise_parallel(output_layouts=Partial(), ),
            "moe.shared_expert.up_proj":
            colwise_parallel(),
        }

        if enable_tp2ep:
            layer_plan["moe.experts"] = ExpertParallel()
            layer_plan["moe.shared_expert.down_proj"] = rowwise_parallel(
                output_layouts=Shard(1), )
            layer_plan["moe"] = PrepareModuleInputOutputWithParams(
                input_layouts=(Shard(1), ),
                desired_input_layouts=(Shard(1), ),
                use_local_input=True,
                output_layouts=(Shard(1), ),
                desired_output_layouts=(Shard(1), ),
            )
            layer_plan["moe.shared_expert"] = prepare_module_input(
                input_layouts=(Shard(1), ),
                desired_input_layouts=(Replicate(), ),
            )

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    if enable_async_tp:
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}{'Async ' if enable_async_tp else ''}"
        "Tensor Parallelism to the model")
