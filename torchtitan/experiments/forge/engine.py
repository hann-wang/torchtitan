# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections.abc import Generator
from dataclasses import asdict, dataclass, field
from typing import Any, cast

import torch
from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.loss import LossFunction, ChunkedLossWrapper
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.observability import structured_logger as sl
from torchtitan.config import Configurable, TORCH_DTYPE_MAP
from torchtitan.config.configs import (
    CommConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.config.override import apply_overrides, OverrideConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.distributed.activation_checkpoint import (
    ActivationCheckpointingConfig,
    MemoryBudgetAC,
    SelectiveAC,
)
from torchtitan.protocols import BaseModel
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools import utils


class ForgeEngine(torch.distributed.checkpoint.stateful.Stateful, Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        hf_assets_path: str = "./tests/assets/tokenizer"
        dump_folder: str = "./outputs"
        model_spec: ModelSpec = field(default_factory=ModelSpec)
        optimizer: OptimizersContainer.Config = field(
            default_factory=OptimizersContainer.Config
        )
        lr_scheduler: LRSchedulersContainer.Config = field(
            default_factory=LRSchedulersContainer.Config
        )
        training: TrainingConfig = field(default_factory=TrainingConfig)
        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        checkpoint: CheckpointManager.Config = field(
            default_factory=CheckpointManager.Config
        )
        activation_checkpoint: ActivationCheckpointingConfig = field(
            default_factory=SelectiveAC.Config
        )
        compile: CompileConfig = field(default_factory=CompileConfig)
        comm: CommConfig = field(default_factory=CommConfig)
        debug: DebugConfig = field(default_factory=DebugConfig)

        def __post_init__(self):
            if isinstance(self.activation_checkpoint, MemoryBudgetAC.Config) and not (
                self.compile.enable and "model" in self.compile.components
            ):
                raise ValueError(
                    "Memory budget activation checkpointing requires the model to be "
                    "compiled: set --compile.enable and include 'model' in "
                    "--compile.components."
                )

        def to_dict(self) -> dict[str, Any]:
            return asdict(self)

    # core configs
    config: Config
    parallel_dims: ParallelDims
    train_spec: ModelSpec

    # swappable training components in ModelSpec
    model_parts: list[torch.nn.Module]
    loss_fn: LossFunction
    optimizers: OptimizersContainer
    lr_schedulers: LRSchedulersContainer

    # non-swappable training components
    checkpointer: CheckpointManager

    # runtime utilities
    device: torch.device
    gc_handler: utils.GarbageCollection
    gradient_accumulation_steps: int
    train_context: Generator[None, None, None]
    pp_has_first_stage: bool
    pp_has_last_stage: bool

    # Fields in ForgeEngine which are not in original Trainer
    # for dataloading
    batch_degree: int
    batch_rank: int
    # for logging
    model_config: BaseModel.Config
    num_flops_per_token: float
    model_param_count: int
    global_batch_size: int

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    @record
    def __init__(self, config: Config):
        torch._C._log_api_usage_once("torchtitan.train")

        self.config = config
        assert (
            config.model_spec is not None
        ), "model_spec must be set before creating Trainer"
        self.model_spec = config.model_spec

        device_module, device_type = utils.device_module, utils.device_type
        # pyrefly: ignore [read-only]
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(self.device)

        # init distributed and build meshes
        self.parallel_dims = parallel_dims = self.init_distributed()

        # validate dense activation sequence length evenness
        seq_len_divisor = (
            parallel_dims.tp if config.parallelism.enable_sequence_parallel else 1
        ) * (2 * parallel_dims.cp if parallel_dims.cp > 1 else 1)
        if config.training.seq_len % seq_len_divisor != 0:
            raise ValueError(
                f"Training sequence length ({config.training.seq_len}) must be "
                f"divisible by {seq_len_divisor} for the configured "
                "sequence/context parallelism."
            )

        # TODO(pianpwk): Transitional until the local-SPMD and full-DTensor
        # backends share one runtime mesh/type mechanism.
        dist_utils.set_spmd_backend(config.parallelism.spmd_backend)

        # Logging needs to happen after distributed initialized
        config.maybe_log()

        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            batch_degree, batch_rank = batch_mesh.size(), batch_mesh.get_local_rank()
        else:
            batch_degree, batch_rank = 1, 0
        self.batch_degree, self.batch_rank = batch_degree, batch_rank

        # take control of garbage collection to avoid stragglers
        self.gc_handler = utils.GarbageCollection(
            gc_freq=config.training.gc_freq, debug=config.training.gc_debug
        )

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        dist_utils.set_determinism(
            parallel_dims,
            self.device,
            config.debug,
            distinct_seed_mesh_dims=["pp"],  # same as `torchtitan/train.py`
        )

        # build model (using meta init)
        self.model_config = model_config = self.model_spec.model

        # Build the collection of model converters. No-op if converters empty
        model_compile_enabled = (
            config.compile.enable and "model" in config.compile.components
        )
        self.model_converters = config.model_converters.build(
            parallel_dims=parallel_dims,
            model_compile_enabled=model_compile_enabled,
        )

        # set the model args from training configs
        model_config.update_from_config(
            config=config,
        )

        # convert configs
        self.model_converters.convert_config(model_config)

        # Apply overrides to the full config tree, before any component is
        # built. The model config is reached via ModelSpec.traverse. Model
        # overrides must run after update_from_config above (it sets sharding
        # config on the pre-override modules); all other components (optimizer,
        # loss, dataloader, …) are built later in __init__.
        if config.override.imports:
            apply_overrides(config.override, config)

        with (
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]),
        ):
            model = model_config.build()

        self.model_converters.convert(model)

        # # Verify all submodules satisfy the Module protocol
        # # TODO: move this to module validate().
        # # This is current put here to verify module build and
        # # converter, which should guanrantee Module protocol.
        # # On the other hand, some parallelism wrappers don't
        # # have this guanrantee, e.g., fully_shard.
        # model.verify_module_protocol()

        # calculate model size and flops per token
        (
            self.model_param_count,
            self.num_flops_per_token,
        ) = model_config.get_nparams_and_flops(model, config.training.seq_len)
        
        # move sharded model to CPU/GPU and initialize weights via DTensor
        buffer_device: torch.device | None
        if config.checkpoint.create_seed_checkpoint:
            init_device = "cpu"
            buffer_device = None
        elif config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = torch.device(device_type)
        else:
            init_device = device_type
            buffer_device = None

        self.loss_fn = config.loss.build(
            compile_config=config.compile,
        )

        # verify batch sizes
        global_batch_size = config.training.global_batch_size
        if global_batch_size < 0:
            # This global batch size results in 1 gradient accumulation
            # step.
            global_batch_size = config.training.local_batch_size * batch_degree
        assert global_batch_size > 0
        assert (
            global_batch_size % (config.training.local_batch_size * batch_degree) == 0
        ), (
            f"global batch size must be multiple of local batch size times "
            f"data-parallel degree ({global_batch_size} "
            f"% ({config.training.local_batch_size} * {batch_degree}) != 0)"
        )
        self.global_batch_size = global_batch_size

        # calculate gradient accumulation steps
        self.gradient_accumulation_steps = global_batch_size // (
            config.training.local_batch_size * batch_degree
        )
        assert self.gradient_accumulation_steps > 0

        with sl.log_trace_span("model_parallelism_init"):
            # apply parallelisms and initialization
            if parallel_dims.pp_enabled:
                if not self.model_spec.pipelining_fn:
                    raise RuntimeError(
                        f"Pipeline Parallel is enabled but {self.model_spec.name} "
                        f"does not support pipelining"
                    )

                # apply both PT-D Pipeline Parallel and SPMD-style PT-D techniques
                (
                    self.pp_schedule,
                    self.model_parts,
                    self.pp_has_first_stage,
                    self.pp_has_last_stage,
                ) = self.model_spec.pipelining_fn(
                    model,
                    parallel_dims=parallel_dims,
                    training=config.training,
                    parallelism=config.parallelism,
                    compile_config=config.compile,
                    ac_config=config.activation_checkpoint,
                    dump_folder=config.dump_folder,
                    device=self.device,
                    model_config=model_config,
                    parallelize_fn=self.model_spec.parallelize_fn,
                    loss_fn=self.loss_fn,
                )
                # when PP is enabled, `model` obj is no longer used after this point,
                # model_parts is used instead
                del model

                for m in self.model_parts:
                    m.to_empty(device=init_device)
                    with torch.no_grad():
                        # TODO: Change this back to init_weights once
                        # autoparallel contains the wrap_init_states
                        cast(BaseModel, m).init_weights(buffer_device=buffer_device)
                    m.train()
            else:
                if not config.checkpoint.create_seed_checkpoint:
                    # Skip parallelize_fn for seed checkpoints -- nothing from
                    # it is needed (AC, compile, nD parallelism, mixed precision, etc.).
                    model = self.model_spec.parallelize_fn(
                        model,
                        parallel_dims=parallel_dims,
                        training=config.training,
                        parallelism=config.parallelism,
                        compile_config=config.compile,
                        ac_config=config.activation_checkpoint,
                        dump_folder=config.dump_folder,
                    )

                model.to_empty(device=init_device)
                with torch.no_grad():
                    # TODO: Change this back to init_weights once
                    # autoparallel contains the wrap_init_states
                    cast(BaseModel, model).init_weights(buffer_device=buffer_device)
                model.train()

                self.model_parts = [model]

        # Set lm_head reference for ChunkedLossWrapper after model construction.
        # Non-PP: single model part always has lm_head.
        # PP: only the last stage has lm_head; non-last stages skip this.
        if isinstance(self.loss_fn, ChunkedLossWrapper):
            if parallel_dims.pp_enabled:
                if self.pp_has_last_stage:
                    lm_head = self.model_parts[-1].lm_head
                    assert (
                        lm_head is not None
                    ), "Last PP stage must have lm_head for ChunkedLossWrapper"
                    self.loss_fn.set_lm_head(
                        lm_head  # pyrefly: ignore[bad-argument-type]
                    )
                    self.model_parts[
                        -1
                    ]._skip_lm_head = True  # pyrefly: ignore[bad-argument-type]
            else:
                assert len(self.model_parts) == 1
                lm_head = self.model_parts[0].lm_head
                assert (
                    lm_head is not None
                ), "Model must have lm_head for ChunkedLossWrapper"
                self.loss_fn.set_lm_head(lm_head)  # pyrefly: ignore[bad-argument-type]
                self.model_parts[
                    0
                ]._skip_lm_head = True  # pyrefly: ignore[bad-argument-type]

        self.model_converters.post_initialization(self.model_parts)

        # build optimizer after applying parallelisms to the model
        self.optimizers = config.optimizer.build(
            model_parts=self.model_parts,
        )
        if self.model_spec.post_optimizer_build_fn is not None:
            self.model_spec.post_optimizer_build_fn(
                self.optimizers, self.model_parts, parallel_dims
            )
        self.lr_schedulers = config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=config.training.steps,
        )
        # Post optimizer step model converters hook.
        self.optimizers.register_step_post_hook(
            lambda *args, **kwargs: self.model_converters.post_optimizer_hook(
                self.model_parts
            )
        )

        self.train_context = dist_utils.get_spmd_context(
            parallel_dims=parallel_dims,
            spmd_typechecking=(
                config.parallelism.spmd_backend == "spmd_types"
                and config.debug.spmd_typechecking
            ),
        )

    @sl.log_trace_span("torch_distributed_init")
    def init_distributed(self) -> ParallelDims:
        config = self.config
        world_size = dist_utils.init_distributed(
            config.comm,
            enable_cpu_backend=config.training.enable_cpu_offload,
            base_folder=config.dump_folder,
        )

        return ParallelDims.from_config(config.parallelism, world_size)

    def build_checkpointer(
        self,
        dataloader: BaseDataLoader | None,
    ) -> CheckpointManager:
        config = self.config
        return config.checkpoint.build(
            dataloader=dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            sd_adapter=(
                self.model_spec.state_dict_adapter(
                    self.model_config, config.hf_assets_path
                )
                if self.model_spec.state_dict_adapter
                else None
            ),
            base_folder=config.dump_folder,
        )

    def close(self) -> None:
        if hasattr(self, "checkpointer") and self.checkpointer:
            self.checkpointer.close()
