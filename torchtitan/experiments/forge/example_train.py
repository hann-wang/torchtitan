# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections.abc import Iterable
from datetime import timedelta
from typing import Any, cast
import math

import spmd_types as spmd
import torch
from torch.distributed.elastic.multiprocessing.errors import record
from torchtitan.observability import structured_logger as sl
from torchtitan.components.dataloader import BaseDataLoader, DataloaderExhaustedError
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.components.validate import Validator
from torchtitan.config import ConfigManager
from torchtitan.distributed import full_dtensor, utils as dist_utils
from torchtitan.distributed.context_parallel import prepare_context_parallel_input
from torchtitan.distributed.spmd_types import annotate_input_spmd_types
from torchtitan.models.common.attention import FlexAttention, VarlenAttention
from torchtitan.models.common.decoder import Decoder
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.trainer import Trainer as TitanTrainer

from .engine import ForgeEngine


class Trainer(ForgeEngine):
    tokenizer: HuggingFaceTokenizer | None
    dataloader: BaseDataLoader
    validator: Validator
    metrics_processor: MetricsProcessor

    # additional training states
    step: int

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    @record
    def __init__(self, config: TitanTrainer.Config):
        if config.debug.print_config:
            logger.info(f"Running with args: {config.to_dict()}")

        # NOTE: Here we are passing in Trainer.Config as a superset of ForgeEngine.Config
        super().__init__(config)

        # build tokenizer
        self.tokenizer = (
            config.tokenizer.build(tokenizer_path=config.hf_assets_path)
            if config.tokenizer is not None
            else None
        )

        # build dataloader
        self.dataloader = config.dataloader.build(
            dp_world_size=self.batch_degree,
            dp_rank=self.batch_rank,
            tokenizer=self.tokenizer,
            seq_len=config.training.seq_len,
            local_batch_size=config.training.local_batch_size,
            snapshot_every_n_steps=(
                config.checkpoint.interval * self.gradient_accumulation_steps
                if config.checkpoint.enable
                else None
            ),
        )

        model_args = self.model_config
        logger.info(f"Built {config.model_spec.name} {config.model_spec.flavor}")

        # metrics logging
        self.metrics_processor = config.metrics.build(
            parallel_dims=self.parallel_dims,
            dump_folder=config.dump_folder,
            pp_schedule=config.parallelism.pipeline_parallel_schedule,
            config_dict=config.to_dict(),
        )
        color = self.metrics_processor.color

        self.metrics_processor.num_flops_per_token = self.num_flops_per_token

        logger.info(
            f"{color.blue}Model {config.model_spec.name} {config.model_spec.flavor} "
            f"{color.red}size: {self.model_param_count:,} total parameters{color.reset}"
        )

        # initialize device memory monitor and get peak flops for MFU calculation
        device_memory_monitor = self.metrics_processor.device_memory_monitor
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
        logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
        device_mem_stats = device_memory_monitor.get_peak_stats()
        logger.info(
            f"{utils.device_type.upper()} memory usage for model: "
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )

        self.metrics_processor.optimizers = self.optimizers
        self.metrics_processor.model_parts = self.model_parts

        # Initialize trainer states that will be saved in checkpoint.
        # These attributes must be initialized before checkpoint loading.
        self.step = 0
        self.ntokens_seen = 0

        # Build validator if validation is configured
        if config.validator.enable:
            pp_schedule, pp_has_first_stage, pp_has_last_stage = (
                (
                    self.pp_schedule,
                    self.pp_has_first_stage,
                    self.pp_has_last_stage,
                )
                if self.parallel_dims.pp_enabled
                else (None, None, None)
            )

            self.validator = config.validator.build(
                parallelism=config.parallelism,
                dp_world_size=self.batch_degree,
                dp_rank=self.batch_rank,
                tokenizer=self.tokenizer,
                parallel_dims=self.parallel_dims,
                loss_fn=self.loss_fn,
                validation_context=self.train_context,
                metrics_processor=self.metrics_processor,
                seq_len=config.training.seq_len,
                local_batch_size=config.training.local_batch_size,
                pp_schedule=pp_schedule,
                pp_has_first_stage=pp_has_first_stage,
                pp_has_last_stage=pp_has_last_stage,
            )

        self.profiler = config.profiler.build()

        logger.info(
            "Trainer is initialized with "
            f"local batch size {config.training.local_batch_size}, "
            f"global batch size {self.global_batch_size}, "
            f"gradient accumulation steps {self.gradient_accumulation_steps}, "
            f"sequence length {config.training.seq_len}, "
            f"total steps {config.training.steps} "
            f"(warmup {config.lr_scheduler.warmup_steps})."
        )

    def batch_generator(
        self, data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """Returns an iterator that processes batches from the data iterator."""
        data_iterator = iter(data_iterable)
        
        while True:
            data_load_start = time.perf_counter()
            try:
                batch = next(data_iterator)
            except StopIteration as ex:
                # If data runs out during gradient accumulation, that
                # entire step will not be executed.
                raise DataloaderExhaustedError() from ex
            input_dict, labels = batch
            ntokens_batch = labels.numel()
            self.metrics_processor.ntokens_since_last_log += ntokens_batch
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )

            # Tensors stay on CPU; moved to GPU per-microbatch during training
            yield input_dict, labels

    @sl.log_trace_span("post_dataloading_process")
    def post_dataloading_process(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        inputs = input_dict["input"]
        # Everything else becomes a model-forward kwarg, forwarded to all PP
        # stages by the schedule. positions is read here so we can build masks.
        extra_kwargs: dict[str, Any] = {
            k: v for k, v in input_dict.items() if k != "input"
        }

        positions = extra_kwargs.get("positions", None)
        
        # positions and attention_masks are optional (Decoder.forward defaults
        # both to None). Build attention masks only for the masked backends
        # (Flex/Varlen), which is where get_attention_masks is defined. A
        # maskless backend (e.g. the SDPA config used by the graph_trainer
        # tests) still receives positions for RoPE but no masks — it relies on
        # is_causal instead.
        if isinstance(self.model_config, Decoder.Config) and positions is not None:
            inner_attention = getattr(
                self.model_config.first_attention, "inner_attention", None
            )
            if isinstance(
                inner_attention, (FlexAttention.Config, VarlenAttention.Config)
            ):
                model = cast(Decoder, self.model_parts[0])
                extra_kwargs["attention_masks"] = model.get_attention_masks(
                    positions=positions,
                )

        if self.parallel_dims.cp_enabled:
            inputs, labels, extra_kwargs = prepare_context_parallel_input(
                inputs,
                labels,
                extra_kwargs,
                self.parallel_dims.get_mesh("cp"),
                self.device,
                self.config.parallelism.context_parallel_load_balancer,
            )
            
        # Accumulate after CP sharding so labels.numel() reflects the actual
        # unique tokens this rank processes (not the full pre-split sequence).
        self.ntokens_seen += labels.numel()
        
        if self.config.parallelism.spmd_backend == "full_dtensor":
            inputs, labels, extra_kwargs = full_dtensor.parallelize_inputs(
                self.parallel_dims, inputs, labels, extra_kwargs
            )
        elif self.config.parallelism.spmd_backend == "spmd_types":
            inputs, labels, extra_kwargs = annotate_input_spmd_types(
                self.parallel_dims,
                inputs,
                labels,
                extra_kwargs,
            )

        return inputs, labels, extra_kwargs

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: float,
    ) -> torch.Tensor:
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        inputs, labels, extra_kwargs = self.post_dataloading_process(input_dict, labels)

        if parallel_dims.pp_enabled:
            # Pipeline Parallel forward / backward inside step() call
            loss_kwargs = {"global_valid_tokens": global_valid_tokens}
            with self.train_context():
                targets, losses = (
                    (labels, []) if self.pp_has_last_stage else (None, None)
                )
                if self.pp_has_first_stage:
                    self.pp_schedule.step(
                        inputs,
                        **extra_kwargs,
                        target=targets,
                        losses=losses,
                        loss_kwargs=loss_kwargs,
                        return_outputs=False,
                    )
                else:
                    self.pp_schedule.step(
                        **extra_kwargs,
                        target=targets,
                        losses=losses,
                        loss_kwargs=loss_kwargs,
                        return_outputs=False,
                    )

            # accumulate losses across pipeline microbatches
            # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
            if self.pp_has_last_stage:
                assert losses is not None
                # All loss classes scale by global_valid_tokens internally
                loss = torch.sum(torch.stack(losses)).to(self.device)
            else:
                loss = torch.tensor([-1.0], device=self.device)
        else:
            # Non-PP forward / backward
            with self.train_context():
                assert len(model_parts) == 1
                pred = model_parts[0](inputs, **extra_kwargs)
                loss, _ = self.loss_fn(pred, labels, global_valid_tokens)
                del pred
                with spmd.no_typecheck():
                    # this propagates types through BWD, causing unnecessary conflicts
                    # between torch_function and internals (e.g. AC). FWD is sufficient.
                    loss.backward()

        # The returned loss here is local SUM loss / global_valid_tokens
        return loss

    def train_step(
        self, data_iterator: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        self.optimizers.zero_grad()
        # Save per-optimizer-group learning rates for logging
        lr_metrics = self.lr_schedulers.get_metrics()

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims

        # Collect all microbatches on CPU and count total valid tokens
        # Here we assume the inputs/labels are on GPU
        microbatches = []
        local_valid_tokens = torch.tensor(0, dtype=torch.int64)
        for _microbatch in range(self.gradient_accumulation_steps):
            with sl.log_trace_span("fetching_batch"):
                input_dict, labels = next(data_iterator)
                local_valid_tokens += (labels != IGNORE_INDEX).sum()
                microbatches.append((input_dict, labels))
        sl.log_trace_scalar({"local_valid_tokens": int(local_valid_tokens)})

        # All-reduce to get global token count across DP ranks
        # Move to GPU for distributed communication
        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            global_valid_tokens = dist_utils.dist_sum(
                local_valid_tokens.to(self.device), batch_mesh
            )
        else:
            global_valid_tokens = float(local_valid_tokens.item())

        # Process each microbatch: move to GPU, forward/backward, then free
        accumulated_losses = []
        for input_dict, labels in microbatches:
            # Move tensors to GPU
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.to(self.device)
            labels = labels.to(self.device)

            loss = self.forward_backward_step(
                input_dict=input_dict,
                labels=labels,
                global_valid_tokens=global_valid_tokens,
            )
            accumulated_losses.append(loss.detach())
        with sl.log_trace_span("optim"):
            grad_norm = dist_utils.clip_grad_norm_(
                [p for m in self.model_parts for p in m.parameters()],
                self.config.training.max_norm,
                foreach=True,
                pp_mesh=parallel_dims.get_optional_mesh("pp"),
                ep_enabled=parallel_dims.ep_enabled,
            )
            self.checkpointer.maybe_wait_for_staging()
            self.optimizers.step()
            self.lr_schedulers.step()

        # Reduce the data collected over gradient accumulation steps.
        loss = torch.sum(torch.stack(accumulated_losses))

        # log metrics
        if not self.metrics_processor.should_log(self.step):
            return
        
        with sl.log_trace_span("collect_dist_metrics"):
            sl.log_trace_scalar({"global_valid_tokens": int(global_valid_tokens)})

            if parallel_dims.dp_cp_enabled:
                loss = loss.detach()
                loss_mesh = parallel_dims.get_optional_mesh("loss")

                # For global_avg_loss, we want the average loss across all ranks:
                # loss = local_loss_sum / global_valid_tokens
                # global_avg_loss = sum(local_loss_sum) / global_valid_tokens
                #                 = sum(loss)
                #
                # For global_max_loss, we want the max of local average losses across ranks:
                # local_avg_loss = local_loss_sum / local_valid_tokens
                #                = (loss * global_valid_tokens) / local_valid_tokens
                # global_max_loss = max(local_avg_loss)
                local_avg_loss = loss * global_valid_tokens / local_valid_tokens
                global_avg_loss, global_max_loss, global_ntokens_seen = (
                    dist_utils.dist_sum(loss, loss_mesh),
                    dist_utils.dist_max(local_avg_loss, loss_mesh),
                    dist_utils.dist_sum(
                        torch.tensor(
                            self.ntokens_seen, dtype=torch.int64, device=self.device
                        ),
                        loss_mesh,
                    ),
                )
            else:
                global_avg_loss = global_max_loss = float(loss.detach().item())
                global_ntokens_seen = self.ntokens_seen
                
        # Crash on invalid loss. global_avg_loss is a SUM reduction, so a infinite
        # loss on any rank propagates here. This reuses the D2H copy already done
        # for logging, so it adds no extra sync.
        # TODO: make this step work even logging is off.
        if not math.isfinite(global_avg_loss):
            raise RuntimeError(
                f"Loss is not finite (global_avg_loss={global_avg_loss}) at "
                f"step {self.step}. Stopping training."
            )
            
        extra_metrics = {
            "n_tokens_seen": global_ntokens_seen,
            **lr_metrics,
        }
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            float(grad_norm.item()),
            extra_metrics=extra_metrics,
        )

    def post_training_tasks(self):
        last_step = not self.should_continue_training()
        if last_step:
            self.model_converters.finalize(self.model_parts)

        self.checkpointer.save(self.step, last_step=last_step)

        # Run validation if validator is available
        if self.config.validator.enable and (
            self.validator.should_validate(self.step) or last_step
        ):
            self.validator.validate(self.model_parts, self.step)

    @record
    def train(self):
        config = self.config
        
        sl.log_trace_instant("training_start")

        self.checkpointer.load(step=config.checkpoint.load_step)

        # Capture loaded step for relative_step calculation.
        # After checkpoint load: self.step = restored step (e.g. 100), or 0 if fresh.
        loaded_step = self.step

        logger.info(f"Training starts at step {self.step + 1}.")

        with self.profiler.active(
            global_step=self.step,
            base_folder=config.dump_folder,
        ) as profiler:
            data_iterator = self.batch_generator(self.dataloader)
            while self.should_continue_training():
                self.step += 1
                sl.set_step(self.step, relative_step=self.step - loaded_step)
                with sl.log_trace_span("step"):
                    self.gc_handler.run(self.step)
                    self.model_converters.pre_step(self.model_parts)
                    try:
                        self.train_step(data_iterator)
                    except DataloaderExhaustedError:
                        logger.warning("Ran out of data; last step was canceled.")
                        break

                    # Save Checkpoint
                    # Run validation if validator is available
                    self.post_training_tasks()

                    # signal the profiler that the next profiling step has started
                    profiler.step()

                    # reduce timeout after first train step for faster signal
                    # (assuming lazy init and compilation are finished)
                    if self.step - loaded_step == 1:
                        dist_utils.set_pg_timeouts(
                            timeout=timedelta(seconds=config.comm.train_timeout_seconds),
                            parallel_dims=self.parallel_dims,
                        )

        if not self.training_enabled():
            # just run validation
            self.post_training_tasks()

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        logger.info("Training completed")

    def training_enabled(self) -> bool:
        return self.step > 0

    def should_continue_training(self) -> bool:
        return self.step < self.config.training.steps

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.step = state_dict["step"]

    def close(self) -> None:
        if self.metrics_processor:
            self.metrics_processor.close()
        super().close()


def main(custom_trainer_class: type[Trainer] | None = None) -> None:
    """Main entry point for training."""
    init_logger()

    import torchtitan

    logger.info(
        "torchtitan version: %s (0.0.0 means __version__ is not defined correctly).",
        torchtitan.__version__,
    )

    config_manager = ConfigManager()
    config = config_manager.parse_args()
    trainer: Trainer | None = None

    try:
        # TODO(local_tensor): Remove this special case once LocalTensor supports
        # init_states() and foreach_allgather. In local tensor mode, skip
        # training/checkpointing as the # model is not fully initialized
        # pyrefly: ignore [missing-attribute]
        if config.comm.mode == "local_tensor":
            logger.info("Local tensor mode enabled - skipping training execution")
            return

        # pyrefly: ignore [missing-attribute]
        if custom_trainer_class is not None:
            trainer = custom_trainer_class(config)
        else:
            trainer = config.build()

        # pyrefly: ignore [missing-attribute]
        if config.checkpoint.create_seed_checkpoint:
            assert (
                int(os.environ["WORLD_SIZE"]) == 1
            ), "Must create seed checkpoint using a single device, to disable sharding."
            assert (
                # pyrefly: ignore [missing-attribute]
                config.checkpoint.enable
            ), "Must enable checkpointing when creating a seed checkpoint."
            trainer.checkpointer.save(curr_step=0, last_step=True)
            logger.info("Created seed checkpoint")
        else:
            trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")


if __name__ == "__main__":
    main(Trainer)
