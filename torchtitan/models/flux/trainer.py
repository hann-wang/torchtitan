# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import os
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import DefaultLoadPlanner

from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed import utils as dist_utils
from torchtitan.models.flux.configs import Distillation, Encoder, Inference, Validation
from torchtitan.models.flux.model.autoencoder import load_ae
from torchtitan.models.flux.model.hf_embedder import FluxEmbedder
from torchtitan.models.flux.parallelize import parallelize_encoders
from torchtitan.models.flux.utils import (
    create_position_encoding_for_latents,
    pack_latents,
    preprocess_data,
)
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.trainer import Trainer
from torchtitan.tools import utils
from torchtitan.tools.logging import logger


def _make_last_batch_run_key(dump_folder: str) -> str:
    normalized_dump_folder = os.path.abspath(dump_folder)
    folder_name = os.path.basename(normalized_dump_folder) or "flux"
    slug = "".join(c if c.isalnum() else "_" for c in folder_name).strip("_")
    digest = hashlib.sha1(normalized_dump_folder.encode()).hexdigest()[:8]
    return f"{slug or 'flux'}_{digest}"


def _write_last_batch_keys(
    step: int, input_dict: dict[str, object], dump_folder: str
) -> None:
    sample_keys = input_dict.get("sample_key")
    if sample_keys is None:
        return

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    run_key = _make_last_batch_run_key(dump_folder)
    path = f"/tmp/flux_last_batch_{run_key}_rank{rank}.txt"
    tmp_path = f"{path}.tmp"

    try:
        with open(tmp_path, "w") as f:
            f.write(f"step={step}\n")
            f.write(f"rank={rank}\n")
            f.write(f"dump_folder={dump_folder}\n")
            if isinstance(sample_keys, (list, tuple)):
                for key in sample_keys:
                    f.write(f"{key}\n")
            else:
                f.write(f"{sample_keys}\n")
        os.replace(tmp_path, path)
    except OSError:
        # Do not let crash breadcrumbs interfere with training.
        pass


class FluxTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        encoder: Encoder = field(default_factory=Encoder)
        validation: Validation = field(default_factory=Validation)
        inference: Inference = field(default_factory=Inference)
        distillation: Distillation = field(default_factory=Distillation)

    def __init__(self, config: Config):
        super().__init__(config)

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        # For Flux model, we need distinct seed across FSDP ranks to ensure we randomly dropout prompts info in dataloader
        dist_utils.set_determinism(
            self.parallel_dims,
            self.device,
            config.debug,
            distinct_seed_mesh_dims=["fsdp", "dp_replicate"],
        )

        # NOTE: self._dtype is the data type used for encoders (image encoder, T5 text encoder, CLIP text encoder).
        # We cast the encoders and it's input/output to this dtype.  If FSDP with mixed precision training is not used,
        # the dtype for encoders is torch.float32 (default dtype for Flux Model).
        # Otherwise, we use the same dtype as mixed precision training process.
        self._dtype = (
            TORCH_DTYPE_MAP[config.training.mixed_precision_param]
            if self.parallel_dims.dp_shard_enabled
            else torch.float32
        )

        # load components
        assert config.model_spec is not None
        model_args = config.model_spec.model

        self.autoencoder = load_ae(
            # pyrefly: ignore [missing-attribute]
            config.encoder.autoencoder_path,
            # pyrefly: ignore [missing-attribute]
            model_args.autoencoder_params,
            device=self.device,
            dtype=self._dtype,
            # pyrefly: ignore [missing-attribute]
            random_init=config.encoder.test_mode,
        )

        self.clip_encoder = FluxEmbedder(
            # pyrefly: ignore [missing-attribute]
            version=config.encoder.clip_encoder,
            # pyrefly: ignore [missing-attribute]
            random_init=config.encoder.test_mode,
        ).to(device=self.device, dtype=self._dtype)
        self.t5_encoder = FluxEmbedder(
            # pyrefly: ignore [missing-attribute]
            version=config.encoder.t5_encoder,
            # pyrefly: ignore [missing-attribute]
            random_init=config.encoder.test_mode,
        ).to(device=self.device, dtype=self._dtype)

        # Apply FSDP to the T5 model / CLIP model
        # pyrefly: ignore [bad-assignment]
        self.t5_encoder, self.clip_encoder = parallelize_encoders(
            t5_model=self.t5_encoder,
            clip_model=self.clip_encoder,
            parallel_dims=self.parallel_dims,
            training=config.training,
        )

        if config.validator.enable:
            # pyrefly: ignore [missing-attribute]
            self.validator.flux_init(
                device=self.device,
                _dtype=self._dtype,
                autoencoder=self.autoencoder,
                t5_encoder=self.t5_encoder,
                clip_encoder=self.clip_encoder,
                trainer_config=config,
            )

        self.teacher_model = None
        self._last_distillation_step_losses: list[torch.Tensor] = []
        if config.distillation.enable:
            self._initialize_distillation(config)

    def _initialize_distillation(self, config: Config) -> None:
        distill_config = config.distillation
        if distill_config.num_rollout_steps <= 0:
            raise ValueError("distillation.num_rollout_steps must be positive")
        if not distill_config.teacher_checkpoint_path:
            raise ValueError(
                "distillation.teacher_checkpoint_path must be set when distillation is enabled"
            )
        if not distill_config.student_checkpoint_path:
            raise ValueError(
                "distillation.student_checkpoint_path must be set when distillation is enabled"
            )

        self._load_model_only_checkpoint(
            self.model_parts[0],
            distill_config.student_checkpoint_path,
            model_name="student",
        )
        self.teacher_model = self._build_teacher_model(config)
        self._load_model_only_checkpoint(
            self.teacher_model,
            distill_config.teacher_checkpoint_path,
            model_name="teacher",
        )
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        logger.info(
            "Initialized Flux distillation: mode=%s, rollout_steps=%d",
            distill_config.mode,
            distill_config.num_rollout_steps,
        )

    def _build_teacher_model(self, config: Config) -> torch.nn.Module:
        assert config.model_spec is not None
        model_config = deepcopy(self.model_config)
        with (
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]),
        ):
            teacher_model = model_config.build()

        teacher_model = config.model_spec.parallelize_fn(
            teacher_model,
            parallel_dims=self.parallel_dims,
            training=config.training,
            model_converters=ModelConvertersContainer.Config(),
            parallelism=config.parallelism,
            compile_config=config.compile,
            ac_config=config.activation_checkpoint,
            dump_folder=config.dump_folder,
        )
        teacher_model.to_empty(device=self.device)
        return teacher_model

    def _load_model_only_checkpoint(
        self, model: torch.nn.Module, checkpoint_path: str, *, model_name: str
    ) -> None:
        if not os.path.isdir(checkpoint_path):
            raise ValueError(
                f"distillation.{model_name}_checkpoint_path is not a directory: "
                f"{checkpoint_path}"
            )
        logger.info("Loading %s model checkpoint from %s", model_name, checkpoint_path)
        model_wrapper = ModelWrapper([model])
        state_dict = model_wrapper.state_dict()

        if os.path.isfile(os.path.join(checkpoint_path, ".metadata")):
            dcp.load(state_dict, checkpoint_id=checkpoint_path)
            model_wrapper.load_state_dict(state_dict)
            logger.info("Finished loading %s DCP model checkpoint", model_name)
            return

        hf_checkpoint_path = self._resolve_hf_checkpoint_path(checkpoint_path)
        if hf_checkpoint_path is not None:
            self._load_hf_model_checkpoint(
                model_wrapper,
                state_dict,
                hf_checkpoint_path,
                model_name=model_name,
            )
            return

        step_dirs = [
            name
            for name in os.listdir(checkpoint_path)
            if name.startswith("step-") and os.path.isdir(os.path.join(checkpoint_path, name))
        ]
        if step_dirs:
            raise ValueError(
                f"distillation.{model_name}_checkpoint_path points to a checkpoint "
                f"folder, not a concrete DCP step. Use one of the step directories "
                f"under {checkpoint_path}, e.g. {os.path.join(checkpoint_path, step_dirs[0])}"
            )

        raise ValueError(
            f"distillation.{model_name}_checkpoint_path is neither a DCP step "
            f"directory with .metadata nor a HuggingFace safetensors directory: "
            f"{checkpoint_path}"
        )

    def _resolve_hf_checkpoint_path(self, checkpoint_path: str) -> str | None:
        index_files = (
            "model.safetensors.index.json",
            "diffusion_pytorch_model.safetensors.index.json",
        )
        if any(os.path.isfile(os.path.join(checkpoint_path, f)) for f in index_files):
            return checkpoint_path

        for subdir in ("transformer", "transformers"):
            candidate = os.path.join(checkpoint_path, subdir)
            if os.path.isdir(candidate) and any(
                os.path.isfile(os.path.join(candidate, f)) for f in index_files
            ):
                return candidate

        return None

    def _load_hf_model_checkpoint(
        self,
        model_wrapper: ModelWrapper,
        state_dict: dict[str, torch.Tensor],
        checkpoint_path: str,
        *,
        model_name: str,
    ) -> None:
        assert self.config.model_spec is not None
        assert self.config.model_spec.state_dict_adapter is not None

        logger.info(
            "Loading %s HuggingFace safetensors checkpoint from %s",
            model_name,
            checkpoint_path,
        )
        sd_adapter = self.config.model_spec.state_dict_adapter(
            self.model_config,
            checkpoint_path,
        )
        hf_state_dict = sd_adapter.to_hf(state_dict)
        planner = DefaultLoadPlanner(
            flatten_state_dict=True,
            flatten_sharded_tensors=True,
        )
        dcp.load(
            hf_state_dict,
            storage_reader=sd_adapter.get_hf_storage_reader(checkpoint_path),
            planner=planner,
        )
        model_wrapper.load_state_dict(sd_adapter.from_hf(hf_state_dict))
        logger.info("Finished loading %s model checkpoint", model_name)

    def _prepare_flux_training_tensors(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _write_last_batch_keys(self.step, input_dict, self.config.dump_folder)

        input_dict["image"] = labels
        input_dict = preprocess_data(
            device=self.device,
            dtype=self._dtype,
            autoencoder=self.autoencoder,
            clip_encoder=self.clip_encoder,
            t5_encoder=self.t5_encoder,
            batch=input_dict,
        )
        labels = input_dict["img_encodings"]

        local_valid_tokens = torch.tensor(
            labels.numel(), dtype=torch.float32, device=self.device
        )
        if self.parallel_dims.dp_enabled:
            batch_mesh = self.parallel_dims.get_mesh("batch")
            # pyrefly: ignore [bad-assignment]
            global_valid_tokens = dist_utils.dist_sum(local_valid_tokens, batch_mesh)
        else:
            global_valid_tokens = local_valid_tokens.float()

        clip_encodings = input_dict["clip_encodings"]
        t5_encodings = input_dict["t5_encodings"]
        bsz = labels.shape[0]

        with torch.no_grad(), torch.device(self.device):
            noise = torch.randn_like(labels)
            timesteps = torch.rand((bsz,))
            sigmas = timesteps.view(-1, 1, 1, 1)
            latents = (1 - sigmas) * labels + sigmas * noise

        bsz, _, latent_height, latent_width = latents.shape
        position_dim = 3  # constant for Flux flow model
        with torch.no_grad(), torch.device(self.device):
            latent_pos_enc = create_position_encoding_for_latents(
                bsz, latent_height, latent_width, position_dim
            )
            text_pos_enc = torch.zeros(bsz, t5_encodings.shape[1], position_dim)
            latents = pack_latents(latents)
            target = pack_latents(noise - labels)

        if self.parallel_dims.cp_enabled:
            from torchtitan.distributed.context_parallel import cp_shard

            (
                latents,
                latent_pos_enc,
                t5_encodings,
                text_pos_enc,
                target,
            ), _ = cp_shard(
                self.parallel_dims.get_mesh("cp"),
                (latents, latent_pos_enc, t5_encodings, text_pos_enc, target),
                None,
                load_balancer_type=None,
            )

        return {
            "latents": latents,
            "latent_pos_enc": latent_pos_enc,
            "t5_encodings": t5_encodings,
            "text_pos_enc": text_pos_enc,
            "clip_encodings": clip_encodings,
            "timesteps": timesteps,
            "target": target,
            "global_valid_tokens": global_valid_tokens,
        }

    def _flux_forward(
        self,
        model: torch.nn.Module,
        *,
        latents: torch.Tensor,
        latent_pos_enc: torch.Tensor,
        t5_encodings: torch.Tensor,
        text_pos_enc: torch.Tensor,
        clip_encodings: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return model(
            img=latents,
            img_ids=latent_pos_enc,
            txt=t5_encodings,
            txt_ids=text_pos_enc,
            y=clip_encodings,
            timesteps=timesteps,
        )

    def _forward_backward_distillation_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        teacher_model = self.teacher_model
        if teacher_model is None:
            raise RuntimeError(
                "Flux distillation is enabled but teacher model is not initialized"
            )

        tensors = self._prepare_flux_training_tensors(
            input_dict=input_dict,
            labels=labels,
        )
        student_model = self.model_parts[0]
        distill_config = self.config.distillation
        num_steps = distill_config.num_rollout_steps

        student_latents = tensors["latents"]
        teacher_latents = tensors["latents"].detach()
        initial_timesteps = tensors["timesteps"]
        step_size = initial_timesteps / num_steps

        total_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        step_losses: list[torch.Tensor] = []
        with self.train_context():
            for rollout_step in range(num_steps):
                current_timesteps = torch.clamp(
                    initial_timesteps - step_size * rollout_step, min=0.0
                )
                next_timesteps = torch.clamp(
                    initial_timesteps - step_size * (rollout_step + 1), min=0.0
                )
                delta = (next_timesteps - current_timesteps).view(-1, 1, 1)

                if distill_config.mode == "off_policy":
                    teacher_input = teacher_latents
                else:
                    teacher_input = student_latents.detach()

                with torch.no_grad():
                    with self.maybe_enable_amp:
                        teacher_pred = self._flux_forward(
                            teacher_model,
                            latents=teacher_input,
                            latent_pos_enc=tensors["latent_pos_enc"],
                            t5_encodings=tensors["t5_encodings"],
                            text_pos_enc=tensors["text_pos_enc"],
                            clip_encodings=tensors["clip_encodings"],
                            timesteps=current_timesteps,
                        )

                with self.maybe_enable_amp:
                    student_pred = self._flux_forward(
                        student_model,
                        latents=student_latents,
                        latent_pos_enc=tensors["latent_pos_enc"],
                        t5_encodings=tensors["t5_encodings"],
                        text_pos_enc=tensors["text_pos_enc"],
                        clip_encodings=tensors["clip_encodings"],
                        timesteps=current_timesteps,
                    )
                    step_loss = (
                        self.loss_fn(student_pred, teacher_pred.detach())
                        / tensors["global_valid_tokens"]
                    )

                (step_loss / num_steps).backward()
                total_loss = total_loss + step_loss.detach()
                step_losses.append(step_loss.detach())

                with torch.no_grad():
                    student_next = student_latents + delta * student_pred.detach()
                teacher_next = teacher_input + delta * teacher_pred
                if distill_config.detach_rollout_latents:
                    student_latents = student_next.detach()
                else:
                    student_latents = student_next
                teacher_latents = teacher_next.detach()

                del student_pred, teacher_pred

            loss = total_loss / num_steps

        self._last_distillation_step_losses = step_losses
        return loss

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Perform a single forward and backward pass through the model.

        Args:
            input_dict: Dictionary containing input data including prompts and other metadata
            labels: Target tensor containing the ground truth image data
            global_valid_tokens: Optional tensor tracking the total number of valid tokens across all processes.
                This field is a placeholder for now as we rescale the loss within forward_backward_step for FLUX.

        Returns:
            torch.Tensor: The computed loss value for this training step
        """

        assert (
            global_valid_tokens is None
        ), "FLUX model don't need to rescale loss by number of global valid tokens"

        if self.config.distillation.enable:
            return self._forward_backward_distillation_step(
                input_dict=input_dict,
                labels=labels,
            )

        tensors = self._prepare_flux_training_tensors(
            input_dict=input_dict,
            labels=labels,
        )
        model = self.model_parts[0]

        with self.train_context():
            with self.maybe_enable_amp:
                latent_noise_pred = self._flux_forward(
                    model,
                    latents=tensors["latents"],
                    latent_pos_enc=tensors["latent_pos_enc"],
                    t5_encodings=tensors["t5_encodings"],
                    text_pos_enc=tensors["text_pos_enc"],
                    clip_encodings=tensors["clip_encodings"],
                    timesteps=tensors["timesteps"],
                )

                # Scale loss as we used SUM reduction for mse loss function
                # pyrefly: ignore [unsupported-operation]
                loss = (
                    self.loss_fn(latent_noise_pred, tensors["target"])
                    / tensors["global_valid_tokens"]
                )
            # latent_noise_pred.shape=(bs, seq_len, vocab_size)
            # need to free to before bwd to avoid peaking memory
            # pyrefly: ignore[unsupported-delete]
            del latent_noise_pred
            loss.backward()

        return loss

    def train_step(
        self, data_iterator: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        self.optimizers.zero_grad()
        # Save the current step learning rate for logging
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims

        if self.gradient_accumulation_steps > 1:
            raise ValueError("FLUX doesn't support gradient accumulation for now.")

        # pyrefly: ignore [no-matching-overload]
        input_dict, labels = next(data_iterator)

        loss = self.forward_backward_step(input_dict=input_dict, labels=labels)

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=False,
            pp_mesh=parallel_dims.get_optional_mesh("pp"),
            ep_enabled=parallel_dims.ep_enabled,
        )
        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()

        # log metrics
        if not self.metrics_processor.should_log(self.step):
            return

        if parallel_dims.dp_cp_enabled:
            loss = loss.detach()
            loss_mesh = parallel_dims.get_optional_mesh("loss")

            # NOTE: the loss returned by train
            global_avg_loss, global_max_loss, global_ntokens_seen = (
                dist_utils.dist_sum(loss, loss_mesh),
                dist_utils.dist_max(loss, loss_mesh),
                dist_utils.dist_sum(
                    torch.tensor(
                        self.ntokens_seen, dtype=torch.int64, device=self.device
                    ),
                    loss_mesh,
                ),
            )
        else:
            global_avg_loss = global_max_loss = loss.detach().item()
            global_ntokens_seen = self.ntokens_seen

        extra_metrics = {
            "n_tokens_seen": global_ntokens_seen,
            "lr": lr,
        }
        if self.config.distillation.enable:
            for i, step_loss in enumerate(self._last_distillation_step_losses, start=1):
                extra_metrics[f"distillation/step_{i}_loss"] = step_loss.item()
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            grad_norm.item(),
            extra_metrics=extra_metrics,
        )
