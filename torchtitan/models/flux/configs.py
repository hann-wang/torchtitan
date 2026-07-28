# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Literal


@dataclass(kw_only=True, slots=True)
class Encoder:
    t5_encoder: str = "google/t5-v1_1-small"
    """T5 encoder to use, HuggingFace model name. This field could be either a local folder path,
        or a Huggingface repo name."""
    clip_encoder: str = "openai/clip-vit-large-patch14"
    """Clip encoder to use, HuggingFace model name. This field could be either a local folder path,
        or a Huggingface repo name."""
    autoencoder_path: str = (
        "torchtitan/experiments/flux/assets/autoencoder/ae.safetensors"
    )
    """Autoencoder checkpoint path to load. This should be a local path referring to a safetensors file."""
    max_t5_encoding_len: int = 256
    """Maximum length of the T5 encoding."""

    test_mode: bool = False
    """Whether to use integration test mode, which will randomly initialize the encoder and use a dummy tokenizer"""


# TODO: maybe consolidate with FluxValidator.Config
@dataclass(kw_only=True, slots=True)
class Validation:
    enable_classifier_free_guidance: bool = False
    """Whether to use classifier-free guidance during sampling"""
    classifier_free_guidance_scale: float = 5.0
    """Classifier-free guidance scale when sampling"""
    denoising_steps: int = 50
    """How many denoising steps to sample when generating an image"""
    eval_freq: int = 100
    """Frequency of evaluation/sampling during training"""


@dataclass(kw_only=True, slots=True)
class Inference:
    """Inference configuration"""

    save_img_folder: str = "inference_results"
    """Path to save the inference results"""
    prompts_path: str = "./torchtitan/experiments/flux/inference/prompts.txt"
    """Path to file with newline separated prompts to generate images for"""
    local_batch_size: int = 2
    """Batch size for inference"""
    img_size: int = 256
    """Image size for inference"""


@dataclass(kw_only=True, slots=True)
class Distillation:
    """Configuration for Flux transformer distillation."""

    enable: bool = False
    """Whether to replace flow-matching training with teacher/student distillation."""

    mode: Literal["off_policy", "on_policy"] = "off_policy"
    """Rollout policy for the teacher inputs during distillation."""

    num_rollout_steps: int = 4
    """Number of latent rollout steps used for the distillation loss."""

    teacher_checkpoint_path: str | None = None
    """Model-only DCP checkpoint path for the frozen teacher transformer."""

    student_checkpoint_path: str | None = None
    """Model-only DCP checkpoint path used to initialize the student transformer."""

    detach_rollout_latents: bool = True
    """Detach recurrent rollout latents between steps to avoid long backprop graphs."""
