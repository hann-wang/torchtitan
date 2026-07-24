# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass, field
from typing import Protocol

import torch.nn as nn

from torchtitan.config import Configurable
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger
from torchtitan.protocols.model import BaseModel


class ModelConverter(Protocol):
    """General model converter interface.

    A model converter is applying a modification to PyTorch model.
    Typical use cases are:
        - Quantization: using QAT, FP8, ... specialized linear layers;
        - Fused optimized layers (e.g. flash-attention, norms, ...)
    """

    def convert(self, model: nn.Module):
        """Inplace conversion of the model."""
        ...
        
    def convert_config(self, model_config: BaseModel.Config):
        """Inplace conversion of the model config."""
        ...

    def pre_step(self, model_parts: list[nn.Module], **kwargs):
        ...

    def post_optimizer_hook(self, model_parts: list[nn.Module], **kwargs):
        """Post-optimizer (optional) hook (e.g. compute weights statistics)."""
        ...

    def post_initialization(self, model_parts: list[nn.Module]):
        ...

    def finalize(self, model_parts: list[nn.Module]):
        ...


class ModelConvertersContainer(Configurable, ModelConverter):
    """Model converters sequential container.

    Builds converters from their Config objects and applies them
    to the model sequentially.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Configuration for model converters (quantization, etc.).

        Each entry in converters should be a Configurable.Config instance
        (e.g. Float8LinearConverter.Config) whose build() constructs the converter.
        """

        converters: list = field(default_factory=list)
        """List of converter Config objects to apply to the model."""

        print_after_conversion: bool = False
        """If true, model definition will be printed after converters are applied."""

    def __init__(
        self,
        config: Config,
        *,
        parallel_dims: ParallelDims,
        model_compile_enabled: bool,
    ):
        self.converters: list[ModelConverter] = [
            cc.build(
                parallel_dims=parallel_dims,
                model_compile_enabled=model_compile_enabled,
            )
            for cc in config.converters
        ]
        self.print_after_conversion = config.print_after_conversion

    def convert(self, model: nn.Module):
        for mh in self.converters:
            mh.convert(model)
        if self.print_after_conversion:
            logger.info(f"Model definition after conversion:\n\n{model}\n\n")
    
    def convert_config(self, model_config: BaseModel.Config):
        for mh in self.converters:
            mh.convert_config(model_config)

    def pre_step(self, model_parts: list[nn.Module], **kwargs):
        for mh in self.converters:
            mh.pre_step(model_parts, **kwargs)

    def post_optimizer_hook(self, model_parts: list[nn.Module], **kwargs):
        for mh in self.converters:
            mh.post_optimizer_hook(model_parts, **kwargs)

    def post_initialization(self, model_parts: list[nn.Module]):
        for mh in self.converters:
            mh.post_initialization(model_parts)

    def finalize(self, model_parts: list[nn.Module]):
        for mh in self.converters:
            mh.finalize(model_parts)

    def is_empty(self):
        return len(self.converters) == 0
