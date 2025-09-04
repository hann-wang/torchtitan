# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from importlib.metadata import version
from importlib.util import find_spec
from typing import Any, List

import torch.nn as nn

from torchtitan.config.job_config import JobConfig, MX
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.expert_parallel import set_token_group_alignment_size_m
from torchtitan.protocols.model_converter import (
    ModelConverter,
    register_model_converter,
)
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from torchtitan.experiments.kernels.blockwise_fp4 import MXFP4Linear, ALIGN_SIZE_M

from .utils import module_filter_fn


class MXConverter(ModelConverter):
    """Converts the linear layers of `model` to `MXLinear`."""

    enabled: bool
    filter_fqns: List[str]
    mx_config: Any  # MXLinearConfig type when imported

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        mx_job_config: MX = job_config.mx
        self.mxfp4_recipe = None

        if not mx_job_config.recipe_name.startswith("mxfp4_"):
            # Ensure minimum torchao versions
            if find_spec("torchao") is None:
                raise ImportError(
                    "torchao is not installed. Please install it to use MXFP8 linear layers."
                )
            torchao_version = version("torchao")

            # Last torchao release was 0.12.0, so nightly build starts with 0.13.0+git...
            is_nightly_build = torchao_version.startswith("0.13.0")
            if not is_nightly_build:
                raise ImportError(
                    f"torchao version {torchao_version} is too old, please install torchao nightly build and try again"
                )

            # Can be removed if we enable the emulated versions
            assert has_cuda_capability(
                10, 0
            ), "MXFP8 is only supported on SM100 or architectures"

            # TP not yet supported with torch.compile

            model_compile_enabled = (
                job_config.compile.enable and "model" in job_config.compile.components
            )
            assert not (
                model_compile_enabled and job_config.parallelism.tensor_parallel_degree > 1
            ), "TP not yet supported with torch.compile for mxfp8"

            # For MoE training with mxfp8, token group sizes must be multiples of 32
            if job_config.mx.moe_fqns_prototype:
                mxfp8_block_size = 32
                set_token_group_alignment_size_m(mxfp8_block_size)
                logger.info(f"Setting token group alignment size to {mxfp8_block_size}")

            # Configure MXFP8
            from torchao.prototype.mx_formats.config import (
                MXFP8Dim1CastKernelChoice,
                MXLinearConfig,
            )

            config = MXLinearConfig.from_recipe_name(mx_job_config.recipe_name)
            config.mxfp8_dim1_cast_kernel_choice = MXFP8Dim1CastKernelChoice[
                mx_job_config.mxfp8_dim1_cast_kernel_choice.upper()
            ]
            self.filter_fqns = mx_job_config.filter_fqns
            self.config = config
            self.enabled = True
            logger.info(f"Float8 training active with recipe {mx_job_config.recipe_name}")
        else:
            self.filter_fqns = mx_job_config.filter_fqns
            self.enabled = True
            set_token_group_alignment_size_m(ALIGN_SIZE_M)
            logger.info(f"Setting token group alignment size to {ALIGN_SIZE_M}")

            self.enable_mxfp4_fa = mx_job_config.enable_mxfp4_fa
            self.enable_mxfp4_gmm = mx_job_config.enable_mxfp4_gmm
            self.enable_mxfp4_linear = mx_job_config.enable_mxfp4_linear
            self.mxfp4_recipe = mx_job_config.recipe_name
            if self.mxfp4_recipe.endswith("_1d1d"):
                self.attrs_2dblock = {
                    "use_2dblock_x": False,
                    "use_2dblock_w": False,
                }
            elif self.mxfp4_recipe.endswith("_1d2d"):
                self.attrs_2dblock = {
                    "use_2dblock_x": False,
                    "use_2dblock_w": True,
                }
            elif self.mxfp4_recipe.endswith("_2d2d"):
                self.attrs_2dblock = {
                    "use_2dblock_x": True,
                    "use_2dblock_w": True,
                }
            else:
                raise ValueError(f"Unknown MXFP4 recipe: {self.mxfp4_recipe}")

    def _apply_attrs_2dblock(self, mod: nn.Module):
        mod.use_mxfp4 = True
        for k, v in self.attrs_2dblock.items():
            setattr(mod, k, v)

    def convert(self, model: nn.Module):
        """
        Converts the linear layers of `model` to `MXLinear`.
        Note that today, only dynamic tensor scaling (the default) is supported.
        This will mutate the model inplace.
        """
        if not self.enabled:
            return

        if self.mxfp4_recipe is None:
            from torchao.prototype.mx_formats.config import MXLinearConfig
            from torchao.quantization import quantize_

            assert isinstance(self.config, MXLinearConfig)
            quantize_(
                model,
                config=self.config,
                filter_fn=partial(module_filter_fn,
                                  filter_fqns=self.filter_fqns),
            )
            logger.info("Swapped to MXLinear layers")

        else:
            from torchao.float8.float8_linear_utils import swap_linear_layers
            from_float = lambda m: MXFP4Linear.from_float(m)
            swap_linear_layers(
                model,
                from_float,
                module_filter_fn=partial(module_filter_fn,
                                         filter_fqns=self.filter_fqns),
            )
            logger.info("Swapped to MXFP4Linear layers")

            if self.enable_mxfp4_fa:
                self._enable_mxfp4_fa(model)
            if self.enable_mxfp4_gmm:
                self._enable_mxfp4_gmm(model)

    def _enable_mxfp4_fa(self, model: nn.Module, prefix: str = ""):
        for mod_name, mod in model.named_children():
            if mod.__class__.__name__.endswith("FlashAttention2"):
                skip = False
                full_name = f"{prefix}{'.' if prefix else ''}{mod_name}"
                for f in self.filter_fqns:
                    if f in full_name:
                        skip = True
                        break
                if not skip:
                    self._apply_attrs_2dblock(mod)
                    logger.info(f"Enable MXFP4 kernel for {full_name}.")
                else:
                    logger.info(f"Skip enabling MXFP4 kernel for {full_name}.")
            else:
                self._enable_mxfp4_fa(
                    mod, f"{prefix}{'.' if prefix else ''}{mod_name}")

    def _enable_mxfp4_gmm(self, model: nn.Module, prefix: str = ""):
        for mod_name, mod in model.named_children():
            if mod.__class__.__name__.endswith("GroupedExperts"):
                skip = False
                full_name = f"{prefix}{'.' if prefix else ''}{mod_name}"
                for f in self.filter_fqns:
                    if f in full_name:
                        skip = True
                        break
                if not skip:
                    self._apply_attrs_2dblock(mod)
                    logger.info(f"Enable MXFP4 kernel for {full_name}.")
                else:
                    logger.info(f"Skip enabling MXFP4 kernel for {full_name}.")
            else:
                self._enable_mxfp4_gmm(
                    mod, f"{prefix}{'.' if prefix else ''}{mod_name}")

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        """
        MXFP8 doesn't require any post-optimizer hooks at the moment
        """
        return


register_model_converter(MXConverter, "mx")
