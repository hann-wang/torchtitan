# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

from dataclasses import dataclass

import spmd_types as spmd
import torch
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.moe import GroupedExperts, MoE
from torchtitan.protocols.module import Module


class GptOssGroupedExperts(GroupedExperts):
    """MLPerf ref impl requires the deepseek-style MoE"""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        pass


class GptOssMoE(MoE):
    """GptOss MoE implementation that inherits from the base MoE class."""

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        pass
