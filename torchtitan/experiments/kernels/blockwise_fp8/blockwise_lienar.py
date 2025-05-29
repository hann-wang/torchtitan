# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from .blockwise_quantization import (
    blockwise_fp8_gemm,
    fp8_blockwise_act_quant,
    fp8_blockwise_weight_quant,
)


class BlockwiseQuantLinear(nn.Linear):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        block_size (int): Block size for quantization. Defaults to 128.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = False,
                 block_size: int = 128,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__(in_features,
                         out_features,
                         bias=bias,
                         device=device,
                         dtype=dtype)

        self.block_size = block_size


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        x, x_scale = fp8_blockwise_act_quant(x, self.block_size)
        w, w_scale = fp8_blockwise_weight_quant(self.weight, self.block_size)
        y = blockwise_fp8_gemm(x, x_scale, w, w_scale, self.block_size)

        if self.bias is not None:
            y += self.bias
        return y

    @classmethod
    def from_float(cls, layer: nn.Linear, block_size: int = 128):
        """
        Create a BlockwiseQuantLinear layer from a standard nn.Linear layer.

        Args:
            layer (nn.Linear): The linear layer to convert.
            block_size (int): Block size for quantization. Defaults to 128.

        Returns:
            BlockwiseQuantLinear: The converted quantized linear layer.
        """
        new_layer = cls(
            layer.in_features,
            layer.out_features,
            layer.bias is not None,
            block_size=block_size,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        )
        new_layer.weight.data.copy_(layer.weight.data)
        if layer.bias is not None:
            new_layer.bias.data.copy_(layer.bias.data)
        return new_layer
