# Modifications Copyright(C) Advanced Micro Devices, Inc.
# All rights reserved.
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._op_schema import PlacementList
from torch.distributed.tensor.placement_types import (
    Partial,
    Replicate,
    Shard,
)

from torchtitan.experiments.kernels.blockwise_fp8.blockwise_quantization import (
    blockwise_fp8_gemm,
    blockwise_fp8_gemm_backward_act,
    blockwise_fp8_gemm_backward_weight,
    fp8_blockwise_act_quant,
    fp8_blockwise_weight_quant,
    fp8_blockwise_act_dequant,
)


@torch.compiler.allow_in_graph
class BlockwiseFP8LinearFunction(torch.autograd.Function):
    """
    Custom autograd function for blockwise FP8 linear operations.
    This function handles the forward and backward passes for the linear layer.
    """

    @staticmethod
    def forward(ctx, x, weight, block_size=128):
        """
        Forward pass for the blockwise FP8 linear operation.

        Args:
            ctx: Context object to save information for backward pass.
            x (torch.Tensor): Input tensor.
            weight (torch.Tensor): Weight tensor.
            bias (torch.Tensor, optional): Bias tensor. Defaults to None.
            block_size (int): Block size for quantization. Defaults to 128.

        Returns:
            torch.Tensor: Output tensor after linear transformation.
        """
        original_shape = x.shape
        x = x.contiguous().view(-1, original_shape[-1])  # Ensure x is 2D

        x, x_scale = fp8_blockwise_act_quant(x, block_size)
        weight, w_scale = fp8_blockwise_weight_quant(weight, block_size)

        # # debug
        # x_scale = torch.ones(
        #     (x.shape[0], x.shape[1] // block_size),
        #     dtype=torch.float32,
        #     device=x.device,
        # )
        # w_scale = torch.ones(
        #     (weight.shape[0] // block_size, weight.shape[1] // block_size),
        #     dtype=torch.float32,
        #     device=weight.device,
        # )

        y = blockwise_fp8_gemm(x, x_scale, weight, w_scale, block_size)

        ctx.save_for_backward(x, x_scale, weight, w_scale)
        ctx.block_size = block_size

        return y.view(*original_shape[:-1], -1)  # Reshape back to original

    @staticmethod
    def backward(ctx, grad_output):
        original_shape = grad_output.shape
        grad_output = grad_output.contiguous().view(-1, original_shape[-1])  # Ensure grad_output is 2D

        block_size = ctx.block_size
        x, x_scale, weight, w_scale = ctx.saved_tensors

        # dequant-quant as deepseek-v3 paper
        inputs_dequant = fp8_blockwise_act_dequant(
            x,
            x_scale,
            block_size=block_size,
            axis=-1,
        )
        inputs_fp8, input_scales = fp8_blockwise_act_quant(
            inputs_dequant,
            block_size=block_size,
            axis=0,
        )

        grad_output_fp8, grad_output_scales = fp8_blockwise_act_quant(
            grad_output,
            block_size=block_size,
            axis=-1,
        )
        grad_output_fp8_m, grad_output_scales_m = fp8_blockwise_act_quant(
            grad_output,
            block_size=block_size,
            axis=0,
        )

        # # debug
        # input_scales = torch.ones(
        #     (x.shape[0] // block_size, x.shape[1]),
        #     dtype=torch.float32,
        #     device=x.device,
        # )
        # grad_output_scales = torch.ones(
        #     (grad_output.shape[0], grad_output.shape[1] // block_size),
        #     dtype=torch.float32,
        #     device=grad_output.device,
        # )
        # grad_output_scales_m = torch.ones(
        #     (grad_output.shape[0] // block_size, grad_output.shape[1]),
        #     dtype=torch.float32,
        #     device=grad_output.device,
        # )
        # grad_output_fp8 = grad_output_fp8_m = grad_output
        # inputs_fp8 = x

        # Compute gradients
        grad_inputs = blockwise_fp8_gemm_backward_act(
            grad_output_fp8,
            grad_output_scales,
            weight,
            w_scale,
            block_size=block_size,
        )
        grad_weights = blockwise_fp8_gemm_backward_weight(
            grad_output_fp8_m,
            grad_output_scales_m,
            inputs_fp8,
            input_scales,
            block_size=block_size,
        )

        return grad_inputs.view(*original_shape[:-1], -1), grad_weights, None


single_mesh_dim_strategies = []
colwise: PlacementList = [
    Shard(-1),
    Replicate(),  # x
    Shard(0),  # w
]
rowwise: PlacementList = [
    Partial(),
    Shard(-1),  # x
    Shard(1),  # w
]
single_mesh_dim_strategies.extend([colwise, rowwise])


class BlockwiseFP8Linear(nn.Linear):
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
        assert not self.bias, "Bias is not supported in BlockwiseFP8Linear"


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """

        device_mesh = None
        if isinstance(x, DTensor):
            device_mesh = x.device_mesh
            output_placement = None

            rowwise_abs: PlacementList = [
                Partial(),
                Shard(x.dim() - 1),  # x
                Shard(1),  # w
            ]
            for acceptable_placement in single_mesh_dim_strategies + [
                    rowwise_abs
            ]:
                if acceptable_placement[1:] == [x.placements[-1], self.weight.placements[-1]]:
                    output_placement = acceptable_placement[0]
                    break
            assert output_placement is not None, \
                f"Unsupported placement strategy for BlockwiseFP8Linear: {x.placements} and {self.weight.placements}"
            x = x.to_local()
            w = self.weight.to_local()
        else:
            w = self.weight
        y = BlockwiseFP8LinearFunction.apply(x, w, self.block_size)

        if device_mesh is not None:
            y = DTensor.from_local(y,
                                   device_mesh=device_mesh,
                                   placements=(output_placement, ))

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



# =============== Test functions for verifying correctness =================


def calc_snr(
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """
    Calculate the Signal-to-Noise Ratio (SNR) between two tensors.
    SNR = 10 * log10(sum(x^2) / (sum((x - y)^2)))
    """
    signal = torch.sum(x**2)
    noise = torch.sum((x - y)**2)
    return 10 * torch.log10(signal / noise)


def calc_cossim(
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """
    Calculate the cosine similarity between two tensors.
    Cosine similarity = dot(x, y) / (norm(x) * norm(y))
    """
    x = x.reshape(-1)
    y = y.reshape(-1)
    dot_product = torch.sum(x * y)
    norm_x = torch.norm(x)
    norm_y = torch.norm(y)
    return dot_product / (norm_x * norm_y)


def verify_blockwise_linear_backward(
    B=1,
    M=512,
    N=512,
    K=512,
    device="cuda",
    atol=1e-1,  # Absolute tolerance for validation
    rtol=1e-1,  # Relative tolerance for validation
    compile=False,
):
    STD = 1

    # Create test tensors
    torch.manual_seed(0)
    dtype = torch.bfloat16
    inputs = torch.nn.Parameter(
        torch.randn(
            (B, M, K),
            device=device,
            dtype=dtype,
        ) * STD)
    weights = torch.nn.Parameter(
        torch.randn(
            (N, K),
            dtype=dtype,
            device=device,
        ) * STD)

    # Create a target for gradient computation
    target = torch.randn((B, M, N), dtype=dtype, device=device) * STD

    # PyTorch reference implementation
    outputs_ref = torch.nn.functional.linear(inputs, weights)

    # Compute loss and gradients with PyTorch
    loss_ref = torch.nn.functional.mse_loss(outputs_ref, target)
    loss_ref.backward()

    grad_inputs_ref = inputs.grad.clone()
    grad_weights_ref = weights.grad.clone()

    # Reset gradients
    inputs.grad.zero_()
    weights.grad.zero_()

    # Compute with our implementation
    blockwise_linear_func = BlockwiseFP8LinearFunction.apply
    if compile:
        blockwise_linear_func = torch.compile(blockwise_linear_func, fullgraph=True)
    outputs = BlockwiseFP8LinearFunction.apply(inputs, weights)
    loss = torch.nn.functional.mse_loss(outputs, target)
    loss.backward()

    # Check if outputs match
    print(f"outputs={outputs}")
    print(f"outputs_ref={outputs_ref}")
    outputs_match = torch.allclose(outputs, outputs_ref, atol=atol, rtol=rtol)
    print(
        f"Outputs match: {outputs_match} SNR={calc_snr(outputs, outputs_ref)} cossim={calc_cossim(outputs, outputs_ref)}"
    )
    if not outputs_match:
        # Compute max absolute difference for debugging
        max_diff = torch.max(torch.abs(outputs - outputs_ref))
        print(f"Max difference in outputs: {max_diff}")

        # Check where the largest differences are
        flat_idx = torch.argmax(torch.abs(outputs - outputs_ref))
        bs = flat_idx // (M * N)
        row = (flat_idx % (M * N)) // N
        col = (flat_idx % (M * N)) % N
        print(f"Largest difference at position [{bs}, {row}, {col}]")
        print(
            f"Triton: {outputs[bs, row, col].item()}, PyTorch: {outputs_ref[bs, row, col].item()}"
        )

    # Check if gradients match
    inputs_grad_match = torch.allclose(inputs.grad,
                                       grad_inputs_ref,
                                       atol=atol,
                                       rtol=rtol)
    weights_grad_match = torch.allclose(weights.grad,
                                        grad_weights_ref,
                                        atol=atol,
                                        rtol=rtol)
    print(f"inputs.grad={inputs.grad}")
    print(f"grad_inputs_ref={grad_inputs_ref}")
    print(
        f"Input gradients match: {inputs_grad_match} SNR={calc_snr(inputs.grad, grad_inputs_ref)} cossim={calc_cossim(inputs.grad, grad_inputs_ref)}"
    )
    print(f"weights.grad={weights.grad}")
    print(f"grad_weights_ref={grad_weights_ref}")
    print(
        f"Weight gradients match: {weights_grad_match} SNR={calc_snr(weights.grad, grad_weights_ref)} cossim={calc_cossim(weights.grad, grad_weights_ref)}"
    )

    if not inputs_grad_match:
        # Compute max absolute difference for debugging
        max_diff = torch.max(torch.abs(inputs.grad - grad_inputs_ref))
        print(f"Max difference in input gradients: {max_diff}")

        # Check where the largest differences are
        flat_idx = torch.argmax(torch.abs(inputs.grad - grad_inputs_ref))
        bs = flat_idx // (K * M)
        row = (flat_idx % (K * M)) // K
        col = (flat_idx % (K * M)) % K
        print(f"Largest difference at position [{bs}, {row}, {col}]")
        print(
            f"Triton: {inputs.grad[bs, row, col].item()}, PyTorch: {grad_inputs_ref[bs, row, col].item()}"
        )

    if not weights_grad_match:
        # Compute max absolute difference for debugging
        max_diff = torch.max(torch.abs(weights.grad - grad_weights_ref))
        print(f"Max difference in weight gradients: {max_diff}")

        # Find where the largest differences are
        flat_idx = torch.argmax(
            torch.abs(weights.grad - grad_weights_ref).view(-1))
        row = flat_idx // K
        col = flat_idx % K
        print(
            f"Largest difference at position [{row}, {col}]")
        print(
            f"Triton: {weights.grad[row, col].item()}, PyTorch: {grad_weights_ref[row, col].item()}"
        )

    return inputs_grad_match, weights_grad_match


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description="Test and benchmark blockwise FP8 linear forward/backward pass.")
    parser.add_argument("--compile",
                        action="store_true",
                        help="Use torch.compile")

    args = parser.parse_args()
    compile = args.compile

    # Run verification test
    print("Verifying backward pass correctness...")
    inputs_match, weights_match = verify_blockwise_linear_backward(
        compile=compile)
