# Modifications Copyright(C) Advanced Micro Devices, Inc.
# All rights reserved.
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
import triton
from torch.library import triton_op, wrap_triton
# from torch.library import custom_op as triton_op
# wrap_triton = lambda x: x
import triton.language as tl

# Import configs and utilities from cg_forward

from torchtitan.experiments.kernels.blockwise_fp8.grouped_gemm.cg_forward import cg_grouped_gemm_forward
from torchtitan.experiments.kernels.blockwise_fp8.grouped_gemm.autotune import (
    STANDARD_CONFIGS,
    ALIGN_SIZE_M,
    STANDARD_DW_CONFIGS,
)
from torchtitan.experiments.kernels.blockwise_fp8.blockwise_quantization import (
    fp8_blockwise_act_quant,
    fp8_blockwise_weight_quant,
    fp8_blockwise_act_dequant,
)

# ============ Triton kernel for contiguous grouped GEMM backward inputs ============
torch_dtype = torch.bfloat16

@triton.autotune(
    configs=STANDARD_CONFIGS,
    key=[
        "N",
        "K",
        "GROUP_SIZE_M",
        "USE_FP8",
    ],
)
@triton.jit
def _kernel_cg_backward_dx(
    # Pointers to matrices
    grad_output_ptr,  # [M_TOTAL, N]
    b_ptr,  # expert weights [num_experts, N, K]
    grad_input_ptr,  # [M_TOTAL, K]
    # Pointer to indices array
    indices_ptr,  # [M_TOTAL]
    # Matrix dimensions
    go_s_ptr, # [M_TOTAL, N // block_size] or None
    b_s_ptr, # [num_experts, N // block_size, K // block_size] or None
    M_TOTAL,  # Total M dimension (sum of all groups)
    N: tl.constexpr,  # N dimension
    K: tl.constexpr,  # K dimension
    # Number of experts
    NUM_EXPERTS: tl.constexpr,
    # Tiling parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # Group size (for aligned loads)
    GROUP_SIZE_M: tl.constexpr = 128,
    USE_FP8: tl.constexpr = False,  # Whether to use FP8 quantization
):
    """
    Computes the gradient with respect to the inputs (backward pass).
    Performs: grad_input = grad_output @ expert_weights
    """
    pid = tl.program_id(0)

    # number of tiles per matrix dimension
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    b_s_n = N // BLOCK_SIZE_N
    b_s_k = K // BLOCK_SIZE_N

    # 2D tile index from linear
    tile_m = pid // num_k_tiles
    tile_k = pid % num_k_tiles

    # starting indices for this tile
    m_start = tile_m * BLOCK_SIZE_M
    k_start = tile_k * BLOCK_SIZE_K

    # Only process if in bounds
    if m_start < M_TOTAL:

        # Create offset arrays for input, output coordinates
        offs_m = tl.arange(0, BLOCK_SIZE_M) + m_start
        offs_k = tl.arange(0, BLOCK_SIZE_K) + k_start

        # Create masks for bounds checking
        mask_m = offs_m < M_TOTAL
        mask_k = offs_k < K

        # Determine the expert group index and load expert ID
        group_idx = m_start // GROUP_SIZE_M
        expert_idx = tl.load(indices_ptr + group_idx * GROUP_SIZE_M)

        # Initialize accumulator for the gradient
        grad_input = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float32)

        # Compute gradient with respect to inputs in tiles along N dimension
        for n in range(0, N, BLOCK_SIZE_N):
            # offsets and mask for N dimension
            offs_n = tl.arange(0, BLOCK_SIZE_N) + n
            mask_n = offs_n < N

            # Masks for grad_output and weights
            mask_go = mask_m[:, None] & mask_n[None, :]
            mask_w = mask_n[:, None] & mask_k[None, :]

            # Load grad_output with bounds checking
            go_ptrs = grad_output_ptr + offs_m[:, None] * N + offs_n[None, :]
            go = tl.load(go_ptrs, mask=mask_go, other=0.0)

            # Load expert weights for the expert assigned to this block
            # For backward pass, we need W, not W^T, so dimensions are [N, K]
            w_ptrs = b_ptr + expert_idx * N * K + offs_n[:, None] * K + offs_k[None, :]
            w = tl.load(w_ptrs, mask=mask_w, other=0.0)

            gow = tl.dot(go, w)

            if USE_FP8:
                ni = n // BLOCK_SIZE_N
                go_s_ptrs = go_s_ptr + offs_m * b_s_n + ni
                b_s_ptrs = b_s_ptr + expert_idx * b_s_n * b_s_k + ni * b_s_k + offs_k // BLOCK_SIZE_N
                go_scale = tl.load(go_s_ptrs, mask=mask_m, other=1.0)
                w_scale = tl.load(b_s_ptrs, mask=mask_k, other=1.0)
                gow = gow * go_scale[:, None] * w_scale[None, :]

            # Compute partial gradient for this N tile: grad_input += grad_output @ weights
            # Note: We're doing matmul without explicit transpose in triton
            grad_input += gow

        # Store results with bounds checking
        grad_input_ptrs = grad_input_ptr + offs_m[:, None] * K + offs_k[None, :]
        mask_gi = mask_m[:, None] & mask_k[None, :]
        tl.store(grad_input_ptrs, grad_input, mask=mask_gi)


# ============ Triton kernel for contiguous grouped GEMM backward weights ============


# =============== Functions for backward pass =================
# ==== simpler approach =============
@triton.autotune(
    configs=STANDARD_DW_CONFIGS,
    key=[
        "N",
        "K",
        "NUM_EXPERTS",
        "GROUP_SIZE_M",
        "USE_FP8",
    ],
)
@triton.jit
def _kernel_cg_backward_dw(
    # Pointers to matrices
    grad_output_ptr,  # [M_TOTAL, N]
    inputs_ptr,  # [M_TOTAL, K]
    grad_weights_ptr,  # [num_experts, N, K]
    indices_ptr,  # [M_total]
    go_s_ptr,  # [M_TOTAL // block_size, N] or None
    a_s_ptr,  # [M_TOTAL // block_size, K] or None
    # Matrix dimensions
    M_TOTAL,  # Total M dimension
    N: tl.constexpr,  # N dimension
    K: tl.constexpr,  # K dimension
    # Number of experts
    NUM_EXPERTS: tl.constexpr,
    # Group parameters
    GROUP_SIZE_M: tl.constexpr,
    # Tiling parameters
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    USE_FP8: tl.constexpr,
):
    """
    Significantly simplified kernel for weight gradient computation.
    This kernel processes one N-K tile across all groups that use the same expert.
    """
    pid = tl.program_id(0)

    # Determine expert and position within the matrix
    expert_id = pid // ((N * K) // (BLOCK_SIZE_N * BLOCK_SIZE_K))
    position_id = pid % ((N * K) // (BLOCK_SIZE_N * BLOCK_SIZE_K))

    # Only process if expert is valid
    if expert_id < NUM_EXPERTS:
        # Calculate positions in N and K dimensions
        n_tiles = K // BLOCK_SIZE_K
        tile_n = position_id // n_tiles
        tile_k = position_id % n_tiles

        n_start = tile_n * BLOCK_SIZE_N
        k_start = tile_k * BLOCK_SIZE_K

        # Only process if in bounds
        if n_start < N and k_start < K:
            # Create offset arrays
            offs_n = tl.arange(0, BLOCK_SIZE_N) + n_start
            offs_k = tl.arange(0, BLOCK_SIZE_K) + k_start

            # Create masks for bounds checking
            mask_n = offs_n < N
            mask_k = offs_k < K

            # Initialize accumulator for the gradient
            grad_weights = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=tl.float32)

            # Go through all groups to find those using this expert
            for group_idx in range(0, M_TOTAL // GROUP_SIZE_M):
                group_start = group_idx * GROUP_SIZE_M

                # Get expert ID for this group
                group_expert = tl.load(indices_ptr + group_start)

                # Only process if this group uses the current expert
                if group_expert == expert_id:
                    # Process the group in blocks
                    for m_offset in range(0, GROUP_SIZE_M, BLOCK_SIZE_M):
                        # Global offsets for group's data
                        m_start = group_start + m_offset
                        offs_m = tl.arange(0, BLOCK_SIZE_M) + m_start

                        # Create mask for M dimension
                        mask_m = offs_m < min(group_start + GROUP_SIZE_M, M_TOTAL)

                        # Load grad_output [M, N]
                        go_ptrs = (
                            grad_output_ptr + offs_m[:, None] * N + offs_n[None, :]
                        )
                        mask_go = mask_m[:, None] & mask_n[None, :]
                        go = tl.load(go_ptrs, mask=mask_go, other=0.0)

                        # Load inputs [M, K]
                        in_ptrs = inputs_ptr + offs_m[:, None] * K + offs_k[None, :]
                        mask_in = mask_m[:, None] & mask_k[None, :]
                        inp = tl.load(in_ptrs, mask=mask_in, other=0.0)

                        # Compute gradient contribution
                        go_t = tl.trans(go)  # Transpose: [N, M]
                        go_t_inp = tl.dot(go_t, inp)

                        if USE_FP8:
                            # Load scales if using FP8
                            mi = m_start // BLOCK_SIZE_M
                            go_s_ptrs = go_s_ptr + mi * N + offs_n
                            a_s_ptrs = a_s_ptr + mi * K + offs_k
                            go_scale = tl.load(go_s_ptrs, mask=mask_n, other=1.0)
                            a_scale = tl.load(a_s_ptrs, mask=mask_k, other=1.0)
                            go_t_inp = go_t_inp * go_scale[:, None] * a_scale[None, :]
                        grad_weights += go_t_inp

            # Store results to the appropriate part of the expert's weight gradients
            grad_w_ptrs = (
                grad_weights_ptr
                + expert_id * N * K
                + offs_n[:, None] * K
                + offs_k[None, :]
            )
            mask_gw = mask_n[:, None] & mask_k[None, :]
            tl.store(grad_w_ptrs, grad_weights, mask=mask_gw)


@triton_op("torchtitan::cg_grouped_gemm_backward_weights", mutates_args={})
def cg_grouped_gemm_backward_weights(
    grad_output: torch.Tensor,  # [M_bufferlen, N]
    inputs: torch.Tensor,  # [M_bufferlen, K]
    expert_indices: torch.Tensor,  # [M_total]
    num_experts: int,
    go_scales: torch.Tensor |
    None = None,  # [M_bufferlen // block_size, N] Optional scales for grad_output
    input_scales: torch.Tensor
    | None = None,  # [M_bufferlen // block_size, K] Optional scales for inputs
) -> torch.Tensor:
    """
    Simple version of backward pass for weights using a single kernel launch.

    Args:
        grad_output: Gradient from output, shape [M_bufferlen, N]
        inputs: Input tensor, shape [M_bufferlen, K]
        expert_indices: Indices tensor mapping each token to its expert, shape [M_total]
        num_experts: Number of experts

    Returns:
        grad_weights: Gradient with respect to expert weights, shape [num_experts, N, K]
    """
    # Validate inputs
    assert grad_output.is_contiguous(), "Grad output tensor must be contiguous"
    assert inputs.is_contiguous(), "Inputs tensor must be contiguous"
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"

    # Get dimensions
    _, N = grad_output.shape
    M_bufferlen, K = inputs.shape
    M_total = expert_indices.shape[0]
    torch._check_is_size(M_total)
    torch._check(M_total % ALIGN_SIZE_M == 0)

    # Check if dimensions match
    assert (
        M_total % ALIGN_SIZE_M == 0
    ), f"M_total ({M_total}) must be a multiple of group_size_m ({ALIGN_SIZE_M})"

    # Ensure expert_indices has the right dtype
    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    # Create output tensor for gradients
    grad_weights = torch.zeros((num_experts, N, K),
                               device=grad_output.device,
                               dtype=torch_dtype)

    # Calculate grid size for the kernel
    # Each thread block handles one expert's N-K tile
    grid = lambda meta: (
            num_experts * triton.cdiv(N, meta["BLOCK_SIZE_N"]) * triton.cdiv(
                K, meta["BLOCK_SIZE_K"]),
    )
    use_fp8 = go_scales is not None and input_scales is not None
    if use_fp8:
        assert go_scales.is_contiguous()
        assert input_scales.is_contiguous()

    # Launch kernel
    wrap_triton(_kernel_cg_backward_dw)[grid](
        grad_output,
        inputs,
        grad_weights,
        expert_indices,
        go_scales,
        input_scales,
        M_TOTAL=M_total,
        N=N,
        K=K,
        NUM_EXPERTS=num_experts,
        GROUP_SIZE_M=ALIGN_SIZE_M,
        USE_FP8=use_fp8,
    )

    return grad_weights

@cg_grouped_gemm_backward_weights.register_fake
def cg_grouped_gemm_backward_weights_fake(
    grad_output: torch.Tensor,  # [M_bufferlen, N]
    inputs: torch.Tensor,  # [M_bufferlen, K]
    expert_indices: torch.Tensor,  # [M_total]
    num_experts: int,
    go_scales: torch.Tensor |
    None = None,  # [M_bufferlen // block_size, N] Optional scales for grad_output
    input_scales: torch.Tensor
    | None = None,  # [M_bufferlen // block_size, K] Optional scales for inputs
) -> torch.Tensor:
    """
    Fake function for backward pass for weights.
    Returns a tensor of zeros with the same shape as the expected output.
    """
    _, N = grad_output.shape
    _, K = inputs.shape
    return torch.zeros((num_experts, N, K),
                       device=grad_output.device,
                       dtype=torch_dtype)


@triton_op("torchtitan::cg_grouped_gemm_backward_inputs", mutates_args={})
def cg_grouped_gemm_backward_inputs(
    grad_output: torch.Tensor,  # [M_bufferlen, N]
    expert_weights: torch.Tensor,  # [num_experts, N, K]
    expert_indices: torch.Tensor,  # [M_total]
    go_scales: torch.Tensor |
    None = None,  # [M_bufferlen, N // block_size] Optional scales for grad_output
    expert_weight_scales: torch.Tensor |
    None = None,  # [num_experts, N // block_size, K // block_size] Optional scales for expert weights
) -> torch.Tensor:
    """
    Backward pass for contiguous grouped GEMM with respect to inputs.

    Args:
        grad_output: Gradient from output, shape [M_total, N]
        expert_weights: Expert weight tensor, shape [num_experts, N, K]
        expert_indices: Indices tensor mapping each token to its expert, shape [M_total]

    Returns:
        grad_inputs: Gradient with respect to inputs, shape [M_total, K]
    """
    # Validate inputs
    assert grad_output.is_contiguous(), "Grad output tensor must be contiguous"
    assert expert_weights.is_contiguous(), "Expert weights tensor must be contiguous"
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"

    # Get dimensions
    M_bufferlen, N = grad_output.shape
    num_experts, _, K = expert_weights.shape
    M_total = expert_indices.shape[0]
    torch._check_is_size(M_total)
    torch._check(M_total % ALIGN_SIZE_M == 0)

    # Check if dimensions match
    assert (
        M_total % ALIGN_SIZE_M == 0
    ), f"M_total ({M_total}) must be a multiple of group_size_m ({ALIGN_SIZE_M})"

    # Create output tensor for gradients
    grad_inputs = torch.zeros((M_bufferlen, K),
                              device=grad_output.device,
                              dtype=torch_dtype)

    # Calculate grid size for the kernel
    grid = lambda meta: (
        triton.cdiv(M_total, meta["BLOCK_SIZE_M"])
        * triton.cdiv(K, meta["BLOCK_SIZE_K"]),
    )

    use_fp8 = go_scales is not None and expert_weight_scales is not None
    if use_fp8:
        assert go_scales.is_contiguous()
        assert expert_weight_scales.is_contiguous()

    # Launch kernel
    wrap_triton(_kernel_cg_backward_dx)[grid](
        grad_output,
        expert_weights,
        grad_inputs,
        expert_indices,
        go_scales,
        expert_weight_scales,
        M_TOTAL=M_total,
        N=N,
        K=K,
        NUM_EXPERTS=num_experts,
        GROUP_SIZE_M=ALIGN_SIZE_M,
        USE_FP8=use_fp8,
    )
    # grad_output_nan = torch.isnan(grad_output).any()
    # expert_weights_nan = torch.isnan(expert_weights).any()
    # grad_inputs_nan = torch.isnan(grad_inputs).any()
    # if grad_output_nan or expert_weights_nan or grad_inputs_nan:
    #     print(
    #         f"grad_output has NaNs: {torch.isnan(grad_output).any()}, expert_weights has NaNs: {torch.isnan(expert_weights).any()}, grad_inputs has NaNs: {torch.isnan(grad_inputs).any()}"
    #     )
    #     print(f"grad_output={grad_output}, grad_input={grad_inputs}")
    return grad_inputs


@cg_grouped_gemm_backward_inputs.register_fake
def cg_grouped_gemm_backward_inputs_fake(
    grad_output: torch.Tensor,  # [M_bufferlen, N]
    expert_weights: torch.Tensor,  # [num_experts, N, K]
    expert_indices: torch.Tensor,  # [M_total]
    go_scales: torch.Tensor |
    None = None,  # [M_bufferlen, N // block_size] Optional scales for grad_output
    expert_weight_scales: torch.Tensor |
    None = None,  # [num_experts, N // block_size, K // block_size] Optional scales for expert weights
) -> torch.Tensor:
    """
    Fake function for backward pass for inputs.
    Returns a tensor of zeros with the same shape as the expected output.
    """
    M_bufferlen, N = grad_output.shape
    _, _, K = expert_weights.shape
    return torch.zeros((M_bufferlen, K),
                       device=grad_output.device,
                       dtype=torch_dtype)


# =============== Update the autograd function =================


@torch.compiler.allow_in_graph
class ContiguousGroupedGEMM(torch.autograd.Function):
    """
    Autograd function for contiguous grouped GEMM with complete backward pass.
    """

    @staticmethod
    def forward(ctx, inputs, expert_weights, expert_indices, use_fp8=False):
        """Forward pass for contiguous grouped GEMM."""

        if use_fp8:
            inputs_fp8, input_scales = fp8_blockwise_act_quant(
                inputs,
                block_size=ALIGN_SIZE_M,
                axis=-1,
            )
            expert_weights_fp8, expert_weight_scales = fp8_blockwise_weight_quant(
                expert_weights,
                block_size=ALIGN_SIZE_M,
            )
            # Save for backward
            ctx.save_for_backward(inputs_fp8, input_scales, expert_weights_fp8,
                                  expert_weight_scales, expert_indices)
        else:
            # Save for backward
            ctx.save_for_backward(inputs, expert_weights, expert_indices)
        ctx.use_fp8 = use_fp8

        if use_fp8:
            res = cg_grouped_gemm_forward(
                inputs=inputs_fp8,
                expert_weights=expert_weights_fp8,
                expert_indices=expert_indices,
                input_scales=input_scales,
                weight_scales=expert_weight_scales,
            )
        else:
            res = cg_grouped_gemm_forward(
                inputs=inputs,
                expert_weights=expert_weights,
                expert_indices=expert_indices,
            )
        return res

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for contiguous grouped GEMM."""
        use_fp8 = ctx.use_fp8
        if use_fp8:
            inputs_fp8, input_scales, expert_weights, expert_weight_scales, expert_indices = ctx.saved_tensors
        else:
            inputs, expert_weights, expert_indices = ctx.saved_tensors

        # Get number of experts
        num_experts = expert_weights.shape[0]

        # Make sure grad_output is contiguous
        grad_output = grad_output.contiguous()

        if use_fp8:
            # dequant-quant as deepseek-v3 paper
            inputs_dequant = fp8_blockwise_act_dequant(
                inputs_fp8,
                input_scales,
                block_size=ALIGN_SIZE_M,
                axis=-1,
            )
            inputs_fp8, input_scales = fp8_blockwise_act_quant(
                inputs_dequant,
                block_size=ALIGN_SIZE_M,
                axis=0,
            )

            # # quantify inputs/weights stored in high-precision format
            # inputs_fp8, input_scales = fp8_blockwise_act_quant(
            #     inputs,
            #     block_size=ALIGN_SIZE_M,
            #     axis=0,
            # )
            # expert_weights, expert_weight_scales = fp8_blockwise_weight_quant(
            #     expert_weights,
            #     block_size=ALIGN_SIZE_M,
            # )

            grad_output_fp8, grad_output_scales = fp8_blockwise_act_quant(
                grad_output,
                block_size=ALIGN_SIZE_M,
                axis=-1,
            )
            grad_output_fp8_m, grad_output_scales_m = fp8_blockwise_act_quant(
                grad_output,
                block_size=ALIGN_SIZE_M,
                axis=0,
            )
        else:
            inputs_fp8 = inputs
            input_scales = None
            expert_weight_scales = None
            grad_output_fp8 = grad_output
            grad_output_scales = None
            grad_output_fp8_m = grad_output
            grad_output_scales_m = None

        # Compute gradients
        grad_inputs = cg_grouped_gemm_backward_inputs(
            grad_output=grad_output_fp8,
            expert_weights=expert_weights,
            expert_indices=expert_indices,
            go_scales=grad_output_scales,
            expert_weight_scales=expert_weight_scales,
        )
        grad_weights = cg_grouped_gemm_backward_weights(
            grad_output=grad_output_fp8_m,
            inputs=inputs_fp8,
            expert_indices=expert_indices,
            num_experts=num_experts,
            go_scales=grad_output_scales_m,
            input_scales=input_scales,
        )

        # No gradient for expert_indices (it's just an index tensor)
        grad_indices = None

        return grad_inputs, grad_weights, grad_indices, None


def cg_grouped_gemm(
    inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    use_fp8: bool = False,
    trans_weights: bool = True,
) -> torch.Tensor:
    """
    Interface for contiguous grouped GEMM with full backward pass support.

    Args:
        inputs: Input tensor of shape [M_total, K]
        expert_weights: Expert weight tensor of shape [num_experts, N, K]
        expert_indices: Indices tensor of shape [M_total] mapping each token to its expert

    Returns:
        Output tensor of shape [M_total, N]
    """
    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    if not trans_weights:
        expert_weights = expert_weights.transpose(-1, -2).contiguous()
    res = ContiguousGroupedGEMM.apply(inputs, expert_weights, expert_indices,
                                       use_fp8)

    return res


# =============== Test functions for verifying correctness =================

def calc_snr(
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """
    Calculate the Signal-to-Noise Ratio (SNR) between two tensors.
    SNR = 10 * log10(sum(x^2) / (sum((x - y)^2)))
    """
    signal = torch.sum(x ** 2)
    noise = torch.sum((x - y) ** 2)
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

def verify_cg_gemm_backward(
    M_total=1024,
    N=512,
    K=512,
    num_experts=8,
    device="cuda",
    atol=1e-1,  # Absolute tolerance for validation
    rtol=1e-1,  # Relative tolerance for validation
    use_fp8=False,
):
    """
    Verify the correctness of the CG-GEMM backward pass against PyTorch's backward pass.

    Returns:
        tuple: (inputs_passed, weights_passed) indicating if the respective gradients match
    """

    STD = 1

    # Ensure M_total is a multiple of group_size_m
    M_total = (M_total // ALIGN_SIZE_M) * ALIGN_SIZE_M
    num_groups = M_total // ALIGN_SIZE_M

    # Create test tensors
    torch.manual_seed(0)
    dtype = torch_dtype
    inputs = torch.nn.Parameter(
        torch.randn(
            (M_total, K),
            device=device,
            dtype=dtype,
        ) * STD)
    expert_weights = torch.nn.Parameter(
        torch.randn(
            (num_experts, N, K),
            dtype=dtype,
            device=device,
        ) * STD)

    # Create expert indices - each token in a group has the same expert
    expert_indices = torch.zeros(
        M_total,
        dtype=torch.int32,
        device=device,
    )
    for g in range(num_groups):
        expert_idx = torch.randint(
            0, num_experts, (1,), device=device, dtype=torch.int32
        ).item()
        start_idx = g * ALIGN_SIZE_M
        end_idx = (g + 1) * ALIGN_SIZE_M
        expert_indices[start_idx:end_idx] = expert_idx

    # Create a target for gradient computation
    target = torch.randn((M_total, N), dtype=dtype, device=device) * STD

    # PyTorch reference implementation
    outputs_ref = torch.zeros((M_total, N), dtype=dtype, device=device)
    for g in range(num_groups):
        group_start = g * ALIGN_SIZE_M
        group_end = (g + 1) * ALIGN_SIZE_M
        expert_idx = expert_indices[group_start].item()

        # Compute output for this group
        outputs_ref[group_start:group_end] = (
            inputs[group_start:group_end] @ expert_weights[expert_idx].t()
        )

    # Compute loss and gradients with PyTorch
    loss_ref = torch.nn.functional.mse_loss(outputs_ref, target)
    loss_ref.backward()

    grad_inputs_ref = inputs.grad.clone()
    grad_weights_ref = expert_weights.grad.clone()

    # Reset gradients
    inputs.grad.zero_()
    expert_weights.grad.zero_()

    # Compute with our implementation
    outputs = cg_grouped_gemm(
        inputs,
        expert_weights,
        expert_indices,
        use_fp8=use_fp8,
        trans_weights=True,
    )
    loss = torch.nn.functional.mse_loss(outputs, target)
    loss.backward()

    # Check if outputs match
    print(f"outputs={outputs}")
    print(f"outputs_ref={outputs_ref}")
    outputs_match = torch.allclose(outputs, outputs_ref, atol=atol, rtol=rtol)
    print(f"Outputs match: {outputs_match} SNR={calc_snr(outputs, outputs_ref)} cossim={calc_cossim(outputs, outputs_ref)}")
    if not outputs_match:
        # Compute max absolute difference for debugging
        max_diff = torch.max(torch.abs(outputs - outputs_ref))
        print(f"Max difference in outputs: {max_diff}")

        # Check where the largest differences are
        flat_idx = torch.argmax(torch.abs(outputs - outputs_ref))
        row = flat_idx // N
        col = flat_idx % N
        print(f"Largest difference at position [{row}, {col}]")
        print(
            f"Triton: {outputs[row, col].item()}, PyTorch: {outputs_ref[row, col].item()}"
        )

    # Check if gradients match
    inputs_grad_match = torch.allclose(
        inputs.grad, grad_inputs_ref, atol=atol, rtol=rtol
    )
    weights_grad_match = torch.allclose(
        expert_weights.grad, grad_weights_ref, atol=atol, rtol=rtol
    )

    print(f"Input gradients match: {inputs_grad_match} SNR={calc_snr(inputs.grad, grad_inputs_ref)} cossim={calc_cossim(inputs.grad, grad_inputs_ref)}")
    print(f"Weight gradients match: {weights_grad_match} SNR={calc_snr(expert_weights.grad, grad_weights_ref)} cossim={calc_cossim(expert_weights.grad, grad_weights_ref)}")

    if not inputs_grad_match:
        # Compute max absolute difference for debugging
        max_diff = torch.max(torch.abs(inputs.grad - grad_inputs_ref))
        print(f"Max difference in input gradients: {max_diff}")

        # Check where the largest differences are
        flat_idx = torch.argmax(torch.abs(inputs.grad - grad_inputs_ref))
        row = flat_idx // K
        col = flat_idx % K
        print(f"Largest difference at position [{row}, {col}]")
        print(
            f"Triton: {inputs.grad[row, col].item()}, PyTorch: {grad_inputs_ref[row, col].item()}"
        )

    if not weights_grad_match:
        # Compute max absolute difference for debugging
        max_diff = torch.max(torch.abs(expert_weights.grad - grad_weights_ref))
        print(f"Max difference in weight gradients: {max_diff}")

        # Find where the largest differences are
        flat_idx = torch.argmax(
            torch.abs(expert_weights.grad - grad_weights_ref).view(-1)
        )
        expert = flat_idx // (N * K)
        row = (flat_idx % (N * K)) // K
        col = (flat_idx % (N * K)) % K
        print(f"Largest difference at expert {expert}, position [{row}, {col}]")
        print(
            f"Triton: {expert_weights.grad[expert, row, col].item()}, PyTorch: {grad_weights_ref[expert, row, col].item()}"
        )

    return inputs_grad_match, weights_grad_match


def benchmark_cg_gemm_backward(
    M_total=1024,
    N=512,
    K=512,
    num_experts=8,
    device="cuda",
    num_runs=10,
    use_fp8=False,
):
    """
    Benchmark the performance of the CG-GEMM backward pass against PyTorch's backward pass.

    Returns:
        dict: Performance metrics
    """
    import time

    # Ensure M_total is a multiple of group_size_m
    M_total = (M_total // ALIGN_SIZE_M) * ALIGN_SIZE_M
    num_groups = M_total // ALIGN_SIZE_M

    # Create test tensors
    torch.manual_seed(0)
    dtype = torch_dtype
    inputs = torch.randn((M_total, K),
                         device=device,
                         dtype=dtype,
                         requires_grad=True)
    expert_weights = torch.randn((num_experts, N, K),
                                 dtype=dtype,
                                 device=device,
                                 requires_grad=True)

    # Create expert indices - each token in a group has the same expert
    expert_indices = torch.zeros(M_total, dtype=torch.int32, device=device)
    for g in range(num_groups):
        expert_idx = torch.randint(0,
                                   num_experts, (1, ),
                                   device=device,
                                   dtype=torch.int32).item()
        start_idx = g * ALIGN_SIZE_M
        end_idx = (g + 1) * ALIGN_SIZE_M
        expert_indices[start_idx:end_idx] = expert_idx

    # Create gradient for backward pass
    grad_output = torch.randn((M_total, N), dtype=dtype, device=device)

    # Benchmark PyTorch reference implementation
    def run_pytorch_backward():
        # PyTorch reference implementation
        outputs_ref = torch.zeros((M_total, N), dtype=dtype, device=device)
        for g in range(num_groups):
            group_start = g * ALIGN_SIZE_M
            group_end = (g + 1) * ALIGN_SIZE_M
            expert_idx = expert_indices[group_start].item()

            # Compute output for this group
            outputs_ref[group_start:group_end] = (
                inputs[group_start:group_end] @ expert_weights[expert_idx].t())

        # Perform backward pass
        outputs_ref.backward(grad_output)

    # Benchmark our Triton implementation
    def run_triton_backward():
        # Forward pass with our implementation
        outputs = cg_grouped_gemm(
            inputs,
            expert_weights,
            expert_indices,
            use_fp8=use_fp8,
            trans_weights=True,
        )

        # Backward pass
        outputs.backward(grad_output)

    # Warmup
    for _ in range(3):
        inputs.grad = None
        expert_weights.grad = None
        run_pytorch_backward()

        inputs.grad = None
        expert_weights.grad = None
        run_triton_backward()

    # Benchmark PyTorch
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        inputs.grad = None
        expert_weights.grad = None
        run_pytorch_backward()
        torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) / num_runs * 1000  # ms

    # Benchmark Triton
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        inputs.grad = None
        expert_weights.grad = None
        run_triton_backward()
        torch.cuda.synchronize()
    triton_time = (time.time() - start_time) / num_runs * 1000  # ms

    # Calculate speedup
    speedup = pytorch_time / triton_time

    print(f"PyTorch backward time: {pytorch_time:.2f} ms")
    print(f"Triton backward time: {triton_time:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")

    return {
        "pytorch_time_ms": pytorch_time,
        "triton_time_ms": triton_time,
        "speedup": speedup,
        "dimensions": f"M={M_total}, N={N}, K={K}, experts={num_experts}",
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test and benchmark CG-GEMM forward/backward pass.")
    parser.add_argument("--use_fp8", action="store_true",
                        help="Use FP8 quantization for inputs and weights")

    args = parser.parse_args()
    use_fp8 = args.use_fp8

    # Run verification test
    print("Verifying backward pass correctness...")
    inputs_match, weights_match = verify_cg_gemm_backward(use_fp8=use_fp8)

    if inputs_match and weights_match:
        print("\nAll gradients match! Running performance benchmark...")
        benchmark_cg_gemm_backward(use_fp8=use_fp8)
    else:
        print("\nGradient tests failed. Skipping benchmark.")
