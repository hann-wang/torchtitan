# Modifications Copyright(C) Advanced Micro Devices, Inc.
# All rights reserved.
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# credit - Persistent rewrite of forward kernel with L2 caching optimization from @AdnanHoque, IBM Research.

import torch
from torch.library import triton_op, wrap_triton
import triton
import triton.language as tl

from torchtitan.experiments.kernels.blockwise_fp8.grouped_gemm.autotune import (
    STANDARD_CONFIGS,
    ALIGN_SIZE_M,
)

torch_dtype = torch.bfloat16

# ============ Triton kernel for contiguous grouped GEMM ============

# L2 Caching optmization


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, super_group_m):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * super_group_m
    group_size_m = min(num_pid_m - first_pid_m, super_group_m)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


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
def _kernel_cg_persistent_forward(
        # Pointers to matrices
        a_ptr,
        b_ptr,
        c_ptr,
        # Pointer to indices array
        indices_ptr,
        # Pointers to block scales (if needed)
        a_s_ptr,
        b_s_ptr,
        # Matrix dimensions
        M_TOTAL,  # Total M dimension (sum of all groups)
        N: tl.constexpr,  # N dimension
        K: tl.constexpr,  # K dimension
        NUM_EXPERTS: tl.constexpr,  # Number of experts
        BLOCK_SIZE_M: tl.constexpr,  # Tiling parameters
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        NUM_SMS: tl.constexpr,
        # NUM_CONSUMER_GROUPS: tl.constexpr,
        # Group size (for aligned loads)
        GROUP_SIZE_M: tl.constexpr = 128,
        SUPER_GROUP_M: tl.constexpr = 32,  # 32 works best
        USE_FP8: tl.
    constexpr = False,  # use FP8 blockwise quantization proposed by DeepSeek-V3
):
    """
    Contiguous Grouped GEMM kernel forward.
    IMPORTANT: Assumes GROUP_SIZE_M is a multiple of BLOCK_SIZE_M or vice versa,
    and all inputs are pre-aligned to these block boundaries.
    """

    c_type = c_ptr.dtype.element_ty

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M_TOTAL, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = SUPER_GROUP_M * num_pid_n

    b_s_n = N // BLOCK_SIZE_K
    b_s_k = K // BLOCK_SIZE_K

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):

        tile_m_idx, tile_n_idx = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, SUPER_GROUP_M
        )

        # starting indices for this tile
        m_start = tile_m_idx * BLOCK_SIZE_M
        n_start = tile_n_idx * BLOCK_SIZE_N

        # Only process if in bounds
        if m_start < M_TOTAL:

            offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
            offs_n = n_start + tl.arange(0, BLOCK_SIZE_N)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for ki in range(k_tiles):

                # Offsets for K dim
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

                # Create masks for bounds checking
                mask_m = offs_m < M_TOTAL
                mask_n = offs_n < N
                mask_k = offs_k < K

                # masks for A and B
                mask_a = mask_m[:, None] & mask_k[None, :]
                mask_b = mask_n[:, None] & mask_k[None, :]

                # Determine the expert group index and load expert ID
                group_idx = m_start // GROUP_SIZE_M
                expert_idx = tl.load(indices_ptr + group_idx * GROUP_SIZE_M)

                # Load inputs (A) with bounds checking
                a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
                a = tl.load(a_ptrs, mask=mask_a, other=0.0)

                # Load expert weights (B) for the expert assigned to this block
                b_ptrs = (
                    b_ptr + expert_idx * N * K + offs_n[:, None] * K + offs_k[None, :]
                )
                b = tl.load(b_ptrs, mask=mask_b, other=0.0)
                ab = tl.dot(a, b.T)

                if USE_FP8:
                    a_s_ptrs = a_s_ptr + offs_m * b_s_k + ki
                    a_scale = tl.load(a_s_ptrs, mask=mask_m, other=1.0)
                    b_s_ptrs = b_s_ptr + expert_idx * b_s_n * b_s_k + (
                        offs_n // BLOCK_SIZE_K) * b_s_k + ki
                    b_scale = tl.load(b_s_ptrs, mask=mask_n, other=1.0)
                    ab = ab * a_scale[:, None] * b_scale[None, :]

                # Accumulate matrix multiplication for this K tile
                accumulator += ab

            tile_id_c += NUM_SMS
            tile_m_idx, tile_n_idx = _compute_pid(
                tile_id_c, num_pid_in_group, num_pid_m, SUPER_GROUP_M
            )

            offs_m = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            # Create masks for bounds checking
            mask_m = offs_m < M_TOTAL
            mask_n = offs_n < N
            mask_c = mask_m[:, None] & mask_n[None, :]

            c = accumulator.to(tl.float32)

            # Store output (C) with bounds checking
            c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
            tl.store(c_ptrs, c.to(c_type), mask=mask_c)


@triton.autotune(
    configs=STANDARD_CONFIGS,
    key=[
        "M_BUFFERLEN",
        "N",
        "K",
        "GROUP_SIZE_M",
    ],
)
@triton.jit
def _kernel_cg_forward_aligned(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Pointer to indices array
    indices_ptr,
    # Matrix dimensions
    M_BUFFERLEN: tl.constexpr,  # Total M dimension (buffer length)
    M_TOTAL: tl.constexpr,  # Total M dimension (sum of all groups)
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
):
    """
    Contiguous Grouped GEMM kernel forward.
    IMPORTANT: Assumes GROUP_SIZE_M is a multiple of BLOCK_SIZE_M
    and all inputs are pre-aligned/padded to these block boundaries.
    """

    pid = tl.program_id(0)

    c_type = c_ptr.dtype.element_ty

    # number of tiles per matrix dimension
    num_m_tiles = tl.cdiv(M_TOTAL, BLOCK_SIZE_M)
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)

    # 2D tile index from linear
    tile_m = pid // num_n_tiles
    tile_n = pid % num_n_tiles

    # starting indices for this tile
    m_start = tile_m * BLOCK_SIZE_M
    n_start = tile_n * BLOCK_SIZE_N

    # Only process if in bounds
    if m_start < M_TOTAL:

        # Create offset arrays for input, output coordinates
        offs_m = tl.arange(0, BLOCK_SIZE_M) + m_start
        offs_n = tl.arange(0, BLOCK_SIZE_N) + n_start

        # Create masks for bounds checking
        mask_m = offs_m < M_TOTAL
        mask_n = offs_n < N

        # Determine the expert group index and load expert ID
        group_idx = m_start // GROUP_SIZE_M
        expert_idx = tl.load(indices_ptr + group_idx * GROUP_SIZE_M)

        # Initialize accumulator
        acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

        # matrix multiplication in tiles along K dimension
        for k in range(0, K, BLOCK_SIZE_K):
            # offsets and mask for K dimension
            offs_k = tl.arange(0, BLOCK_SIZE_K) + k
            mask_k = offs_k < K

            # masks for A and B
            mask_a = mask_m[:, None] & mask_k[None, :]
            mask_b = mask_n[:, None] & mask_k[None, :]

            # Load inputs (A) with bounds checking
            a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
            a = tl.load(a_ptrs, mask=mask_a, other=0.0)

            # Load expert weights (B) for the expert assigned to this block
            b_ptrs = b_ptr + expert_idx * N * K + offs_n[:, None] * K + offs_k[None, :]
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)

            # Accumulate matrix multiplication for this K tile
            acc += tl.dot(a, b.T)

        # Store results with bounds checking

        c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
        mask_c = mask_m[:, None] & mask_n[None, :]
        tl.store(c_ptrs, acc.to(c_type), mask=mask_c)


# =============== End Triton Kernel for CGGEMM ===============


# =============== Forward Wrapper for CGGEMM =================


@triton_op("torchtitan::cg_grouped_gemm_forward", mutates_args={})
def cg_grouped_gemm_forward(
    inputs: torch.Tensor,  # [M_bufferlen, K]
    expert_weights: torch.Tensor,  # [num_experts, N, K]
    expert_indices: torch.Tensor,  # [M_total]
    input_scales: torch.Tensor | None = None,  # [M_bufferlen, K // block_size] or None
    weight_scales: torch.Tensor | None = None,  # [num_experts, N // block_size, K // block_size] or None
) -> torch.Tensor:
    """
    contiguous grouped GEMM forward pass for MoE.
    All tokens mapped to the same expert must be in contiguous blocks of size group_size_m.

    Args:
        inputs: Input tensor of shape [M_bufferlen, K]
        expert_weights: Expert weight tensor of shape [num_experts, N, K]
        expert_indices: Indices tensor of shape [M_total] mapping each token to its expert
        x_scale: Input tensor scales of shape [M_total, 1]
        w_scale: Expert weight tensor scales of shape [num_experts, N]
    Returns:
        Output tensor of shape [M_total, N]
    """
    # Validate inputs
    assert inputs.is_contiguous(), "Input tensor must be contiguous"
    assert expert_weights.is_contiguous(), "Expert weights tensor must be contiguous"
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"
    # Check if inputs are properly aligned
    M_bufferlen, K = inputs.shape
    M_total = expert_indices.shape[0]
    torch._check_is_size(M_total)
    torch._check(M_total % ALIGN_SIZE_M == 0)
    assert (
        M_total % ALIGN_SIZE_M == 0
    ), f"M_total ({M_total}) must be a multiple of group_size_m ({ALIGN_SIZE_M})"

    # Convert expert_indices to int32 if needed
    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    # Get dimensions
    num_experts, N, K_weights = expert_weights.shape

    # Validate dimensions
    assert K == K_weights, f"Input K ({K}) must match weight K ({K_weights})"
    # assert (
    #     expert_indices.shape[0] == M_total
    # ), f"Expert indices length ({expert_indices.shape[0]}) must match M_total ({M_total})"

    # Create output tensor
    output = torch.zeros((M_bufferlen, N),
                         device=inputs.device,
                         dtype=torch_dtype)

    # Calculate grid size for the kernel
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    use_fp8 = input_scales is not None and weight_scales is not None
    if use_fp8:
        assert input_scales.is_contiguous()
        assert weight_scales.is_contiguous()

    grid = (NUM_SMS, 1, 1)
    # Launch kernel
    wrap_triton(_kernel_cg_persistent_forward)[grid](
        inputs,
        expert_weights,
        output,
        expert_indices,
        input_scales,
        weight_scales,
        M_TOTAL=M_total,
        N=N,
        K=K,
        NUM_EXPERTS=num_experts,
        GROUP_SIZE_M=ALIGN_SIZE_M,
        NUM_SMS=NUM_SMS,
        SUPER_GROUP_M=32,
        USE_FP8=use_fp8,
    )

    return output

@cg_grouped_gemm_forward.register_fake
def cg_grouped_gemm_forward_fake(
    inputs: torch.Tensor,  # [M_bufferlen, K]
    expert_weights: torch.Tensor,  # [num_experts, N, K]
    expert_indices: torch.Tensor,  # [M_total]
    a_scales: torch.Tensor
    | None = None,  # [M_bufferlen, K // block_size] or None
    b_scales: torch.Tensor
    | None = None,  # [num_experts, N // block_size, K // block_size] or None
) -> torch.Tensor:
    """
    Fake function for cg_grouped_gemm_forward.
    Returns a tensor of zeros with the same shape as the expected output.
    """
    M_bufferlen, _ = inputs.shape
    _, N, _ = expert_weights.shape
    return torch.zeros((M_bufferlen, N),
                       dtype=torch_dtype,
                       device=inputs.device)


@triton_op("torchtitan::cg_grouped_gemm_forward_dynamic", mutates_args={})
def cg_grouped_gemm_forward_dynamic(
    inputs: torch.Tensor,  # [M_total, K]
    expert_weights: torch.Tensor,  # [num_experts, N, K]
    expert_indices: torch.Tensor,  # [M_total]
) -> torch.Tensor:
    """
    contiguous grouped GEMM forward pass for MoE.
    All tokens mapped to the same expert must be in contiguous blocks of size group_size_m.

    Args:
        inputs: Input tensor of shape [M_total, K]
        expert_weights: Expert weight tensor of shape [num_experts, N, K]
        expert_indices: Indices tensor of shape [M_total] mapping each token to its expert
        group_size_m: Size of contiguous token blocks for each expert (default: 128)

    Returns:
        Output tensor of shape [M_total, N]
    """
    # Validate inputs
    assert inputs.is_contiguous(), "Input tensor must be contiguous"
    assert expert_weights.is_contiguous(), "Expert weights tensor must be contiguous"
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"

    # Check if inputs are properly aligned
    M_total, K = inputs.shape
    assert (
        M_total % ALIGN_SIZE_M == 0
    ), f"M_total ({M_total}) must be a multiple of group_size_m ({ALIGN_SIZE_M})"
    # assert (
    #    expert_indices.shape[0] == M_total // group_size_m
    # ), "Expert indices length must match number of groups"

    # Convert expert_indices to int32 if needed
    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    # Get dimensions
    num_experts, N, K_weights = expert_weights.shape

    # Validate dimensions
    assert K == K_weights, f"Input K ({K}) must match weight K ({K_weights})"
    assert (
        expert_indices.shape[0] == M_total
    ), "Expert indices length must match M_total"

    # Create output tensor
    output = torch.empty((M_total, N), device=inputs.device, dtype=inputs.dtype)

    # Calculate grid size for the kernel
    grid = lambda meta: (
        triton.cdiv(M_total, meta["BLOCK_SIZE_M"])
        * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )

    # Launch kernel
    wrap_triton(_kernel_cg_forward_aligned)[grid](
        inputs,
        expert_weights,
        output,
        expert_indices,
        M_TOTAL=M_total,
        N=N,
        K=K,
        NUM_EXPERTS=num_experts,
        GROUP_SIZE_M=ALIGN_SIZE_M,
    )

    return output


@cg_grouped_gemm_forward_dynamic.register_fake
def cg_grouped_gemm_forward_dynamic_fake(
    inputs: torch.Tensor,  # [M_total, K]
    expert_weights: torch.Tensor,  # [num_experts, N, K]
    expert_indices: torch.Tensor,  # [M_total]
) -> torch.Tensor:
    """
    Fake function for cg_grouped_gemm_forward_dynamic.
    Returns a tensor of zeros with the same shape as the expected output.
    """
    M_total, _ = inputs.shape
    _, N, _ = expert_weights.shape
    return torch.zeros((M_total, N), dtype=inputs.dtype, device=inputs.device)


# =============== End Forward Wrapper for CGGEMM =================
# =====
# Example of how to use the kernel with ContiguousGroupedGEMM class
class ContiguousGroupedGEMM(torch.autograd.Function):
    """
    Autograd function for contiguous grouped GEMM with block alignment.
    """

    @staticmethod
    def forward(ctx, inputs, expert_weights, expert_indices):
        """Forward pass ."""
        return cg_grouped_gemm_forward(
            inputs=inputs,
            expert_weights=expert_weights,
            expert_indices=expert_indices,
            # use_tma=use_tma,
        )

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass not implemented, yet!."""
        raise NotImplementedError("Backward pass not implemented")


def cg_grouped_gemm(
    inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    # use_tma: bool = True,
) -> torch.Tensor:
    """
    Interface for contiguous grouped GEMM.
    """
    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    return ContiguousGroupedGEMM.apply(
        inputs, expert_weights, expert_indices
    )


# Example usage and verify correctness:
# Use debug.py


# if __name__ == "__main__":
#   test_contiguous_grouped_gemm()
