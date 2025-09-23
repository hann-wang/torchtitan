# Modifications Copyright(C) Advanced Micro Devices, Inc.
# All rights reserved.
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch.library import triton_op, wrap_triton
import triton
import triton.language as tl

from torchtitan.experiments.kernels.mxfp4.mxfp_quantization import BLOCK_SIZE_DEFAULT
from torchtitan.experiments.kernels.mxfp4.mxfp_grouped_gemm.autotune import (
    STANDARD_CONFIGS,
    ALIGN_SIZE_M,
)


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
        "USE_2DBLOCK_A",
        "USE_2DBLOCK_B",
        "QUANT_BLOCK_SIZE",
    ],
)
@triton.jit
def _kernel_mxfp4_grouped_gemm_forward(
        # Pointers to matrices
        a_ptr,
        b_ptr,
        c_ptr,
        # Pointer to indices array
        indices_ptr,
        # Pointers to mxfp scales
        a_s_ptr,
        b_s_ptr,
        # Matrix strides
        stride_am,
        stride_ak,
        stride_be,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_cn,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsn,
        stride_bsk,
        # Matrix dimensions
        M_TOTAL,  # Total M dimension (sum of all groups)
        N: tl.constexpr,  # N dimension
        K: tl.constexpr,  # K dimension
        NUM_EXPERTS: tl.constexpr,  # Number of experts
        BLOCK_SIZE_M: tl.constexpr,  # Tiling parameters
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        NUM_SMS: tl.constexpr,
        USE_2DBLOCK_A: tl.constexpr,
        USE_2DBLOCK_B: tl.constexpr,
        QUANT_BLOCK_SIZE: tl.constexpr,
        # Group size (for aligned loads)
        GROUP_SIZE_M: tl.constexpr = 128,
        SUPER_GROUP_M: tl.constexpr = 32,  # 32 works best
):
    """
    Contiguous Grouped GEMM kernel forward.
    IMPORTANT: Assumes GROUP_SIZE_M is a multiple of BLOCK_SIZE_M or vice versa,
    and all inputs are pre-aligned to these block boundaries.
    """
    if USE_2DBLOCK_A:
        tl.assume(BLOCK_SIZE_M % QUANT_BLOCK_SIZE == 0)
    if USE_2DBLOCK_B:
        tl.assume(BLOCK_SIZE_N % QUANT_BLOCK_SIZE == 0)
    tl.assume(BLOCK_SIZE_K % QUANT_BLOCK_SIZE == 0)
    PACKED_BLOCK_SIZE_K: tl.constexpr = BLOCK_SIZE_K // 2
    K_PACKED: tl.constexpr = K // 2

    c_type = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M_TOTAL, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = SUPER_GROUP_M * num_pid_n

    # number of MXFP blocks inside a tile
    n_rep_k: tl.constexpr = BLOCK_SIZE_K // QUANT_BLOCK_SIZE
    # total number of scales
    Ks: tl.constexpr = (K // QUANT_BLOCK_SIZE)

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

            if USE_2DBLOCK_A:
                offs_m_scale = offs_m // QUANT_BLOCK_SIZE
            else:
                offs_m_scale = offs_m
            if USE_2DBLOCK_B:
                offs_n_scale = offs_n // QUANT_BLOCK_SIZE
            else:
                offs_n_scale = offs_n

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for ki in range(k_tiles):

                # Offsets for K dim
                offs_k_pack = ki * PACKED_BLOCK_SIZE_K + tl.arange(
                    0, PACKED_BLOCK_SIZE_K)
                offs_k_scale = ki * n_rep_k + tl.arange(0, n_rep_k)
                mask_k_scale = offs_k_scale < Ks

                # Create masks for bounds checking
                mask_m = offs_m < M_TOTAL
                mask_n = offs_n < N
                mask_k_pack = offs_k_pack < K_PACKED

                # masks for A and B
                mask_a = mask_m[:, None] & mask_k_pack[None, :]
                mask_b = mask_k_pack[:, None] & mask_n[None, :]

                # Determine the expert group index and load expert ID
                group_idx = m_start // GROUP_SIZE_M
                expert_idx = tl.load(indices_ptr + group_idx * GROUP_SIZE_M)

                # Load inputs (A) with bounds checking
                a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k_pack[None, :] * stride_ak
                a = tl.load(a_ptrs, mask=mask_a, other=0)

                # Load expert weights (B) for the expert assigned to this block
                b_ptrs = (b_ptr + expert_idx * stride_be +
                          offs_n[None, :] * stride_bn +
                          offs_k_pack[:, None] * stride_bk)
                b = tl.load(b_ptrs, mask=mask_b, other=0)

                a_s_ptrs = a_s_ptr + offs_m_scale[:, None] * stride_asm + offs_k_scale[
                    None, :] * stride_ask
                # B scales are N x K even though B operand is K x N.
                b_s_ptrs = b_s_ptr + expert_idx * stride_bse + offs_n_scale[:, None] * stride_bsn + offs_k_scale[
                    None, :] * stride_bsk

                a_s = tl.load(a_s_ptrs,
                              mask=mask_m[:, None] & mask_k_scale[None, :],
                              other=1)
                b_s = tl.load(b_s_ptrs,
                              mask=mask_n[:, None] & mask_k_scale[None, :],
                              other=1)
                accumulator += tl.dot_scaled(a,
                                             a_s,
                                             "e2m1",
                                             b,
                                             b_s,
                                             "e2m1",
                                             lhs_k_pack=True,
                                             rhs_k_pack=True,
                                             out_dtype=tl.float32)

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

            # Store output (C) with bounds checking
            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[
                None, :] * stride_cn
            tl.store(c_ptrs, accumulator.to(c_type), mask=mask_c)


# =============== Forward Wrapper for CGGEMM =================


@triton_op("torchtitan::mxfp4_grouped_gemm_forward", mutates_args={})
def mxfp4_grouped_gemm_forward(
    inputs: torch.Tensor,  # [M_bufferlen, K]
    expert_weights: torch.Tensor,  # [num_experts, N, K] if trans_weights
    expert_indices: torch.Tensor,  # [M_total]
    input_scales: torch.Tensor,
    weight_scales: torch.Tensor,
    trans_weights: bool = True,
    use_2dblock_x: bool = False,
    use_2dblock_w: bool = True,
    output_dtype: torch.dtype = torch.bfloat16,
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
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"

    # Check if inputs are properly aligned
    M_bufferlen, K = inputs.shape
    # assume packed along K dim
    K *= 2
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
    if trans_weights:
        num_experts, N, K_weights = expert_weights.shape
        stride_be, stride_bn, stride_bk = expert_weights.stride()
        stride_bse, stride_bsn, stride_bsk = weight_scales.stride()
    else:
        num_experts, K_weights, N = expert_weights.shape
        stride_be, stride_bk, stride_bn = expert_weights.stride()
        stride_bse, stride_bsk, stride_bsn = weight_scales.stride()
    # assume k_packed
    K_weights *= 2

    # Validate dimensions
    assert K == K_weights, f"Input K ({K}) must match weight K ({K_weights})"
    # assert (
    #     expert_indices.shape[0] == M_total
    # ), f"Expert indices length ({expert_indices.shape[0]}) must match M_total ({M_total})"

    # Create output tensor
    output = torch.zeros((M_bufferlen, N),
                         device=inputs.device,
                         dtype=output_dtype)

    stride_am, stride_ak = inputs.stride()
    stride_asm, stride_ask = input_scales.stride()
    stride_cm, stride_cn = output.stride()

    # Calculate grid size for the kernel
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    grid = (NUM_SMS, 1, 1)
    # Launch kernel

    wrap_triton(_kernel_mxfp4_grouped_gemm_forward)[grid](
        inputs,
        expert_weights,
        output,
        expert_indices,
        input_scales,
        weight_scales,
        stride_am,
        stride_ak,
        stride_be,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_cn,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsn,
        stride_bsk,
        M_TOTAL=M_total,
        N=N,
        K=K,
        NUM_EXPERTS=num_experts,
        GROUP_SIZE_M=ALIGN_SIZE_M,
        NUM_SMS=NUM_SMS,
        SUPER_GROUP_M=32,
        USE_2DBLOCK_A=use_2dblock_x,
        USE_2DBLOCK_B=use_2dblock_w,
        QUANT_BLOCK_SIZE=BLOCK_SIZE_DEFAULT,
    )

    return output
