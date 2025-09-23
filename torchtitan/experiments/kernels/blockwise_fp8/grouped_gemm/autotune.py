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
import triton.language as tl


# ===== Supporting utils, CUDA and TMA =====

ALIGN_SIZE_M = 128
AUTOTUNE = False

# ================== End of supporting functions ==================
if AUTOTUNE:
    STANDARD_CONFIGS = [
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": ALIGN_SIZE_M,
                "BLOCK_SIZE_K": ALIGN_SIZE_M,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": ALIGN_SIZE_M,
                "BLOCK_SIZE_K": ALIGN_SIZE_M,
            },
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": ALIGN_SIZE_M,
                "BLOCK_SIZE_K": ALIGN_SIZE_M,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": ALIGN_SIZE_M,
                "BLOCK_SIZE_K": ALIGN_SIZE_M,
            },
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": ALIGN_SIZE_M,
                "BLOCK_SIZE_K": ALIGN_SIZE_M,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": ALIGN_SIZE_M,
                "BLOCK_SIZE_K": ALIGN_SIZE_M,
            },
            num_stages=2,
            num_warps=8,
        ),
    ]

    STANDARD_DW_CONFIGS = [
        triton.Config(
            {
                "BLOCK_SIZE_M": ALIGN_SIZE_M,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": ALIGN_SIZE_M,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
            },
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": ALIGN_SIZE_M,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": ALIGN_SIZE_M,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
            },
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": ALIGN_SIZE_M,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": ALIGN_SIZE_M,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
            },
            num_stages=2,
            num_warps=8,
        ),
    ]
else:
    STANDARD_CONFIGS = [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": ALIGN_SIZE_M,
                "BLOCK_SIZE_K": ALIGN_SIZE_M,
            },
            num_stages=2,
            num_warps=4,
        ),
    ]
    STANDARD_DW_CONFIGS = [
        triton.Config(
            {
                "BLOCK_SIZE_M": ALIGN_SIZE_M,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
            },
            num_stages=2,
            num_warps=4,
        ),
    ]
