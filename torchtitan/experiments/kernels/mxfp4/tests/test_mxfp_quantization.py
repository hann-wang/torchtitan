# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
from torchtitan.experiments.kernels.mxfp4.mxfp_quantization import (
    convert_to_mxfp4,
    convert_from_mxfp4,
)

from .utils import (
    prepare_data,
    convert_from_mxfp4_pytorch,
    convert_to_mxfp4_pytorch,
)

# Note: Python SegFault might be related to https://github.com/pytorch/pytorch/issues/125234


@pytest.mark.parametrize("tensor_shape", [(128, 64), (2048, 2048),
                                          (4, 128, 64)])
@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("is_2d_block", [False, True])
@pytest.mark.parametrize("data_type", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_sr", [False, True])
@pytest.mark.parametrize("use_asm", [False, True])
@pytest.mark.parametrize("compile", [False])
def test_mxfp_1d_quantization(tensor_shape, axis, is_2d_block, data_type,
                              use_sr, use_asm, compile):
    if use_sr and not use_asm:
        pytest.skip("Non-asm mode dose not support Stochastic Rounding.")

    if compile:
        quant_func = torch.compile(
            torch.ops.torchtitan.convert_to_mxfp4, fullgraph=True)
        dequant_func = torch.compile(
            torch.ops.torchtitan.convert_from_mxfp4, fullgraph=True)
    else:
        quant_func = torch.ops.torchtitan.convert_to_mxfp4
        dequant_func = torch.ops.torchtitan.convert_from_mxfp4

    x = prepare_data(tensor_shape, data_type)
    data_lp_ref, scales_ref = convert_to_mxfp4_pytorch(
        x,
        axis=axis,
        is_2d_block=is_2d_block,
    )
    x_dq_ref = convert_from_mxfp4_pytorch(
        data_lp_ref,
        scales_ref,
        output_dtype=data_type,
        axis=axis,
        is_2d_block=is_2d_block,
    )

    data_lp, scales = quant_func(
        x,
        axis=axis,
        is_2d_block=is_2d_block,
        use_sr=use_sr,
        use_asm=use_asm,
    )
    assert torch.all(scales_ref == scales).item()
    if not use_sr:
        assert torch.all(data_lp_ref == data_lp).item()
    else:
        data_lp_ref_lo = (data_lp_ref.to(torch.int8) & 0xF)
        data_lp_ref_hi = ((data_lp_ref.to(torch.int8) >> 4) & 0xF)
        data_lp_lo = (data_lp.to(torch.int8) & 0xF)
        data_lp_hi = ((data_lp.to(torch.int8) >> 4) & 0xF)
        assert torch.all(
            torch.max(torch.abs(data_lp_ref_lo - data_lp_lo)) <= 1).item()
        assert torch.all(
            torch.max(torch.abs(data_lp_ref_hi - data_lp_hi)) <= 1).item()

    x_dq = dequant_func(data_lp,
                        scales,
                        output_dtype=data_type,
                        axis=axis,
                        is_2d_block=is_2d_block,
                        use_asm=use_asm)

    if not use_sr:
        assert torch.allclose(x_dq_ref, x_dq)
    else:
        dq_mae = (x_dq_ref - x_dq).abs().mean().item()
        assert dq_mae < 1.0 if is_2d_block else 0.5
