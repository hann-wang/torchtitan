import pytest
import torch
from torchtitan.experiments.kernels.blockwise_fp4.mxfp_quantization import (
    convert_to_mxfp4_1dblock,
    convert_from_mxfp4_1dblock,
    convert_to_mxfp4_2dblock,
    convert_from_mxfp4_2dblock,
)

from .utils import (
    prepare_data,
    convert_from_mxfp4_1dblock_pytorch,
    convert_from_mxfp4_2dblock_pytorch,
    convert_to_mxfp4_1dblock_pytorch,
    convert_to_mxfp4_2dblock_pytorch,
)

# Note: Python SegFault might be related to https://github.com/pytorch/pytorch/issues/125234


@pytest.mark.parametrize("tensor_shape", [(128, 64), (2048, 2048),
                                          (4, 128, 64)])
@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("data_type", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_asm", [False, True])
@pytest.mark.parametrize("compile", [False])
def test_mxfp_1d_quantization(tensor_shape, axis, data_type, use_asm, compile):

    if compile:
        quant_func = torch.compile(
            torch.ops.torchtitan.convert_to_mxfp4_1dblock, fullgraph=True)
        dequant_func = torch.compile(
            torch.ops.torchtitan.convert_from_mxfp4_1dblock, fullgraph=True)
    else:
        quant_func = torch.ops.torchtitan.convert_to_mxfp4_1dblock
        dequant_func = torch.ops.torchtitan.convert_from_mxfp4_1dblock

    x = prepare_data(tensor_shape, data_type)
    data_lp_ref, scales_ref = convert_to_mxfp4_1dblock_pytorch(x, axis=axis)
    x_dq_ref = convert_from_mxfp4_1dblock_pytorch(data_lp_ref,
                                                  scales_ref,
                                                  output_dtype=data_type,
                                                  axis=axis)

    data_lp, scales = quant_func(x, axis=axis, use_asm=use_asm)
    assert torch.all(scales_ref == scales).item()
    assert torch.all(data_lp_ref == data_lp).item()

    x_dq = dequant_func(data_lp,
                        scales,
                        output_dtype=data_type,
                        axis=axis,
                        use_asm=use_asm)
    assert torch.allclose(x_dq_ref, x_dq)


@pytest.mark.parametrize("tensor_shape", [(128, 64), (2048, 2048)])
@pytest.mark.parametrize("axis", [-1, 0])
@pytest.mark.parametrize("data_type", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_asm", [False, True])
@pytest.mark.parametrize("compile", [False])
def test_mxfp_2d_quantization(tensor_shape, axis, data_type, use_asm, compile):

    if compile:
        quant_func = torch.compile(
            torch.ops.torchtitan.convert_to_mxfp4_2dblock, fullgraph=True)
        dequant_func = torch.compile(
            torch.ops.torchtitan.convert_from_mxfp4_2dblock, fullgraph=True)
    else:
        quant_func = torch.ops.torchtitan.convert_to_mxfp4_2dblock
        dequant_func = torch.ops.torchtitan.convert_from_mxfp4_2dblock

    x = prepare_data(tensor_shape, data_type)
    data_lp_ref, scales_ref = convert_to_mxfp4_2dblock_pytorch(x, axis=axis)
    x_dq_ref = convert_from_mxfp4_2dblock_pytorch(data_lp_ref,
                                                  scales_ref,
                                                  output_dtype=data_type,
                                                  axis=axis)
    data_lp, scales = quant_func(x, axis=axis, use_asm=use_asm)

    assert torch.all(scales_ref == scales).item()
    assert torch.all(data_lp_ref == data_lp).item()

    x_dq = dequant_func(data_lp,
                        scales,
                        output_dtype=data_type,
                        axis=axis,
                        use_asm=use_asm)
    assert torch.allclose(x_dq_ref, x_dq)
