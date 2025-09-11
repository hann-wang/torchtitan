import pytest
from tabulate import tabulate
import torch
from torchtitan.experiments.kernels.blockwise_fp4.mxfp_quantization import (
    convert_to_mxfp4_1dblock,
    convert_from_mxfp4_1dblock,
    convert_to_mxfp4_2dblock,
    convert_from_mxfp4_2dblock,
)
from torchtitan.experiments.kernels.blockwise_fp4.mxfp_linear import (
    blockwise_mxfp4_gemm,
    MXFP4LinearFunction,
)

from .utils import prepare_data, calc_cossim, calc_snr


@pytest.mark.parametrize("shape", [(256, 384, 128), (512, 1024, 256)])
@pytest.mark.parametrize("use_2dblock_a", [False, True])
@pytest.mark.parametrize("use_2dblock_b", [False, True])
@pytest.mark.parametrize("trans_a", [False, True])
@pytest.mark.parametrize("trans_b", [False, True])
@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("data_type", [torch.bfloat16, torch.float32])
def test_mxfp4_linear_kernel(shape, use_2dblock_a, use_2dblock_b, trans_a, trans_b,
                      contiguous, data_type):
    M, N, K = shape
    if trans_a:
        if contiguous:
            a = prepare_data((K, M), data_type)
        else:
            a = prepare_data((M, K), data_type).transpose(-1, -2)
            assert not a.is_contiguous()
        a_axis = -2
    else:
        if contiguous:
            a = prepare_data((M, K), data_type)
        else:
            a = prepare_data((K, M), data_type).transpose(-1, -2)
            assert not a.is_contiguous()
        a_axis = -1
    if trans_b:
        if contiguous:
            b = prepare_data((N, K), data_type)
        else:
            b = prepare_data((K, N), data_type).transpose(-1, -2)
            assert not b.is_contiguous()
        b_axis = -1
    else:
        if contiguous:
            b = prepare_data((K, N), data_type)
        else:
            b = prepare_data((N, K), data_type).transpose(-1, -2)
            assert not b.is_contiguous()
        b_axis = -2
    if use_2dblock_a:
        a_lp, a_scales = torch.ops.torchtitan.convert_to_mxfp4_2dblock(
            a, axis=a_axis)
        a_dq = torch.ops.torchtitan.convert_from_mxfp4_2dblock(
            a_lp, a_scales, output_dtype=data_type, axis=a_axis)
    else:
        a_lp, a_scales = torch.ops.torchtitan.convert_to_mxfp4_1dblock(
            a, axis=a_axis)
        a_dq = torch.ops.torchtitan.convert_from_mxfp4_1dblock(
            a_lp, a_scales, output_dtype=data_type, axis=a_axis)
    if use_2dblock_b:
        b_lp, b_scales = torch.ops.torchtitan.convert_to_mxfp4_2dblock(
            b, axis=b_axis)
        b_dq = torch.ops.torchtitan.convert_from_mxfp4_2dblock(
            b_lp, b_scales, output_dtype=data_type, axis=b_axis)
    else:
        b_lp, b_scales = torch.ops.torchtitan.convert_to_mxfp4_1dblock(
            b, axis=b_axis)
        b_dq = torch.ops.torchtitan.convert_from_mxfp4_1dblock(
            b_lp, b_scales, output_dtype=data_type, axis=b_axis)

    c = torch.ops.torchtitan.blockwise_mxfp4_gemm(
        a_lp,
        a_scales,
        b_lp,
        b_scales,
        trans_a=trans_a,
        trans_b=trans_b,
        use_2dblock_a=use_2dblock_a,
        use_2dblock_b=use_2dblock_b,
        output_dtype=data_type,
    )
    c_ref = (a_dq.T if trans_a else a_dq) @ (b_dq.T if trans_b else b_dq)

    assert torch.allclose(c_ref, c)


@pytest.mark.parametrize("shape", [(1, 512, 384, 128), (4, 1024, 1024, 2048)])
@pytest.mark.parametrize("use_2dblock_w", [False, True])
@pytest.mark.parametrize("use_2dblock_x", [False, True])
@pytest.mark.parametrize("use_grad_sr", [False, True])
@pytest.mark.parametrize("compile", [False])
@pytest.mark.parametrize("data_type", [torch.bfloat16, torch.float32])
def test_mxfp4_linear_autograd_function(shape, use_2dblock_x, use_2dblock_w,
                                        use_grad_sr, compile, data_type):
    B, M, N, K = shape
    inputs = prepare_data((B, M, K), data_type).requires_grad_(True)
    weights = prepare_data((N, K), data_type).requires_grad_(True)

    # Create a target for gradient computation
    target = prepare_data((B, M, N), data_type)

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
    mxfp4_linear_func = MXFP4LinearFunction.apply
    if compile:
        mxfp4_linear_func = torch.compile(mxfp4_linear_func, fullgraph=True)
    outputs = mxfp4_linear_func(inputs, weights, use_2dblock_x, use_2dblock_w,
                                use_grad_sr)
    loss = torch.nn.functional.mse_loss(outputs, target)
    loss.backward()

    output_snr = calc_snr(outputs, outputs_ref)
    output_sim = calc_cossim(outputs, outputs_ref)
    dx_snr = calc_snr(inputs.grad, grad_inputs_ref)
    dx_sim = calc_cossim(inputs.grad, grad_inputs_ref)
    dw_snr = calc_snr(weights.grad, grad_weights_ref)
    dw_sim = calc_cossim(weights.grad, grad_weights_ref)

    print()
    print(
        tabulate([
            ["O", output_snr, output_sim],
            ["dX", dx_snr, dx_sim],
            ["dW", dw_snr, dw_sim],
        ],
                 headers=["Tensor", "SNR", "Cosine Sim"],
                 tablefmt="github"))

    min_sr = 8 if use_grad_sr else 10
    assert output_snr > min_sr
    assert dx_snr > min_sr
    assert dw_snr > min_sr
