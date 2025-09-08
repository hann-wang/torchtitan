import pytest
from tabulate import tabulate
import torch

from torchtitan.experiments.kernels.blockwise_fp4.mxfp_grouped_gemm import (ALIGN_SIZE_M, mxfp4_grouped_gemm,)

from .utils import prepare_data, calc_cossim, calc_snr


@pytest.mark.parametrize("shape", [(1024, 512, 512, 8)])
@pytest.mark.parametrize("use_2dblock_x", [False, True])
@pytest.mark.parametrize("use_2dblock_w", [False, True])
@pytest.mark.parametrize("trans_weights", [False, True])
@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("data_type", [torch.bfloat16, torch.float32])
def test_mxfp_group_gemm(shape, use_2dblock_x, use_2dblock_w, trans_weights,
                         contiguous, data_type):
    M_total, N, K, num_experts = shape
    # Ensure M_total is a multiple of group_size_m
    M_total = (M_total // ALIGN_SIZE_M) * ALIGN_SIZE_M
    num_groups = M_total // ALIGN_SIZE_M

    # Create test tensors
    device = torch.device("cuda")

    if contiguous:
        inputs = prepare_data((M_total, K), data_type).requires_grad_(True)
    else:
        inputs = prepare_data((K, M_total), data_type).transpose(-1, -2).requires_grad_(True)
    if trans_weights:
        if contiguous:
            expert_weights = prepare_data((num_experts, N, K),
                                          data_type).requires_grad_(True)
        else:
            expert_weights = prepare_data(
                (num_experts, K, N),
                data_type).transpose(-1, -2).requires_grad_(True)
    else:
        if contiguous:
            expert_weights = prepare_data((num_experts, K, N),
                                          data_type).requires_grad_(True)
        else:
            expert_weights = prepare_data(
                (num_experts, N, K),
                data_type).transpose(-1, -2).requires_grad_(True)

    # Create expert indices - each token in a group has the same expert
    expert_indices = torch.zeros(
        M_total,
        dtype=torch.int32,
        device=device,
    )
    for g in range(num_groups):
        expert_idx = torch.randint(0,
                                   num_experts, (1, ),
                                   device=device,
                                   dtype=torch.int32).item()
        start_idx = g * ALIGN_SIZE_M
        end_idx = (g + 1) * ALIGN_SIZE_M
        expert_indices[start_idx:end_idx] = expert_idx

    # Create a target for gradient computation
    target = prepare_data((M_total, N), data_type)

    # PyTorch reference implementation
    outputs_ref = torch.zeros((M_total, N), dtype=data_type, device=device)
    for g in range(num_groups):
        group_start = g * ALIGN_SIZE_M
        group_end = (g + 1) * ALIGN_SIZE_M
        expert_idx = expert_indices[group_start].item()

        # Compute output for this group
        current_weight = expert_weights[expert_idx]
        if trans_weights:
            current_weight = current_weight.t()
        outputs_ref[group_start:group_end] = (
            inputs[group_start:group_end] @ current_weight)

    # Compute loss and gradients with PyTorch
    loss_ref = torch.nn.functional.mse_loss(outputs_ref, target)
    loss_ref.backward()

    grad_inputs_ref = inputs.grad.clone()
    grad_weights_ref = expert_weights.grad.clone()

    # Reset gradients
    inputs.grad.zero_()
    expert_weights.grad.zero_()

    # Compute with our implementation
    outputs = mxfp4_grouped_gemm(
        inputs,
        expert_weights,
        expert_indices,
        trans_weights=trans_weights,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
    )
    loss = torch.nn.functional.mse_loss(outputs, target)
    loss.backward()

    # # Check if outputs match
    # print(f"outputs={outputs}")
    # print(f"outputs_ref={outputs_ref}")

    output_snr = calc_snr(outputs, outputs_ref)
    output_sim = calc_cossim(outputs, outputs_ref)
    dx_snr = calc_snr(inputs.grad, grad_inputs_ref)
    dx_sim = calc_cossim(inputs.grad, grad_inputs_ref)
    dw_snr = calc_snr(expert_weights.grad, grad_weights_ref)
    dw_sim = calc_cossim(expert_weights.grad, grad_weights_ref)

    print()
    print(
        tabulate([
            ["O", output_snr, output_sim],
            ["dX", dx_snr, dx_sim],
            ["dW", dw_snr, dw_sim],
        ],
                 headers=["Tensor", "SNR", "Cosine Sim"],
                 tablefmt="github"))

    assert output_snr > 10
    assert dx_snr > 10
    assert dw_snr > 10
