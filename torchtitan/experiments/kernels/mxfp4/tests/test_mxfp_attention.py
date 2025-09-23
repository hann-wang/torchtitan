# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pytest
from tabulate import tabulate
import torch

from torchtitan.experiments.kernels.mxfp4.triton_flash_attention_mxfp4 import (
    triton_attention_mxfp4,
)
from .utils import calc_snr, calc_cossim


def attention_vanilla_forward_pytorch_ref_impl(q,
                                               k,
                                               v,
                                               sm_scale,
                                               causal,
                                               layout="bshd"):
    """Compute reference output and softmax_lse using PyTorch's built-in function"""

    if layout == "bshd":
        num_heads = q.shape[2]
        n_kv_heads = k.shape[2]
        n_rep = num_heads // n_kv_heads

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    else:
        raise ValueError(f"Unknown layout {layout}")

    o_ref = torch.nn.functional.scaled_dot_product_attention(q,
                                                             k,
                                                             v,
                                                             is_causal=causal,
                                                             scale=sm_scale,
                                                             enable_gqa=n_rep
                                                             > 1)
    if layout == "bshd":
        o_ref = o_ref.transpose(1, 2)
    return o_ref


class AttnConfig:

    def __init__(self, seqlen_q, seqlen_kv, num_head_q, num_head_kv,
                 head_dim_qk, head_dim_v):
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.num_head_q = num_head_q
        self.num_head_kv = num_head_kv
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_v


test_cases = [
    AttnConfig(seqlen_q=1024,
               seqlen_kv=1024,
               num_head_q=32,
               num_head_kv=32,
               head_dim_qk=128,
               head_dim_v=128),
    AttnConfig(seqlen_q=1024,
               seqlen_kv=1024,
               num_head_q=64,
               num_head_kv=8,
               head_dim_qk=128,
               head_dim_v=128),
    AttnConfig(seqlen_q=1024,
               seqlen_kv=1024,
               num_head_q=32,
               num_head_kv=8,
               head_dim_qk=128,
               head_dim_v=128),
    AttnConfig(seqlen_q=1024,
               seqlen_kv=1024,
               num_head_q=64,
               num_head_kv=8,
               head_dim_qk=128,
               head_dim_v=128),
    AttnConfig(seqlen_q=1024,
               seqlen_kv=1024,
               num_head_q=16,
               num_head_kv=16,
               head_dim_qk=192,
               head_dim_v=128),
    AttnConfig(seqlen_q=1024,
               seqlen_kv=1024,
               num_head_q=128,
               num_head_kv=128,
               head_dim_qk=192,
               head_dim_v=128),
    AttnConfig(seqlen_q=1024,
               seqlen_kv=1024,
               num_head_q=32,
               num_head_kv=8,
               head_dim_qk=128,
               head_dim_v=128),
    AttnConfig(seqlen_q=1024,
               seqlen_kv=1024,
               num_head_q=48,
               num_head_kv=8,
               head_dim_qk=128,
               head_dim_v=128),
    # begin regression tests for SWDEV-548136
    # this will fail because currently tl.dot_scaled does not support k<64
    # AttnConfig(seqlen_q=4096 + 64,
    #            seqlen_kv=4096 + 64,
    #            num_head_q=2,
    #            num_head_kv=1,
    #            head_dim_qk=32,
    #            head_dim_v=32),
    AttnConfig(seqlen_q=2048,
               seqlen_kv=2048,
               num_head_q=64,
               num_head_kv=8,
               head_dim_qk=128,
               head_dim_v=128),
    # end regression tests for SWDEV-548136
]


@pytest.mark.parametrize("batch", [4])
@pytest.mark.parametrize("config", test_cases)
@pytest.mark.parametrize("causal", [True])
def test_attention(batch, config, causal):
    device = "cuda"
    dtype = torch.bfloat16
    seqlen_q, seqlen_kv, num_head_q, num_head_kv, head_dim_qk, head_dim_v = (
        config.seqlen_q,
        config.seqlen_kv,
        config.num_head_q,
        config.num_head_kv,
        config.head_dim_qk,
        config.head_dim_v,
    )
    q_layout = (batch, seqlen_q, num_head_q, head_dim_qk)
    k_layout = (batch, seqlen_kv, num_head_kv, head_dim_qk)
    v_layout = (batch, seqlen_kv, num_head_kv, head_dim_v)

    torch.manual_seed(1234)

    query = torch.randn(q_layout,
                        device=device,
                        dtype=dtype,
                        requires_grad=True)
    key = torch.randn(k_layout, device=device, dtype=dtype, requires_grad=True)
    value = torch.randn(v_layout,
                        device=device,
                        dtype=dtype,
                        requires_grad=True)
    query_ref = query.clone().detach().requires_grad_()
    key_ref = key.clone().detach().requires_grad_()
    value_ref = value.clone().detach().requires_grad_()

    sm_scale = query.shape[-1]**(-0.5)
    o_ref = attention_vanilla_forward_pytorch_ref_impl(query_ref, key_ref,
                                                       value_ref, sm_scale,
                                                       causal)
    loss_ref = o_ref.mean()
    loss_ref.backward()
    o = triton_attention_mxfp4(
        query.transpose(1, 2).contiguous(),
        key.transpose(1, 2).contiguous(),
        value.transpose(1, 2).contiguous(),
        bias=None,
        alibi_slopes=None,
        sm_scale=sm_scale,
        dropout_p=0.0,
        cu_seqlens_q=0,
        cu_seqlens_k=0,
        max_seqlens_q=seqlen_q,
        max_seqlens_k=seqlen_kv,
        causal=causal,
        return_scores=False,
        use_exp2=True,
        layout="bhsd",
    )[0].transpose(1, 2)

    # loss = o.mean()
    # loss.backward()

    output_snr = calc_snr(o_ref, o)
    output_sim = calc_cossim(o_ref, o)
    print()
    print(
        tabulate([
            ["O", output_snr, output_sim],
        ],
                 headers=["Tensor", "SNR", "Cosine Sim"],
                 tablefmt="github"))
    # query_grad_snr = compute_snr(query_ref.grad, query.grad)
    # key_grad_snr = compute_snr(key_ref.grad, key.grad)
    # value_grad_snr = compute_snr(value_ref.grad, value.grad)
    # print(out_snr, query_grad_snr, key_grad_snr, value_grad_snr)
    # assert out_snr > 20, "out_snr too low"
    # if len(test_cases) > 9 and config == test_cases[9]:
    #     # lower the SNR threshold for this specific case
    #     assert query_grad_snr > 12, "query_grad_snr too low"
    # else:
    #     assert query_grad_snr > 15, "query_grad_snr too low"
    # assert key_grad_snr > 15, "key_grad_snr too low"
    # assert value_grad_snr > 15, "value_grad_snr too low"


# @pytest.mark.parametrize("batch", [4])
# @pytest.mark.parametrize("config", test_cases)
# @pytest.mark.parametrize("causal", [True, False])
# def test_attention_fp8_with_sparse_do(batch, config, causal):
#     # regression test for SWDEV-548136
#     device = torch.device("cuda")
#     torch.manual_seed(1234)

#     dtype = torch.bfloat16
#     seqlen_q, seqlen_kv, num_head_q, num_head_kv, head_dim_qk, head_dim_v = (
#         config.seqlen_q,
#         config.seqlen_kv,
#         config.num_head_q,
#         config.num_head_kv,
#         config.head_dim_qk,
#         config.head_dim_v,
#     )
#     q_shape = (batch, seqlen_q, num_head_q, head_dim_qk)
#     k_shape = (batch, seqlen_kv, num_head_kv, head_dim_qk)
#     v_shape = (batch, seqlen_kv, num_head_kv, head_dim_v)
#     do_shape = (batch, seqlen_q, num_head_q, head_dim_v)

#     do = torch.randn(do_shape, device=device, dtype=dtype) * 1e-3
#     do_mask_0 = (torch.randn(do_shape[:-2], device=device, dtype=dtype)
#                  > 0.9).unsqueeze(-1).unsqueeze(-1)
#     do_mask_1 = (torch.randn(do_shape[:-1], device=device, dtype=dtype)
#                  > 0.9).unsqueeze(-1)
#     do = do * do_mask_0 * do_mask_1

#     q = torch.randn(q_shape, device=device, dtype=dtype)
#     k = torch.randn(k_shape, device=device, dtype=dtype)
#     v = torch.randn(v_shape, device=device, dtype=dtype)

#     sm_scale = q.shape[-1]**-0.5

#     q_fp8, q_descale = block_scaling_node(q)
#     k_fp8, k_descale = block_scaling_node(k)
#     v_fp8, v_descale = block_scaling_node(v)
#     p_scale = F8_FWD_MAX

#     o, softmax_lse, _ = attention_block_forward_triton_impl(
#         q_fp8,
#         k_fp8,
#         v_fp8,
#         q_descale,
#         k_descale,
#         v_descale=v_descale,
#         p_scale=p_scale,
#         sm_scale=sm_scale,
#         alibi_slopes=None,
#         causal=causal,
#         bias=None,
#         dropout_p=0.0,
#         layout="bshd",
#         cu_seqlens_q=0,
#         cu_seqlens_k=0,
#         max_seqlens_q=seqlen_q,
#         max_seqlens_k=seqlen_kv,
#         return_scores=False,
#         use_exp2=True,
#         use_fp8=True,
#     )

#     dq, dk, dv = attention_block_backward_triton_impl(
#         do,
#         q,
#         k,
#         v,
#         o,
#         softmax_lse.clone(),
#         torch.scalar_tensor(1.0, device=device),
#         torch.scalar_tensor(1.0, device=device),
#         torch.scalar_tensor(1.0, device=device),
#         p_scale=1.0,
#         sm_scale=sm_scale,
#         alibi_slopes=None,
#         causal=causal,
#         layout="bshd",
#         cu_seqlens_q=0,
#         cu_seqlens_k=0,
#         max_seqlen_q=seqlen_q,
#         max_seqlen_k=seqlen_kv,
#         use_exp2=True,
#         use_fp8=False,
#     )

#     dq_fp8, dk_fp8, dv_fp8 = attention_block_backward_triton_impl(
#         do,
#         q_fp8,
#         k_fp8,
#         v_fp8,
#         o,
#         softmax_lse,
#         q_descale,
#         k_descale,
#         v_descale,
#         p_scale=p_scale,
#         sm_scale=sm_scale,
#         alibi_slopes=None,
#         causal=causal,
#         layout="bshd",
#         cu_seqlens_q=0,
#         cu_seqlens_k=0,
#         max_seqlen_q=seqlen_q,
#         max_seqlen_k=seqlen_kv,
#         use_exp2=True,
#         use_fp8=True,
#     )

#     dq_snr = compute_snr(dq, dq_fp8)
#     dk_snr = compute_snr(dk, dk_fp8)
#     dv_snr = compute_snr(dv, dv_fp8)
#     print(f"dq_snr: {dq_snr}, dk_snr: {dk_snr}, dv_snr: {dv_snr}")
#     assert dq_snr > 15, "query_grad_snr too low"
#     assert dk_snr > 15, "key_grad_snr too low"
#     assert dv_snr > 15, "value_grad_snr too low"
