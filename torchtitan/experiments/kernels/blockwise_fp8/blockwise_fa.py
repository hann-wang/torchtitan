# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional
import math
import torch

from .triton_flash_attention_fp8_block import triton_attention_block


def attention_forward_core_ref_impl(q, k, v, sm_scale, causal, use_exp2):
    original_dtype = q.dtype
    DEBUG = False
    if DEBUG:
        print()
        print("attention_forward_core_ref_impl")
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("sm_scale:", sm_scale)
        print("causal:", causal)
        print("use_exp2:", use_exp2)

    # Compute attention scores
    attention_scores = torch.matmul(q.to(torch.float32),
                                    k.transpose(-2, -1).to(torch.float32))
    if DEBUG:
        print("attention_scores:", attention_scores, attention_scores.shape)

    # Scale scores
    attention_scaled_scores = sm_scale * attention_scores
    if DEBUG:
        print("attention_scaled_scores:", attention_scaled_scores,
              attention_scaled_scores.shape)

    # Apply causal mask if necessary
    if causal:
        L_q, L_k = q.shape[1], k.shape[1]
        row_idx = torch.arange(L_q, device=q.device).unsqueeze(1)
        col_idx = torch.arange(L_k, device=q.device).unsqueeze(0)
        col_offset = L_q - L_k
        causal_mask = row_idx >= (col_offset + col_idx)
        if DEBUG:
            print("causal_mask:", causal_mask)
        # set -inf to places the causal mask is false
        attention_scaled_scores = attention_scaled_scores.masked_fill(
            torch.logical_not(causal_mask.unsqueeze(0)), float('-inf'))
        if DEBUG:
            print("attention_scaled_scores after causal:",
                  attention_scaled_scores, attention_scaled_scores.shape)

    # Compute max for numerical stability
    max_scores = torch.max(attention_scaled_scores, dim=-1, keepdim=True)[0]
    if DEBUG:
        print("max_scores:", max_scores, max_scores.shape)
    if causal:
        # Replace -inf in max_scores with zeros to avoid NaN in subtraction
        max_scores = torch.where(torch.isinf(max_scores),
                                 torch.zeros_like(max_scores), max_scores)
        if DEBUG:
            print("max_scores if causal:", max_scores, max_scores.shape)

    # Shift scores
    attention_shifted_scaled_scores = attention_scaled_scores - max_scores
    if DEBUG:
        print("attention_shifted_scaled_scores:",
              attention_shifted_scaled_scores,
              attention_shifted_scaled_scores.shape)

    # Exponentiate
    if use_exp2:
        RCP_LN = 1 / math.log(2)
        exp_scores = torch.exp2(RCP_LN * attention_shifted_scaled_scores)
    else:
        exp_scores = torch.exp(attention_shifted_scaled_scores)

    if DEBUG:
        print("exp_scores:", exp_scores, exp_scores.shape)

    # Sum of exponentials
    sum_exp_scores = torch.sum(exp_scores, dim=-1, keepdim=True)
    if DEBUG:
        print("sum_exp_scores:", sum_exp_scores, sum_exp_scores.shape)
    if causal:
        # if sum of exp scores is 0.0 it means scores where -inf, we cannot compute softmax and softmax_lse. Setting to 1 deals with -inf case cleanly
        sum_exp_scores = torch.where(sum_exp_scores == 0,
                                     torch.ones_like(sum_exp_scores),
                                     sum_exp_scores)
    if DEBUG:
        print("sum_exp_scores:", sum_exp_scores, sum_exp_scores.shape)

    # Compute softmax probabilities
    softmax = exp_scores / sum_exp_scores

    if DEBUG:
        print("softmax:", softmax, softmax.shape)

    # Compute log-sum-exp
    if use_exp2:
        LN2 = math.log(2)
        RCP_LN = 1 / math.log(2)
        max_scores_base2 = max_scores * RCP_LN
        softmax_lse_base2 = max_scores_base2 + torch.log2(sum_exp_scores)
        softmax_lse = softmax_lse_base2 * LN2
        softmax_lse.squeeze_(-1)
    else:
        softmax_lse = max_scores + torch.log(sum_exp_scores)
        softmax_lse = softmax_lse.squeeze(-1)

    if DEBUG:
        print("softmax_lse:", softmax_lse, softmax_lse.shape)

    # Compute output
    o = torch.matmul(softmax, v.to(torch.float32)).to(original_dtype)
    if DEBUG:
        print("o:", o, o.shape)

    return o, softmax_lse, exp_scores, softmax, attention_shifted_scaled_scores, attention_scaled_scores, attention_scores


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (torch.unsqueeze(x, dim=3).expand(bs, slen, n_kv_heads, n_rep,
                                             head_dim).reshape(
                                                 bs, slen, n_kv_heads * n_rep,
                                                 head_dim))


def attention_vanilla_forward_pytorch_ref_impl(q, k, v, sm_scale, causal,
                                               layout, use_exp2):
    """Compute reference output and softmax_lse using PyTorch's built-in function"""

    # Ensure the layout is 'bhsd'
    if layout == "bshd":
        num_heads = q.shape[2]
        n_kv_heads = k.shape[2]
        n_rep = num_heads // n_kv_heads
        # if n_rep > 1:
        #     k = repeat_kv(k, n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        #     v = repeat_kv(v, n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    elif layout != "bhsd":
        raise ValueError(f"Unknown layout {layout}")

    # # Prepare tensors in [batch_size * num_heads, seq_len, head_dim] format
    # batch_size, num_heads, seq_len_q, head_dim = q.shape
    # seq_len_k = k.shape[2]

    # # Merge batch and heads dimensions
    # q = q.reshape(batch_size * num_heads, seq_len_q, head_dim)
    # k = k.reshape(batch_size * num_heads, seq_len_k, head_dim)
    # v = v.reshape(batch_size * num_heads, seq_len_k, head_dim)

    # # Call the core attention function
    # o, softmax_lse, exp_scores, softmax, attention_shifted_scaled_scores, attention_scaled_scores, attention_scores = attention_forward_core_ref_impl(
    #     q, k, v, sm_scale, causal, use_exp2)

    # # Reshape outputs back to [batch_size, num_heads, seq_len, head_dim]
    # o = o.reshape(batch_size, num_heads, seq_len_q, head_dim)
    # softmax_lse = softmax_lse.reshape(batch_size, num_heads, seq_len_q)
    # exp_scores = exp_scores.reshape(batch_size, num_heads, seq_len_q,
    #                                 seq_len_k)
    # softmax = softmax.reshape(batch_size, num_heads, seq_len_q, seq_len_k)
    # attention_shifted_scaled_scores = attention_shifted_scaled_scores.reshape(
    #     batch_size, num_heads, seq_len_q, seq_len_k)
    # attention_scaled_scores = attention_scaled_scores.reshape(
    #     batch_size, num_heads, seq_len_q, seq_len_k)
    # attention_scores = attention_scores.reshape(batch_size, num_heads,
    #                                             seq_len_q, seq_len_k)

    o = torch.nn.functional.scaled_dot_product_attention(q,
                                                         k,
                                                         v,
                                                         is_causal=causal,
                                                         scale=sm_scale,
                                                         enable_gqa=n_rep > 1)

    softmax_lse = exp_scores = softmax = attention_shifted_scaled_scores = attention_scaled_scores = attention_scores = None

    # Restore original layout if necessary
    if layout == "bshd":
        o = o.transpose(1, 2)

    return o, softmax_lse, exp_scores, softmax, attention_shifted_scaled_scores, attention_scaled_scores, attention_scores


def flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    is_causal: bool,
    dropout: float = 0.0,
    softmax_scale: Optional[float] = None,
    use_fp8: bool = False,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`float`):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        use_top_left_mask (`bool`, defaults to `False`):
            flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
    """
    causal = is_causal
    assert query_states.shape[1] == key_states.shape[1]
    seqlen = query_states.shape[1]

    if softmax_scale is None:
        softmax_scale = 1. / math.sqrt(query_states.size(-1))

    # attn_output = attention_vanilla_forward_pytorch_ref_impl(
    #     query_states, key_states, value_states, softmax_scale, causal,
    #     "bshd", False)[0]

    query_states = query_states.contiguous()
    key_states = key_states.contiguous()
    value_states = value_states.contiguous()

    triton_results = triton_attention_block(
        query_states,
        key_states,
        value_states,
        # alibi_slopes
        None,
        # bias
        None,
        softmax_scale,
        dropout,
        # cu_seqlens_q
        0,
        # cu_seqlens_k
        0,
        # max_seqlens_q
        seqlen,
        # cu_seqlens_k
        seqlen,
        causal,
        # return_scores
        False,
        # use_exp2
        True,
        "bshd",
        use_fp8=use_fp8,
    )
    attn_output = triton_results[0]

    return attn_output
