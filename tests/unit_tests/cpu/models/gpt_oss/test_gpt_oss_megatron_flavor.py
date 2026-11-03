# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import pytest
import torch
import torch.nn as nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.activation import ClampedSwiGLU, SwiGLU
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.gpt_oss import build_model_config
from torchtitan.models.gpt_oss.moe import GptOssGroupedLinear


def _config(flavor: str):
    return build_model_config(
        flavor,
        seq_len=128,
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            tensor_parallel_degree=1,
        ),
    )


def test_20b_keeps_biased_clamped_experts():
    moe = _config("20b").layers[0].feed_forward
    experts = moe.routed_experts
    assert type(experts.w13) is GptOssGroupedLinear.Config
    assert type(experts.w2) is GptOssGroupedLinear.Config
    assert type(experts.activation_fn) is ClampedSwiGLU.Config
    assert moe.router.aux_loss is not None
    attn = _config("20b").layers[0].attention
    assert attn.qkv_linear.wqkv.bias is True
    assert attn.wo.bias is True
    assert moe.router.gate.bias is True


def test_20b_megatron_uses_deepseek_experts_and_megatron_init():
    config = _config("20b_megatron")
    layer = config.layers[0]
    experts = layer.feed_forward.routed_experts
    assert type(experts.w13) is GroupedLinear.Config
    assert type(experts.w2) is GroupedLinear.Config
    assert type(experts.activation_fn) is SwiGLU.Config
    assert layer.feed_forward.load_balance_coeff is None
    assert layer.feed_forward.router.aux_loss is None
    assert layer.attention.qkv_linear.wqkv.bias is False
    assert layer.attention.wo.bias is False
    assert layer.feed_forward.router.gate.bias is False
    assert config.lm_head.bias is False
    assert "bias" not in layer.attention.qkv_linear.wqkv.param_init
    assert "bias" not in layer.attention.wo.param_init
    assert "bias" not in layer.feed_forward.router.gate.param_init

    init_std = 0.008
    output_std = init_std / math.sqrt(2.0 * len(config.layers))
    assert config.tok_embeddings.param_init["weight"].func is nn.init.normal_
    assert config.tok_embeddings.param_init["weight"].keywords["std"] == init_std
    assert config.lm_head.param_init["weight"].keywords["std"] == init_std
    assert layer.attention.qkv_linear.wqkv.param_init["weight"].keywords["std"] == init_std
    assert (
        layer.attention.wo.param_init["weight"].keywords["std"]
        == output_std
    )
    assert layer.attention.param_init["sinks"].keywords["std"] == init_std
    assert (
        layer.feed_forward.router.gate.param_init["weight"].keywords["std"] == init_std
    )
    assert experts.w2.param_init["weight"].keywords["std"] == output_std

    gate_up = torch.empty(4, 2, 32, 16)
    experts.w13.param_init["weight"](gate_up)
    assert gate_up[:, 0].std().item() == pytest.approx(init_std, rel=0.05, abs=1e-3)
    assert gate_up[:, 1].std().item() == pytest.approx(init_std, rel=0.05, abs=1e-3)
