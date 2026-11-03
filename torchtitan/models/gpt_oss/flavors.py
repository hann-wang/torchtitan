# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPT-OSS model flavors."""

import dataclasses
import math
from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    ModelConfigConverter,
    validate_converter_compatibility,
)
from torchtitan.models.common import (
    BatchWiseLoadBalanceLoss,
    ColumnParallelLinear,
    CosSinRoPE,
    Embedding,
    HiMidLoLinear,
    Linear,
    RMSNorm,
    RoPE,
    RowParallelLinear,
    Softmax,
    TransformerBlock,
)
from torchtitan.models.common.activation import ClampedSwiGLU
from torchtitan.models.common.attention import (
    FlexInnerAttention,
    QKVLinear,
    SlidingWindowFlexInnerAttention,
    SlidingWindowVarlenInnerAttention,
    VarlenInnerAttention,
)
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_routed_experts_config,
)
from torchtitan.models.common.decoder_sharding import (
    colwise_sharding_config,
    inner_attention_local_spmd_sharding_config,
    lm_head_sharding_config,
    norm_sharding_config,
    pre_lm_head_norm_sharding_config,
    residual_activation_placement,
    rope_cache_sharding_config,
    rowwise_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.common.moe import MoE, RoutedExperts, TokenChoiceTopKRouter
from torchtitan.models.common.moe.sharding import (
    moe_sharding_config,
    routed_experts_sharding_config,
    router_gate_sharding_config,
    router_sharding_config,
)
from torchtitan.models.common.moe.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.models.common.param_init import depth_scaled_std
from .model import GptOssAttention, GptOssModel, GptOssTransformerBlock
from .moe import GptOssGroupedLinear
from .sharding import attention_sharding_config, routed_expert_linear_sharding_config

__all__ = [
    "GptOssModel",
    "MODEL_FLAVORS",
    "build_model_config",
]

_NORM_INIT = {"weight": nn.init.ones_}
# GPT-OSS uses std=0.02 for embeddings (model-specific)
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=0.02)}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


# Primus GPT-OSS pretrain sets this; Megatron's own default is 0.02.
_MEGATRON_INIT_METHOD_STD = 0.008


def _megatron_linear_init(std: float) -> dict[str, Callable]:
    """Megatron ``init_method_normal``: N(0, std), with linear bias left at 0."""
    return {
        "weight": partial(nn.init.normal_, mean=0.0, std=std),
        "bias": nn.init.zeros_,
    }


def _megatron_output_std(init_method_std: float, n_layers: int) -> float:
    """``init_method_std / sqrt(2 * num_layers)``, Megatron's output-layer std."""
    return init_method_std / math.sqrt(2.0 * n_layers)


def _make_gptoss_attn_config(
    *,
    dim: int,
    layer_id: int,
    attn_backend: str = "varlen",
    n_heads: int = 64,
    n_kv_heads: int = 8,
    head_dim: int = 64,
    sliding_window_size: int | None = None,
    rope: RoPE.Config,
    enable_sp: bool,
    megatron_init_std: float | None = None,
    n_layers: int | None = None,
    linear_bias: bool = True,
) -> GptOssAttention.Config:
    """Build a fully-specified, sharded GptOssAttention.Config for a layer.

    All linear sub-configs have their in_features/out_features set.
    The default init is depth-scaled, including QKV, the output projection,
    and sinks. ``megatron_init_std`` switches those to Megatron's normal init.
    """

    inner_attention = get_attention_config(attn_backend)

    if sliding_window_size is not None and isinstance(
        inner_attention, VarlenInnerAttention.Config
    ):
        inner_attention = SlidingWindowVarlenInnerAttention.Config(
            **{
                f.name: getattr(inner_attention, f.name)
                for f in dataclasses.fields(inner_attention)
            },
            sliding_window_size=sliding_window_size,
        )
    elif sliding_window_size is not None and isinstance(
        inner_attention, FlexInnerAttention.Config
    ):
        inner_attention = SlidingWindowFlexInnerAttention.Config(
            block_size=inner_attention.block_size,
            kernel_options=inner_attention.kernel_options,
            sliding_window_size=sliding_window_size,
        )
    inner_attention = dataclasses.replace(
        inner_attention, sharding_config=inner_attention_local_spmd_sharding_config()
    )
    x_layout = residual_activation_placement(enable_sp=enable_sp)

    if megatron_init_std is None:
        qkv_init = _depth_init(layer_id)
        wo_init = _depth_init(layer_id)
        sinks_init = {
            "sinks": partial(
                nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)
            )
        }
    else:
        if n_layers is None:
            raise ValueError("n_layers is required when megatron_init_std is set")
        # QKV uses init_method. The attention output projection uses
        # output_layer_init_method. Learnable sinks use init_method, matching
        # Megatron's learnable softmax offset.
        qkv_init = _megatron_linear_init(megatron_init_std)
        wo_init = _megatron_linear_init(
            _megatron_output_std(megatron_init_std, n_layers)
        )
        sinks_init = {
            "sinks": partial(nn.init.normal_, mean=0.0, std=megatron_init_std)
        }
    if not linear_bias:
        qkv_init = {"weight": qkv_init["weight"]}
        wo_init = {"weight": wo_init["weight"]}

    qkv = QKVLinear.Config(
        head_dim=head_dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        wqkv=ColumnParallelLinear.Config(
            in_features=dim,
            out_features=(n_heads + 2 * n_kv_heads) * head_dim,
            bias=linear_bias,
            param_init=qkv_init,
            sharding_config=colwise_sharding_config(input_layout=x_layout),
        ),
    )

    return GptOssAttention.Config(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        dim=dim,
        qkv_linear=qkv,
        wo=RowParallelLinear.Config(
            in_features=n_heads * head_dim,
            out_features=dim,
            bias=linear_bias,
            param_init=wo_init,
            sharding_config=rowwise_sharding_config(output_layout=x_layout),
        ),
        sliding_window_size=sliding_window_size,
        inner_attention=inner_attention,
        param_init=sinks_init,
        rope=dataclasses.replace(rope, sharding_config=rope_cache_sharding_config()),
        sharding_config=attention_sharding_config(enable_sp=enable_sp),
    )


def _make_gptoss_experts_config(
    *,
    dim: int,
    hidden_dim: int,
    num_experts: int,
    layer_id: int,
    top_k: int,
    enable_ep: bool,
    deepseek_v3_experts: bool = False,
    megatron_init_std: float | None = None,
    n_layers: int | None = None,
) -> RoutedExperts.Config:
    """Build a fully-specified, sharded RoutedExperts.Config for one layer."""
    if deepseek_v3_experts:
        # Bias-free grouped gate/up and down projections with SwiGLU, the same
        # expert module DeepSeek-V3 uses. Megatron initializes fc1 with
        # init_method and fc2 with output_layer_init_method.
        if megatron_init_std is None:
            std = depth_scaled_std(0.02, layer_id)
            w1_init = partial(nn.init.trunc_normal_, std=std)
            w2_init = w1_init
        else:
            if n_layers is None:
                raise ValueError("n_layers is required when megatron_init_std is set")
            w1_init = partial(nn.init.normal_, mean=0.0, std=megatron_init_std)
            w2_init = partial(
                nn.init.normal_,
                mean=0.0,
                std=_megatron_output_std(megatron_init_std, n_layers),
            )
        return make_routed_experts_config(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            param_init={"w1_EFD": w1_init, "w2_EDF": w2_init, "w3_EFD": w1_init},
            enable_ep=enable_ep,
        )

    # TODO: Audit GPT-OSS gate/up depth scaling separately, including its biases.
    std = depth_scaled_std(0.02, layer_id)
    experts_init = {
        "weight": partial(nn.init.trunc_normal_, std=std),
        "bias": partial(nn.init.trunc_normal_, std=std),
    }
    return RoutedExperts.Config(
        w13=GptOssGroupedLinear.Config(
            group_size=num_experts,
            in_features=dim,
            out_features=hidden_dim,
            num_linears=2,
            param_init=experts_init,
            sharding_config=routed_expert_linear_sharding_config(enable_ep=enable_ep),
        ),
        w2=GptOssGroupedLinear.Config(
            group_size=num_experts,
            in_features=hidden_dim,
            out_features=dim,
            param_init=experts_init,
            sharding_config=routed_expert_linear_sharding_config(enable_ep=enable_ep),
        ),
        activation_fn=ClampedSwiGLU.Config(),
        token_dispatcher=AllToAllTokenDispatcher.Config(
            num_experts=num_experts,
            top_k=top_k,
        ),
        sharding_config=routed_experts_sharding_config(),
    )


def _build_gptoss_layers(
    *,
    dim: int,
    n_layers: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    load_balance_coeff: float | None,
    attn_backend: str = "varlen",
    rope: RoPE.Config,
    enable_sp: bool,
    enable_ep: bool,
    deepseek_v3_experts: bool = False,
    megatron_init_std: float | None = None,
    aux_loss: bool = True,
    linear_bias: bool = True,
) -> list[TransformerBlock.Config]:
    """Build sharded per-layer configs for GPT-OSS.

    Even-indexed layers (0, 2, 4, ...) use sliding window attention.
    All dimensional fields are set directly.
    ``aux_loss`` adds the batch-wise load-balance loss. ``load_balance_coeff``
    is the auxiliary-loss-free expert bias; ``None`` leaves that bias out.
    """
    norm_sharding = norm_sharding_config(enable_sp=enable_sp)
    layers = []
    for layer_id in range(n_layers):
        attn_cfg = _make_gptoss_attn_config(
            dim=dim,
            layer_id=layer_id,
            attn_backend=attn_backend,
            sliding_window_size=128 if layer_id % 2 == 0 else None,
            rope=rope,
            enable_sp=enable_sp,
            megatron_init_std=megatron_init_std,
            n_layers=n_layers,
            linear_bias=linear_bias,
        )
        routed_experts_cfg = _make_gptoss_experts_config(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            layer_id=layer_id,
            top_k=top_k,
            enable_ep=enable_ep,
            deepseek_v3_experts=deepseek_v3_experts,
            megatron_init_std=megatron_init_std,
            n_layers=n_layers,
        )
        if megatron_init_std is None:
            router_init = _depth_init(layer_id)
        else:
            router_init = _megatron_linear_init(megatron_init_std)
        if not linear_bias:
            router_init = {"weight": router_init["weight"]}
        aux_loss_cfg = (
            BatchWiseLoadBalanceLoss.Config(coeff=1e-3, num_experts=num_experts)
            if aux_loss
            else None
        )
        moe_cfg = MoE.Config(
            num_experts=num_experts,
            load_balance_coeff=load_balance_coeff,
            routed_experts=routed_experts_cfg,
            router=TokenChoiceTopKRouter.Config(
                num_experts=num_experts,
                score_func=Softmax.Config(),
                route_norm=True,
                gate=HiMidLoLinear.Config(
                    in_features=dim,
                    out_features=num_experts,
                    backward_mode="hi_mid_lo",
                    bias=linear_bias,
                    param_init=router_init,
                    sharding_config=router_gate_sharding_config(),
                ),
                top_k=top_k,
                aux_loss=aux_loss_cfg,
                sharding_config=router_sharding_config(),
            ),
            sharding_config=moe_sharding_config(enable_sp=enable_sp),
        )
        layer_cfg = GptOssTransformerBlock.Config(
            attention=attn_cfg,
            attention_norm=RMSNorm.Config(
                normalized_shape=dim,
                param_init=_NORM_INIT,
                sharding_config=norm_sharding,
            ),
            ffn_norm=RMSNorm.Config(
                normalized_shape=dim,
                param_init=_NORM_INIT,
                sharding_config=norm_sharding,
            ),
            feed_forward=moe_cfg,
        )
        layers.append(layer_cfg)
    return layers


def _debugmodel(
    attn_backend: str = "varlen",
    *,
    seq_len: int,
    parallelism: ParallelismConfig,
) -> GptOssModel.Config:
    dim = 256
    hidden_dim = 2880
    n_layers = 4
    return GptOssModel.Config(
        max_context_length=seq_len,
        vocab_size=2048,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=2048,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(
                enable_sp=parallelism.enable_sequence_parallel
            ),
        ),
        norm=RMSNorm.Config(
            normalized_shape=dim,
            param_init=_NORM_INIT,
            sharding_config=pre_lm_head_norm_sharding_config(
                enable_sp=parallelism.enable_sequence_parallel
            ),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=2048,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_gptoss_layers(
            dim=dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            num_experts=8,
            top_k=4,
            load_balance_coeff=None,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=64,
                max_context_length=seq_len,
                theta=150000.0,
                scaling="yarn",
                rope_factor=32,
                beta_fast=32.0,
                beta_slow=1.0,
                truncate=False,
                original_seq_len=4096,
            ),
            enable_sp=parallelism.enable_sequence_parallel,
            enable_ep=parallelism.expert_parallel_degree > 1,
        ),
    )


def _20b_megatron(
    attn_backend: str = "varlen",
    *,
    seq_len: int,
    parallelism: ParallelismConfig,
) -> GptOssModel.Config:
    """GPT-OSS 20B with Primus Megatron experts and initialization.

    Experts are the bias-free SwiGLU grouped experts used by DeepSeek-V3.
    QKV, the attention output projection, the router, the experts, and the
    vocabulary projection have no bias, matching Primus ``add_bias_linear``
    and ``add_qkv_bias`` both being false. The router has neither an
    auxiliary load-balance loss nor the
    auxiliary-loss-free expert bias. Weights use Megatron's three normal
    inits: ``init_method`` (std
    ``_MEGATRON_INIT_METHOD_STD``) for embeddings, QKV, the router, sinks, and
    expert fc1; ``output_layer_init_method`` (that std divided by
    ``sqrt(2 * num_layers)``) for the attention output projection and expert
    fc2. The vocabulary projection uses ``init_method``, matching Megatron's
    GPT output layer when MuP is off.
    """
    dim = 2880
    hidden_dim = 2880
    n_layers = 24
    init_std = _MEGATRON_INIT_METHOD_STD
    return GptOssModel.Config(
        max_context_length=seq_len,
        dim=dim,
        vocab_size=201088,
        tok_embeddings=Embedding.Config(
            num_embeddings=201088,
            embedding_dim=dim,
            param_init={"weight": partial(nn.init.normal_, mean=0.0, std=init_std)},
            sharding_config=token_embedding_sharding_config(
                enable_sp=parallelism.enable_sequence_parallel
            ),
        ),
        norm=RMSNorm.Config(
            normalized_shape=dim,
            param_init=_NORM_INIT,
            sharding_config=pre_lm_head_norm_sharding_config(
                enable_sp=parallelism.enable_sequence_parallel
            ),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=201088,
            param_init={"weight": partial(nn.init.normal_, mean=0.0, std=init_std)},
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_gptoss_layers(
            dim=dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            num_experts=32,
            top_k=4,
            load_balance_coeff=None,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=64,
                max_context_length=seq_len,
                theta=150000.0,
                scaling="yarn",
                rope_factor=32,
                beta_fast=32.0,
                beta_slow=1.0,
                truncate=False,
                original_seq_len=4096,
            ),
            enable_sp=parallelism.enable_sequence_parallel,
            enable_ep=parallelism.expert_parallel_degree > 1,
            deepseek_v3_experts=True,
            megatron_init_std=init_std,
            aux_loss=False,
            linear_bias=False,
        ),
    )


def _20b(
    attn_backend: str = "varlen",
    *,
    seq_len: int,
    parallelism: ParallelismConfig,
) -> GptOssModel.Config:
    dim = 2880
    hidden_dim = 2880
    n_layers = 24
    return GptOssModel.Config(
        max_context_length=seq_len,
        dim=dim,
        vocab_size=201088,
        tok_embeddings=Embedding.Config(
            num_embeddings=201088,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(
                enable_sp=parallelism.enable_sequence_parallel
            ),
        ),
        norm=RMSNorm.Config(
            normalized_shape=dim,
            param_init=_NORM_INIT,
            sharding_config=pre_lm_head_norm_sharding_config(
                enable_sp=parallelism.enable_sequence_parallel
            ),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=201088,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_gptoss_layers(
            dim=dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            num_experts=32,
            top_k=4,
            load_balance_coeff=None,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=64,
                max_context_length=seq_len,
                theta=150000.0,
                scaling="yarn",
                rope_factor=32,
                beta_fast=32.0,
                beta_slow=1.0,
                truncate=False,
                original_seq_len=4096,
            ),
            enable_sp=parallelism.enable_sequence_parallel,
            enable_ep=parallelism.expert_parallel_degree > 1,
        ),
    )


def _120b(
    attn_backend: str = "varlen",
    *,
    seq_len: int,
    parallelism: ParallelismConfig,
) -> GptOssModel.Config:
    dim = 2880
    hidden_dim = 2880
    n_layers = 36
    return GptOssModel.Config(
        max_context_length=seq_len,
        dim=dim,
        vocab_size=201088,
        tok_embeddings=Embedding.Config(
            num_embeddings=201088,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(
                enable_sp=parallelism.enable_sequence_parallel
            ),
        ),
        norm=RMSNorm.Config(
            normalized_shape=dim,
            param_init=_NORM_INIT,
            sharding_config=pre_lm_head_norm_sharding_config(
                enable_sp=parallelism.enable_sequence_parallel
            ),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=201088,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_gptoss_layers(
            dim=dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            num_experts=128,
            top_k=4,
            load_balance_coeff=None,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(
                dim=64,
                max_context_length=seq_len,
                theta=150000.0,
                scaling="yarn",
                rope_factor=32,
                beta_fast=32.0,
                beta_slow=1.0,
                truncate=False,
                original_seq_len=4096,
            ),
            enable_sp=parallelism.enable_sequence_parallel,
            enable_ep=parallelism.expert_parallel_degree > 1,
        ),
    )


MODEL_FLAVORS = {
    "debugmodel": (_debugmodel, 131072),
    "20b": (_20b, 131072),
    "20b_megatron": (_20b_megatron, 131072),
    "120b": (_120b, 131072),
}


def build_model_config(
    flavor: str,
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
    attn_backend: str = "varlen",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> GptOssModel.Config:
    get_config, max_context_len = MODEL_FLAVORS[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    config = get_config(
        attn_backend=attn_backend,
        seq_len=context_len,
        parallelism=parallelism,
    )
    if converters is not None:
        validate_converter_compatibility(converters)
        for c in converters:
            config = c.build().convert(config)
    return config
