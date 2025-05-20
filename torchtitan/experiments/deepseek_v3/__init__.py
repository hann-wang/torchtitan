# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.datasets.tokenizer.tiktoken import build_tiktoken_tokenizer
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from torchtitan.models.llama3 import pipeline_llama
from .infra.parallelize_deepseek import parallelize_deepseek
from .model_config import ModelArgs, deepseek_config_registry
from .model import DeepseekForCausalLM

__all__ = [
    "ModelArgs",
    "DeepseekForCausalLM",
    "llama4_configs",
]

deepseek_configs = {
    "debugmodel":
    ModelArgs(
        vocab_size=102400,
        hidden_size=512,
        intermediate_size=1024,
        moe_intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        n_shared_experts=1,
        n_routed_experts=8,
        routed_scaling_factor=1.0,
        kv_lora_rank=512,
        q_lora_rank=None,
        qk_rope_head_dim=64,
        v_head_dim=64,
        qk_nope_head_dim=64,
        topk_method="greedy",
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        norm_topk_prob=False,
        scoring_func="softmax",
        max_position_embeddings=4096,
        rope_scaling={
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
        },
    ),
    "V2-Lite":
    deepseek_config_registry["deepseek-ai/DeepSeek-V2-Lite"],
    "V3":
    deepseek_config_registry["deepseek-ai/deepseek-v3"],
}

register_train_spec(
    TrainSpec(
        name="DeepSeek",
        cls=DeepseekForCausalLM,
        config=deepseek_configs,
        parallelize_fn=parallelize_deepseek,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_tiktoken_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    ))
