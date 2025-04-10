# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple
from dataclasses import dataclass

import torch.nn as nn

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.datasets.tokenizer.hftoken import build_hf_tokenizer
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec, BaseModelArgs

from transformers.models.opt.modeling_opt import OPTForCausalLM, OPTConfig
from .infra.parallelize_opt import parallelize_opt

__all__ = ["TransformerModelArgs", "opt_configs"]


@dataclass
class TransformerModelArgs(BaseModelArgs):
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by hf config


    def update_from_config(self, hf_config: OPTConfig) -> None:
        self.dim=hf_config.hidden_size
        self.n_layers=hf_config.num_hidden_layers
        self.n_heads=hf_config.num_attention_heads
        self.n_kv_heads=hf_config.num_attention_heads
        self.vocab_size = hf_config.vocab_size

    def get_nparams_and_flops(self, model: nn.Module,
                              seq_len: int) -> tuple[int, int]:
        nparams = sum(p.numel() for p in model.parameters())
        nparams_embedding = sum(
            sum(p.numel() for p in m.parameters()) for m in model.children()
            if isinstance(m, nn.Embedding))

        l, h, q, t = (
            self.n_layers,
            self.n_heads,
            self.dim // self.n_heads,
            seq_len,
        )
        # Reasoning behind the factor of 12 for the self-attention part of the formula:
        # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
        # 2. the flash attention does 1 more matmul recomputation in the backward
        #    but recomputation should not be counted in calculating MFU           (+0)
        # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
        # 4. we follow the convention and do not account for sparsity in causal attention
        num_flops_per_token = 6 * (nparams -
                                   nparams_embedding) + 12 * l * h * q * t

        return nparams, num_flops_per_token

opt_configs = {
    "125m": TransformerModelArgs(
        dim=768,
        n_layers=12,
        n_heads=12,
    ),
}

register_train_spec(
    TrainSpec(
        name="opt",
        cls=OPTForCausalLM,
        config=opt_configs,
        parallelize_fn=parallelize_opt,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    ))
