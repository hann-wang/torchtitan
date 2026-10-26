# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import re
from typing import Any, TYPE_CHECKING

from torchtitan.models.base.state_dict_adapter import dtensor_safe, StateDictAdapter

from torchtitan.models.common.rope import ComplexRoPE, CosSinRoPE

if TYPE_CHECKING:
    from .model import Llama3Model


logger = logging.getLogger(__name__)


class Llama3StateDictAdapter(StateDictAdapter):
    def __init__(
        self,
        model_config: Llama3Model.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)

        self.model_config = model_config
        self.hf_assets_path = hf_assets_path

        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.qkv_linear.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.qkv_linear.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.qkv_linear.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

        # pyrefly: ignore [missing-attribute]
        self.qk_norm_over_heads = model_config.layers[0].attention.qk_norm_over_heads
        if self.qk_norm_over_heads:
            # Instella-3B names its pre-norms differently from Llama.
            del self.from_hf_map["model.layers.{}.input_layernorm.weight"]
            del self.from_hf_map["model.layers.{}.post_attention_layernorm.weight"]
            self.from_hf_map.update(
                {
                    "model.layers.{}.pre_attention_layernorm.weight": "layers.{}.attention_norm.weight",
                    "model.layers.{}.pre_feedforward_layernorm.weight": "layers.{}.ffn_norm.weight",
                }
            )

    # HuggingFace permutation function (exact copy from their conversion script)
    @dtensor_safe
    def _permute(self, w, n_heads_arg, dim1=None, dim2=None):
        if dim1 is None:
            dim1 = w.shape[0]
        if dim2 is None:
            dim2 = w.shape[1]
        return (
            w.view(n_heads_arg, dim1 // n_heads_arg // 2, 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
            .clone()
        )

    @dtensor_safe
    def _reverse_permute(self, w, n_heads_arg, dim1=None, dim2=None):
        if dim1 is None:
            dim1 = w.shape[0]
        if dim2 is None:
            dim2 = w.shape[1]
        return (
            w.view(n_heads_arg, 2, dim1 // n_heads_arg // 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
        )

    def _uses_complex_rope_layout(self) -> bool:
        """Validate the configured RoPE and report whether Q/K rows are paired.

        Native ``ComplexRoPE`` rotates adjacent dimensions, so it requires the
        Hugging Face half-split Q/K row permutation. ``CosSinRoPE`` uses
        Hugging Face's rotate-half convention directly and needs no permutation.
        """
        rope_types = {
            type(layer.attention.rope)
            for layer in self.model_config.layers
            if layer.attention.rope is not None
        }
        unsupported = rope_types - {ComplexRoPE.Config, CosSinRoPE.Config}
        if unsupported:
            names = ", ".join(sorted(rope.__qualname__ for rope in unsupported))
            raise ValueError(
                "HF checkpoint conversion requires ComplexRoPE.Config or "
                f"CosSinRoPE.Config; got {names}."
            )
        if len(rope_types) > 1:
            raise ValueError(
                "HF checkpoint conversion requires one RoPE layout across "
                f"Llama layers; got {[rope.__qualname__ for rope in rope_types]}."
            )
        return rope_types == {ComplexRoPE.Config}

    def to_hf(
        self, state_dict: dict[str, Any], *, to_quantized_layout: bool = False
    ) -> dict[str, Any]:
        if to_quantized_layout:
            raise ValueError(
                f"{type(self).__name__} does not support loading quantized "
                "HF checkpoints."
            )
        state_dict = self._native_fused_linears_to_hf(state_dict)
        # pyrefly: ignore [missing-attribute]
        attn = self.model_config.layers[0].attention
        n_heads = attn.n_heads
        n_kv_heads = attn.n_kv_heads if attn.n_kv_heads is not None else n_heads
        dim = self.model_config.dim  # pyrefly: ignore [missing-attribute]
        head_dim = attn.head_dim or dim // n_heads
        permute_qk = self._uses_complex_rope_layout()
        hf_state_dict = {}

        to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}

        for key, value in state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)

                new_key = to_hf_map.get(abstract_key)
                if new_key is None:
                    continue

                # Apply HF permutation for Q/K weights
                if (
                    permute_qk
                    and abstract_key == "layers.{}.attention.qkv_linear.wq.weight"
                ):
                    value = self._permute(value, n_heads)
                if (
                    permute_qk
                    and abstract_key == "layers.{}.attention.qkv_linear.wk.weight"
                ):
                    key_value_dim = head_dim * n_kv_heads
                    value = self._permute(value, n_kv_heads, key_value_dim, dim)
                # Norms over all heads scale the same rows as Q/K weights.
                if (
                    permute_qk
                    and abstract_key == "layers.{}.attention.q_norm.weight"
                ):
                    value = self._permute(value.unsqueeze(-1), n_heads).squeeze(-1)
                if (
                    permute_qk
                    and abstract_key == "layers.{}.attention.k_norm.weight"
                ):
                    key_value_dim = head_dim * n_kv_heads
                    value = self._permute(
                        value.unsqueeze(-1), n_kv_heads, key_value_dim, 1
                    ).squeeze(-1)

                new_key = new_key.format(layer_num)
            else:
                if (
                    self.model_config.enable_weight_tying  # pyrefly: ignore [missing-attribute]
                    and key == "lm_head.weight"
                ):
                    if self.fqn_to_index_mapping:
                        self.fqn_to_index_mapping.pop("lm_head.weight", None)
                    continue
                new_key = to_hf_map[key]

            hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(
        self, hf_state_dict: dict[str, Any], *, is_quantized_layout: bool = False
    ) -> dict[str, Any]:
        if is_quantized_layout:
            raise ValueError(
                f"{type(self).__name__} does not support loading quantized "
                "HF checkpoints."
            )
        permute_qk = self._uses_complex_rope_layout()
        if (
            self.model_config.enable_weight_tying  # pyrefly: ignore [missing-attribute]
            and "lm_head.weight" not in hf_state_dict
        ):
            assert "model.embed_tokens.weight" in hf_state_dict
            hf_state_dict["lm_head.weight"] = hf_state_dict["model.embed_tokens.weight"]

        # pyrefly: ignore [missing-attribute]
        attn = self.model_config.layers[0].attention
        n_heads = attn.n_heads
        n_kv_heads = attn.n_kv_heads if attn.n_kv_heads is not None else n_heads
        dim = self.model_config.dim  # pyrefly: ignore [missing-attribute]
        head_dim = attn.head_dim or dim // n_heads
        state_dict = {}

        for key, value in hf_state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)

                # Reverse-permute Q and K for RoPE compatibility
                if (
                    permute_qk
                    and abstract_key == "model.layers.{}.self_attn.q_proj.weight"
                ):
                    value = self._reverse_permute(value, n_heads)
                if (
                    permute_qk
                    and abstract_key == "model.layers.{}.self_attn.k_proj.weight"
                ):
                    key_value_dim = head_dim * n_kv_heads
                    value = self._reverse_permute(value, n_kv_heads, key_value_dim, dim)
                if (
                    permute_qk
                    and abstract_key == "model.layers.{}.self_attn.q_norm.weight"
                ):
                    value = self._reverse_permute(
                        value.unsqueeze(-1), n_heads
                    ).squeeze(-1)
                if (
                    permute_qk
                    and abstract_key == "model.layers.{}.self_attn.k_norm.weight"
                ):
                    key_value_dim = head_dim * n_kv_heads
                    value = self._reverse_permute(
                        value.unsqueeze(-1), n_kv_heads, key_value_dim, 1
                    ).squeeze(-1)

                new_key = self.from_hf_map[abstract_key]
                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = self.from_hf_map[key]
                if new_key is None:
                    continue

            state_dict[new_key] = value

        return self._native_fused_linears_from_hf(state_dict)
