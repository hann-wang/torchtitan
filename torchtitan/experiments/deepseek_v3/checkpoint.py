# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
from typing import Dict, Optional, Set, Tuple, Iterable
import re

import torch
from safetensors import safe_open

from transformers.utils import cached_file


logger = logging.getLogger(__name__)

_DEFAULT_SAFETENSOR_FILE_NAME = "model.safetensors.index.json"

PARAM_MAPPING = {
    r"\.moe\.router\.gate\.weight$": ".mlp.gate.weight",
    # r"\.mlp\.expert_bias": ".mlp.gate.e_score_correction_bias","
    "moe.shared_expert.w1": "mlp.shared_experts.gate_proj.weight",
    "moe.shared_expert.w2": "mlp.shared_experts.down_proj.weight",
    "moe.shared_expert.w3": "mlp.shared_experts.up_proj.weight",
    "moe.shared_expert": "mlp.shared_experts",
    ".moe.": ".mlp.",
    ".feed_forward.": ".mlp.",
    r"^model.tok_embeddings": "model.embed_tokens",
    r"^output": "lm_head",
}

EXPERT_WEIGHT_MAPPING = {
    "gate_proj": "w1",
    "up_proj": "w3",
    "down_proj": "w2",
}

def read_weights_from_json(file_path: str) -> Optional[Dict[str, str]]:
    try:
        with open(file_path, "r") as file:
            data = json.load(file)

        if "weight_map" in data and isinstance(data["weight_map"], dict):
            return data["weight_map"]
        else:
            logger.error("No 'weight_map' dictionary found in the JSON file.")
            return None
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"An error occurred while reading the JSON file: {str(e)}")
        return None


def get_hf_weight_map_and_path(
    model_id: str,
) -> Tuple[Dict[str, str], str]:
    """Get the weight map for a given HF model id and also the cache path for loading the weights"""
    try:
        index_file = cached_file(model_id, _DEFAULT_SAFETENSOR_FILE_NAME)
    except Exception as e:
        logger.error(
            f"Model `{model_id}` not found in HF cache. "
            f"You can download the model using `python download.py {model_id}"
        )
        raise e

    weight_map = read_weights_from_json(index_file)
    weight_path = os.path.dirname(index_file)
    logger.info(f"Loading weights from: {weight_path}")
    return weight_map, weight_path


def get_needed_files(
    state_dict_keys: Iterable[str], weight_map: Dict[str, str]
) -> Set[str]:
    needed_files = set()
    for param in state_dict_keys:
        file = weight_map.get(param)
        if file:
            needed_files.add(file)
        elif param.endswith("weight"):
            raise ValueError(
                f"Parameter {param} not found in weight map, please check..."
            )
    logger.debug(f"Needed files: {needed_files}")
    return needed_files


def load_safetensor_file(
    full_path: str, device: torch.device
) -> Dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(full_path, framework="pt", device=device) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    logger.debug(f"Loaded {len(tensors)} tensors from {full_path}")
    return tensors


def combine_expert_weights(
    state_dict: Dict[str, torch.Tensor],
    expert_weights: Dict[str, torch.Tensor],
    updated_states: Set[str],
) -> None:
    for key in state_dict.keys():
        try:
            sd_key_prefix = re.search(r"^(.+?)\.experts\.", key).group(1)
            hf_key_prefix = sd_key_prefix.replace(".moe", ".mlp")
        except AttributeError:
            continue
        for k in expert_weights.keys():
            if k.startswith(hf_key_prefix):
                k_splitted = k.split(".")
                expert_id = int(k_splitted[5])
                weight_name = k_splitted[-2]
                sd_weight_name = EXPERT_WEIGHT_MAPPING[weight_name]
                sd_key = f"{sd_key_prefix}.experts.{sd_weight_name}"
                w = state_dict[sd_key]
                w.data[expert_id, :, :] = expert_weights[k].T
        updated_states.add(key)


def load_safetensor_weights(
    model: torch.nn.Module,
    weight_map: Dict[str, str],
    file_location: str,
    device: torch.device,
):
    """
    Load safetensor weights into a `nn.Module`.

    Args:
        model (Module): The PyTorch module to load weights into. It may be a
        model chunk or a full model.
        weight_map (Dict[str, str]): Mapping of model parameters to file names.
        file_location (str): Directory containing the weight files.
        device (torch.device): The device to load tensors onto.
    """
    model_state_dict = model.state_dict()
    param_key_reverse_mapping = {}
    hf_keys = []
    expert_weights = {}
    for param in model_state_dict.keys():
        replaced = False
        for pattern, value in PARAM_MAPPING.items():
            if re.search(pattern, param):
                new_key = re.sub(pattern, value, param)
                param_key_reverse_mapping[new_key] = param
                hf_keys.append(new_key)
                replaced = True
                break
        if re.search(".experts.", param):
            hf_param_name = param.replace(".moe.", ".mlp.")
            key_prefix = re.search(r"^(.+?)\.experts\.",
                                   hf_param_name).group(0)
            for k in weight_map.keys():
                if k.startswith(key_prefix):
                    hf_keys.append(k)
                    expert_weights[k] = None
            replaced = True
        if not replaced:
            hf_keys.append(param)
    needed_files = get_needed_files(hf_keys, weight_map)
    updated_states: Set[str] = set()

    for file in needed_files:
        full_path = os.path.join(file_location, file)
        try:
            checkpoint = load_safetensor_file(full_path, "cpu")
        except FileNotFoundError:
            logger.error(f"File not found: {full_path}")
        except Exception as e:
            logger.error(f"Error during checkpoint processing of {full_path}: {str(e)}")

        matched_keys = set(checkpoint.keys()) & set(hf_keys)
        for key in matched_keys:
            if key in expert_weights:
                # Handle expert weights separately
                expert_weights[key] = checkpoint[key]
                continue
            sd_key = param_key_reverse_mapping.get(key, key)
            # Check shape
            hf_tensor = checkpoint[key]
            if ".shared_expert.w" in sd_key:
                hf_tensor = hf_tensor.T.unsqueeze(0).contiguous()
            if model_state_dict[sd_key].shape != hf_tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {key}: "
                    f"model needs {model_state_dict[sd_key].shape}, but "
                    f"checkpoint has {hf_tensor.shape}")
            model_state_dict[sd_key] = hf_tensor.to(device)
            updated_states.add(sd_key)

    combine_expert_weights(
        model_state_dict,
        expert_weights,
        updated_states,
    )

    missing_keys = set(model_state_dict.keys()) - updated_states
    if missing_keys:
        raise RuntimeError(
            f"Partially updated state dict. Missing parameters: {missing_keys}"
        )

    model.load_state_dict(model_state_dict, strict=False, assign=True)
    logger.debug(f"Successfully loaded {len(updated_states)} weights into model")


def load_weights_from_hf(
    model: torch.nn.Module,
    distribution: str,
    device: torch.device,
):
    """
    Load the weights from Hugging Face format (index file + multiple safetensor
    files), and fill into `model`.  Model config is needed b/c we permute
    wq and wk weights based on attn heads.
    """

    weight_map, weight_path = get_hf_weight_map_and_path(distribution)

    load_safetensor_weights(
        model,
        weight_map,
        weight_path,
        device,
    )
