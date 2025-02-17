# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from transformers import AutoTokenizer
from torchtitan.datasets.tokenizer.tiktoken import TikTokenizer
from torchtitan.datasets.tokenizer.tokenizer import Tokenizer

from torchtitan.logging import logger


def build_tokenizer(tokenizer_type: str, tokenizer_path: str) -> Tokenizer:
    logger.info(f"Building {tokenizer_type} tokenizer locally from {tokenizer_path}")
    if tokenizer_type == "tiktoken":
        return TikTokenizer(tokenizer_path)
    elif tokenizer_type == "hf":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        original_encode_func = tokenizer.encode
        def new_encode_func(*args, **kwargs):
            bos = kwargs.pop("bos")
            eos = kwargs.pop("eos")
            t = original_encode_func(*args, **kwargs)
            if bos:
                t.insert(0, tokenizer.bos_token_id)
            if eos:
                t.append(tokenizer.eos_token_id)
            return t

        tokenizer.encode = new_encode_func

        return tokenizer
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
