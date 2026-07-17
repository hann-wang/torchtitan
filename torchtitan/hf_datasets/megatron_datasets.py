# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset, Subset

from torchtitan.tools.logging import logger
from .megatron_indexed_datasets import IndexedDataset


def split_dataset_by_node(
    dataset: IndexedDataset,
    rank: int,
    world_size: int,
    contiguous: bool = True,
) -> Subset:
    index = rank
    num_shards = world_size
    if not 0 <= index < num_shards:
        raise ValueError("index should be in [0, num_shards-1]")
    if contiguous:
        div = len(dataset) // num_shards
        mod = len(dataset) % num_shards
        start = div * index + min(index, mod)
        end = start + div + (1 if index < mod else 0)
        indices = range(start, end)
    else:
        indices = range(index, len(dataset), num_shards)

    return Subset(dataset, indices)


class MegatronTextDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_prefix: str,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        self.dataset_name = dataset_prefix
        ds = IndexedDataset(dataset_prefix)
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size, contiguous=False)

        self.seq_len = seq_len
        self.infinite = infinite

        # Variables for checkpointing
        self._sample_idx = 0
        self._token_buffer: list[int] = []
        self._positions_buffer: list[int] = []

    def __len__(self) -> int:
        return len(self._data)

    # def __getitem__(self, index: int):
    #     return self._data[index]

    def _get_data_iter(self):
        # For map-style datasets, resume by skipping to the correct index
        if self._sample_idx == len(self._data):
            return iter([])
        else:
            subset = Subset(self._data, range(self._sample_idx, len(self._data)))
            return iter(subset)

    def _normalize_positions(self, positions: list[int]) -> list[int]:
        offset = positions[0]
        if offset > 0:
            for i, p in enumerate(positions):
                if p == 0:
                    break
                positions[i] = p - offset
        return positions

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample_tokens in self._get_data_iter():
                sample_tokens_list = sample_tokens.tolist()
                self._token_buffer.extend(sample_tokens_list)
                self._sample_idx += 1
                self._positions_buffer.extend(range(len(sample_tokens_list)))

                while len(self._token_buffer) >= max_buffer_token_len:
                    x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                    pos = torch.LongTensor(
                        self._normalize_positions(
                            self._positions_buffer[:max_buffer_token_len]
                        )
                    )
                    # update tokens to the remaining tokens
                    self._token_buffer = self._token_buffer[max_buffer_token_len:]
                    self._positions_buffer = self._positions_buffer[
                        max_buffer_token_len:
                    ]
                    input = x[:-1]
                    label = x[1:]
                    positions = pos[:-1]
                    yield {"input": input, "positions": positions}, label

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def load_state_dict(self, state_dict):
        self._token_buffer = state_dict["token_buffer"]
        self._sample_idx = state_dict["sample_idx"]
        self._positions_buffer = state_dict["positions_buffer"]

    def state_dict(self):
        _state_dict = {
            "token_buffer": self._token_buffer,
            "sample_idx": self._sample_idx,
            "positions_buffer": self._positions_buffer,
        }
        return _state_dict
