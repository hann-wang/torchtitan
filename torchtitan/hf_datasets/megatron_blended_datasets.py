# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import glob
import os
import re
import time
from typing import Optional, Union

import numpy
import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset, Subset

import torchtitan.hf_datasets.helpers_cpp as helpers  # pyrefly: ignore[missing-import]
from torchtitan.tools.logging import logger
from .megatron_datasets import MegatronTextDataset

_VERBOSE = False


def normalize(weights: list[float]) -> list[float]:
    """Do non-exponentiated normalization

    Args:
        weights (list[float]): The weights

    Returns:
        list[float]: The normalized weights
    """

    w = numpy.array(weights, dtype=numpy.float64)
    w_sum = numpy.sum(w)
    w = (w / w_sum).tolist()
    return w


class BlendedDataset(IterableDataset, Stateful):
    """Conjugating class for a set of MegatronTextDataset instances

    Args:
        datasets (list[MegatronTextDataset]): The MegatronTextDataset instances to blend

        weights (list[Union[int, float]]): The weights that determine the dataset blend ratios

        size (Optional[int]): The number of samples to draw from the blend. If None, for each
            dataset index idx draw exactly weights[idx] samples from datasets[idx].

        config (BlendedMegatronTextDatasetConfig): The config

    Raises:
        RuntimeError: When the dataset has fewer or more samples than 'size' post-initialization
    """

    def __init__(
        self,
        datasets: list[MegatronTextDataset],
        weights: list[Union[int, float]],
        size: Optional[int],
        seq_len: int = 2048,
        infinite: bool = False,
    ) -> None:
        assert len(datasets) == len(weights)
        assert len(datasets) < 32767
        assert all(map(lambda _: type(_) == type(datasets[0]), datasets))
        assert all(map(lambda _: _ > 0, weights))
        assert all(map(lambda _: type(_) == type(weights[0]), weights))
        if size is None and isinstance(weights[0], float):
            assert all(map(lambda _: _ == int(_), weights))

        # Alert user to unnecessary blending
        if len(datasets) == 1:
            logger.warning("Building a BlendedDataset for a single MegatronTextDataset")

        if size is not None:
            weights = normalize(weights)

        self.datasets = datasets
        self.weights = weights
        self.size = size

        self.dataset_index, self.dataset_sample_index = self._build_indices()

        self.seq_len = seq_len
        self.infinite = infinite

        # Variables for checkpointing
        self._sample_idx = 0
        self._token_buffer: list[int] = []

    def _get_data_iter(self):
        # For map-style datasets, resume by skipping to the correct index
        if self._sample_idx == len(self):
            return iter([])
        else:
            subset = Subset(self, range(self._sample_idx, len(self)))
            return iter(subset)

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample_tokens in self._get_data_iter():
                self._token_buffer.extend(sample_tokens.tolist())
                self._sample_idx += 1

                while len(self._token_buffer) >= max_buffer_token_len:
                    x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self._token_buffer = self._token_buffer[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield {"input": input}, label

            if not self.infinite:
                logger.warning("Blended dataset has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning("Blended dataset is being re-looped")

    def load_state_dict(self, state_dict):
        self._token_buffer = state_dict["token_buffer"]
        self._sample_idx = state_dict["sample_idx"]

    def state_dict(self):
        _state_dict = {
            "token_buffer": self._token_buffer,
            "sample_idx": self._sample_idx,
        }
        return _state_dict

    def __len__(self) -> int:
        return self.dataset_index.shape[0]

    # def __getitem__(self, idx: int):
    #     dataset_id = self.dataset_index[idx]
    #     dataset_sample_id = self.dataset_sample_index[idx]
    #     return self.datasets[dataset_id][dataset_sample_id]

    def _build_indices(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Build and optionally cache the dataset index and the dataset sample index

        The dataset index is a 1-D mapping which determines the dataset to query. The dataset
        sample index is a 1-D mapping which determines the sample to request from the queried
        dataset.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: The dataset index and the dataset sample index
        """

        logger.info(f"Build and save the {type(self).__name__} indices")

        # Build the dataset and dataset sample indexes
        logger.info("\tBuild and save the dataset and dataset sample indexes")
        t_beg = time.time()

        if self.size is not None:
            dataset_index = numpy.zeros(self.size, dtype=numpy.int16)
            dataset_sample_index = numpy.zeros(self.size, dtype=numpy.int64)
            helpers.build_blending_indices(
                dataset_index,
                dataset_sample_index,
                self.weights,
                len(self.datasets),
                self.size,
                _VERBOSE,
            )
        else:
            size = sum(self.weights)
            # pyrefly: ignore[no-matching-overload]
            dataset_index = numpy.zeros(size, dtype=numpy.int16)
            # pyrefly: ignore[no-matching-overload]
            dataset_sample_index = numpy.zeros(size, dtype=numpy.int64)
            helpers.build_exhaustive_blending_indices(
                dataset_index, dataset_sample_index, self.weights, len(self.datasets)
            )

        dataset_indices, dataset_sizes = numpy.unique(dataset_index, return_counts=True)
        for i, (_index, _size) in enumerate(zip(dataset_indices, dataset_sizes)):
            if len(self.datasets[_index]) < _size:
                raise IndexError(
                    f"The {self.datasets[_index].dataset_name} blend oversamples the contributing datasets and, "
                    f"for example, requests {_size} samples from "
                    f"{type(self.datasets[_index]).__name__} number {i} in excess of its size "
                    f"{len(self.datasets[_index])}."
                )

        t_end = time.time()
        logger.debug(f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        return dataset_index, dataset_sample_index


def get_prefixes(start_path: str):
    if start_path.endswith(".idx"):
        return [start_path[:-4]]
    idx_files = glob.glob(f"{start_path}/**/*.idx", recursive=True)
    prefixes = []
    for idx_file in idx_files:
        prefix = re.sub(r"\.idx$", "", idx_file)
        bin_file = f"{prefix}.bin"
        os.path.exists(bin_file)
        prefixes.append(prefix)

    return prefixes


def build_megatron_blended_datasets(
    dataset_path: str,
    seq_len: int,
    dp_rank: int,
    dp_world_size: int,
    infinite: bool,
    probabilities: list[float] | None = None,
):
    start_paths = dataset_path.split(",")
    dataset_prefixes = []
    for start_path in start_paths:
        dataset_prefixes += get_prefixes(start_path)
    logger.info(f"Using pre-tokenized datasets: {dataset_prefixes}")
    if len(dataset_prefixes) == 1:
        ds = MegatronTextDataset(
            dataset_prefixes[0],
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=infinite,
        )
    else:
        lower_datasets = [
            MegatronTextDataset(
                dataset_prefix,
                seq_len=seq_len,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
                infinite=False,
            )
            for dataset_prefix in dataset_prefixes
        ]
        if probabilities is not None:
            assert len(probabilities) == len(lower_datasets)
        else:
            probabilities = [len(lower_dataset) for lower_dataset in lower_datasets]
        ds = BlendedDataset(
            lower_datasets,
            probabilities,
            size=None,
            seq_len=seq_len,
            infinite=infinite,
        )
    return ds
