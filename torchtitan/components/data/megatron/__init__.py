# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Grain sources for pre-tokenized Megatron ``.bin``/``.idx`` datasets."""

import glob
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from torchtitan.components.data.dataset import (
    DatasetConcatConfig,
    DatasetConfig,
    DatasetMixConfig,
    SampleProcessor,
    SingleDatasetConfig,
    TextSequence,
    WeightedDataset,
)
from torchtitan.components.data.types import DatasetBuildContext
from torchtitan.config import Configurable

from .indexed_datasets import IndexedDataset

logger = logging.getLogger(__name__)

__all__ = [
    "IndexedDataset",
    "MegatronIndexedSource",
    "PretokenizedProcessor",
    "get_megatron_prefixes",
    "megatron_dataset_config",
]


class MegatronIndexedSource(Configurable):
    """Random access to the documents of one Megatron indexed dataset."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        path_prefix: str
        """Dataset prefix, i.e. the ``.idx``/``.bin`` path without extension."""

        mmap: bool = True

    def __init__(self, config: Config) -> None:
        self._dataset = IndexedDataset(config.path_prefix, mmap=config.mmap)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"tokens": self._dataset[index]}


class PretokenizedProcessor(SampleProcessor):
    """Turns a pre-tokenized document into next-token input and label pairs.

    Documents are used as stored; no BOS/EOS tokens are added.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        pass

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        del config, context

    def __call__(
        self, sample: dict[str, Any], rng: np.random.Generator
    ) -> TextSequence | None:
        del rng
        tokens = np.asarray(sample["tokens"], dtype=np.int64)
        if len(tokens) < 2:
            return None
        return TextSequence(input_ids=tokens[:-1], labels=tokens[1:])


def get_megatron_prefixes(start_path: str) -> list[str]:
    """Return dataset prefixes for an ``.idx`` file or a directory of them."""
    if start_path.endswith(".idx"):
        return [start_path[: -len(".idx")]]
    prefixes = []
    for idx_file in sorted(glob.glob(f"{start_path}/**/*.idx", recursive=True)):
        prefix = idx_file[: -len(".idx")]
        if not os.path.exists(f"{prefix}.bin"):
            raise FileNotFoundError(f"missing {prefix}.bin for {idx_file}")
        prefixes.append(prefix)
    return prefixes


def megatron_dataset_config(
    dataset_path: str, weights: list[float] | None = None
) -> DatasetConfig:
    """Build an unpacked document dataset from Megatron indexed datasets.

    Args:
        dataset_path: Comma-separated ``.idx`` files or directories that are
            searched recursively for ``.idx`` files.
        weights: Optional per-prefix sampling weights. Without weights, all
            documents are concatenated (and shuffled globally by the loader).

    Wrap the result in a packing config (e.g. ``ConcatThenSplitPackingConfig``)
    before passing it to ``GrainDataLoader``.
    """
    prefixes = [
        prefix
        for start_path in dataset_path.split(",")
        for prefix in get_megatron_prefixes(start_path)
    ]
    if not prefixes:
        raise FileNotFoundError(f"no Megatron datasets found in {dataset_path!r}")
    logger.info(f"Using pre-tokenized datasets: {prefixes}")

    datasets = [
        SingleDatasetConfig(
            source=MegatronIndexedSource.Config(path_prefix=prefix),
            processor=PretokenizedProcessor.Config(),
            post_filters=(lambda sample: sample is not None,),
        )
        for prefix in prefixes
    ]
    if weights is not None:
        if len(weights) != len(datasets):
            raise ValueError(
                f"got {len(weights)} weights for {len(datasets)} Megatron datasets"
            )
        return DatasetMixConfig(
            datasets=tuple(
                WeightedDataset(dataset=dataset, weight=weight)
                for dataset, weight in zip(datasets, weights, strict=True)
            )
        )
    if len(datasets) == 1:
        return datasets[0]
    return DatasetConcatConfig(datasets=tuple(datasets))
