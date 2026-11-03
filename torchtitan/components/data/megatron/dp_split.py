# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DP split that stripes fixed token windows the way Megatron's sampler does."""

from dataclasses import dataclass

import grain.python as grain
import numpy as np

from torchtitan.components.data.dataset import (
    DatasetConcatConfig,
    DatasetConfig,
    SingleDatasetConfig,
    TextSequence,
)
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy

from .indexed_datasets import IndexedDataset


def _full_window_count(total_tokens: int, num_tokens: int) -> int:
    """Return how many ``num_tokens`` input windows fit in ``total_tokens``.

    Each window also consumes one extra token as the final label, matching
    ``build_sample_idx`` with ``add_extra_token_to_sequence``.
    """
    if num_tokens <= 0:
        raise ValueError("num_tokens_per_microbatch must be positive")
    if total_tokens <= 1:
        return 0
    return (total_tokens - 1) // num_tokens


def _indexed_prefixes(dataset: DatasetConfig) -> list[str]:
    """Return indexed-dataset prefixes in the order the dataset concatenates them."""
    from . import MegatronIndexedSource

    if isinstance(dataset, SingleDatasetConfig):
        source = dataset.source
        if not isinstance(source, MegatronIndexedSource.Config):
            raise TypeError(
                "MegatronDpSplitPackingConfig requires a Megatron indexed dataset, "
                f"got source {type(source).__qualname__}"
            )
        return [source.path_prefix]
    if isinstance(dataset, DatasetConcatConfig):
        prefixes: list[str] = []
        for child in dataset.datasets:
            prefixes.extend(_indexed_prefixes(child))
        return prefixes
    raise TypeError(
        "MegatronDpSplitPackingConfig requires megatron_dataset_config() "
        "without blend weights, so the sequences stay in file order. "
        f"Got {type(dataset).__qualname__}."
    )


class _FlatMegatronTokens:
    """Token spans over indexed sequences concatenated in index order."""

    def __init__(self, prefixes: list[str]) -> None:
        datasets = [IndexedDataset(prefix, mmap=True) for prefix in prefixes]
        lengths = np.concatenate(
            [np.asarray(dataset.sequence_lengths, dtype=np.int64) for dataset in datasets]
        )
        sequence_offsets = np.cumsum(
            [0, *[len(dataset) for dataset in datasets]], dtype=np.int64
        )
        prefix = np.empty(len(lengths) + 1, dtype=np.int64)
        prefix[0] = 0
        np.cumsum(lengths, out=prefix[1:])
        self._datasets = datasets
        self._lengths = lengths
        self._sequence_offsets = sequence_offsets
        self._prefix = prefix

    @property
    def total_tokens(self) -> int:
        return int(self._prefix[-1])

    def read(self, start: int, length: int) -> np.ndarray:
        """Return ``length`` tokens beginning at flat offset ``start``."""
        if length < 0 or start < 0 or start + length > self.total_tokens:
            raise IndexError(
                f"token span [{start}, {start + length}) is outside "
                f"the {self.total_tokens}-token corpus"
            )
        parts: list[np.ndarray] = []
        remaining = length
        cursor = start
        sequence = int(np.searchsorted(self._prefix, cursor, side="right") - 1)
        while remaining:
            sequence_start = int(self._prefix[sequence])
            offset = cursor - sequence_start
            take = min(int(self._lengths[sequence]) - offset, remaining)
            dataset_index = int(
                np.searchsorted(self._sequence_offsets, sequence, side="right") - 1
            )
            local_index = sequence - int(self._sequence_offsets[dataset_index])
            part = self._datasets[dataset_index].get(
                local_index, offset=offset, length=take
            )
            parts.append(np.asarray(part, dtype=np.int64))
            remaining -= take
            cursor += take
            sequence += 1
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts)


class _MegatronDpWindows:
    """One rank's fixed-length windows into a shared flat token stream."""

    def __init__(
        self,
        tokens: _FlatMegatronTokens,
        window_indices: np.ndarray,
        *,
        num_tokens: int,
        max_context_length: int,
    ) -> None:
        self._tokens = tokens
        self._window_indices = window_indices
        self._num_tokens = num_tokens
        repeats = num_tokens // max_context_length
        self._positions = np.tile(
            np.arange(max_context_length, dtype=np.int64), repeats
        )
        self._padding_mask = np.zeros(num_tokens, dtype=np.bool_)

    def __len__(self) -> int:
        return len(self._window_indices)

    def __getitem__(self, index: int) -> TextSequence:
        start = int(self._window_indices[index]) * self._num_tokens
        raw = self._tokens.read(start, self._num_tokens + 1)
        return TextSequence(
            input_ids=raw[:-1],
            labels=raw[1:],
            positions=self._positions,
            padding_mask=self._padding_mask,
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class MegatronDpSplitPackingConfig:
    """Stripe fixed token windows across DP ranks.

    ``dataset`` is the unpacked output of ``megatron_dataset_config``: one
    indexed dataset, or a concatenation of them. Sequences stay in that
    order and are treated as one token stream. Window ``k`` starts at flat
    offset ``k * num_tokens_per_microbatch`` and reads one extra token so
    labels stay next-token aligned across sequence and file boundaries.
    Rank ``r`` receives windows ``r, r + dp, r + 2dp, ...`` after an optional
    shared permutation. A trailing group that does not fill every rank is
    dropped.

    The result is a Grain map dataset. ``GrainDataLoader`` checkpoints its
    iterator index, including across the collator and prefetch wrappers.
    """

    dataset: DatasetConfig

    def build(
        self,
        *,
        context: DatasetBuildContext,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> grain.MapDataset:
        prefixes = _indexed_prefixes(self.dataset)
        num_tokens = context.num_tokens_per_microbatch
        context_length = context.max_context_length
        if context_length <= 0:
            raise ValueError("max_context_length must be positive")
        if num_tokens % context_length != 0:
            raise ValueError(
                "megatron dp split requires num_tokens_per_microbatch "
                f"({num_tokens}) to be a multiple of max_context_length "
                f"({context_length})"
            )

        tokens = _FlatMegatronTokens(prefixes)
        num_windows = _full_window_count(tokens.total_tokens, num_tokens)
        dp_world_size = dataset_iteration_policy.dp_world_size
        dp_rank = dataset_iteration_policy.dp_rank
        if dp_world_size <= 0 or not 0 <= dp_rank < dp_world_size:
            raise ValueError(
                f"invalid dp rank {dp_rank} for world size {dp_world_size}"
            )
        num_global = (num_windows // dp_world_size) * dp_world_size
        if num_global == 0:
            raise ValueError(
                f"megatron corpus has {tokens.total_tokens} tokens, which does "
                f"not fill one window of {num_tokens} tokens on each of "
                f"{dp_world_size} data-parallel ranks"
            )

        order = np.arange(num_global, dtype=np.int64)
        if dataset_iteration_policy.shuffle:
            np.random.RandomState(dataset_iteration_policy.seed).shuffle(order)
        window_indices = np.ascontiguousarray(order[dp_rank::dp_world_size])
        dataset = grain.MapDataset.source(
            _MegatronDpWindows(
                tokens,
                window_indices,
                num_tokens=num_tokens,
                max_context_length=context_length,
            )
        )
        if dataset_iteration_policy.repeat:
            dataset = dataset.repeat()
        return dataset
