# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from datasets import Dataset, Features, interleave_datasets, load_dataset, Value
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.megatron_blended_datasets import (
    build_megatron_blended_datasets,
)
from torchtitan.tools.logging import logger

PYTHON_EDU_CONTENT_DIR = "/wekafs/hanwang2/python-edu"
MATHEMATICS_DATASET_DIR = "/wekafs/hanwang2/mathematics_dataset-v1.0"

OFFLINE_MODE = (
    os.environ.get("HF_HUB_OFFLINE", "0") != "0"
    or os.environ.get("NO_STREAMING", "0") != "0"
)


def _load_hf_dataset(
    dataset_path: str,
    split: str,
    name: str | None = None,
    features: Features | None = None,
    columns: list[str] | None = None,
    field: str | None = None,
    num_proc: int | None = None,
    shuffle: bool = False,
):
    kwargs: dict[str, Any] = {}
    if columns:
        kwargs["columns"] = columns
    if field:
        kwargs["field"] = field
    if num_proc is not None:
        kwargs["num_proc"] = num_proc
    """Load hf dataset with default configuration."""
    if OFFLINE_MODE:
        ds = load_dataset(
            dataset_path, name=name, split=split, features=features, **kwargs
        )
        if shuffle:
            ds = ds.shuffle(42)
        ds = ds.to_iterable_dataset()  # type: ignore[attr-defined]
        return ds

    return load_dataset(
        dataset_path,
        name=name,
        split=split,
        streaming=True,
        features=features,
        **kwargs,
    )


def _load_openhermes_dataset(*args, **kwargs):
    ds = _load_hf_dataset(*args, **kwargs)
    ds = ds.map(
        lambda x: {"conversations": x["conversations"], "id": 0},
    )
    return ds


def _load_python_edu_dataset(dataset_path: str):
    jsonl_file = os.path.join(dataset_path, "data.jsonl.gz")
    assert os.path.exists(jsonl_file)
    ds = load_dataset("json", data_files={"train": jsonl_file})["train"]
    if OFFLINE_MODE:
        ds: L = ds.to_iterable_dataset()  # type: ignore[attr-defined]
    # pyrefly: ignore[missing-attribute]
    ds = ds.map(
        lambda x: {"id": 0, "text": x["text"]},
        features=Features(
            {  # type: ignore[bad-argument-type]
                "id": Value("int64"),
                "text": Value("string"),
                "source": Value("string"),
            }
        ),
    )
    return ds


def _load_mathematics_dataset(
    dataset_path: str,
):
    file_paths = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if "readme" in file:
                continue
            if file.endswith(".txt"):
                file_paths.append(os.path.join(root, file))
    ds = load_dataset("text", data_files={"train": file_paths}, sample_by="paragraph")[
        "train"
    ]
    if OFFLINE_MODE:
        ds = ds.to_iterable_dataset()  # type: ignore[attr-defined]

    def process_text(x):
        lines = x["text"].split("\n")
        x["text"] = f"Question: {lines[0]}\nAnswer: {lines[1]}"
        return x

    ds = ds.map(process_text)  # type: ignore[attr-defined]
    return ds


def _process_c4_text(sample: dict[str, Any]) -> dict[str, Any]:
    """Process C4 dataset sample text."""
    return sample


def _process_wiki_text(sample: dict[str, Any]) -> dict[str, Any]:
    """Process C4 dataset sample text."""
    sample["text"] = sample["page"]
    return sample


def _process_openmath_text(doc: dict[str, str]) -> dict[str, Any]:
    doc["text"] = (
        "Problem:"
        + "\n"
        + doc["problem"]
        + "\n\n"
        + "Solution:"
        + "\n"
        + doc["generated_solution"]
        + "\n"
        + "Final Answer: The final answer is "
        + doc["expected_answer"]
        + ". I hope it is correct."
    )
    return doc


def _process_tulu_text(sample: dict[str, Any]) -> dict[str, Any]:
    sample["text"] = json.dumps(sample["messages"], ensure_ascii=False)
    return sample


def _process_openhermes_text(sample: dict[str, Any]) -> dict[str, Any]:
    sample["text"] = json.dumps(sample["conversations"], ensure_ascii=False)
    return sample


def _process_webinstructsub_text(sample: dict[str, Any]) -> dict[str, Any]:
    sample["text"] = f"Question:\n{sample['question']}\n\nAnswer:\n{sample['answer']}"
    return sample


def _process_cosmopedia_text(sample: dict[str, Any]) -> dict[str, Any]:
    conversation = [
        {
            "role": "user",
            "content": sample["prompt"],
        },
        {
            "role": "assistant",
            "content": sample["text"],
        },
    ]
    sample["text"] = json.dumps(conversation, ensure_ascii=False)
    return sample


# Add your dataset here - more information at docs/datasets.md
DATASETS = {
    "c4": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_hf_dataset, split="train", name="en"),
        sample_processor=_process_c4_text,
    ),
    "c4_test": DatasetConfig(
        path="tests/assets/c4_test",
        loader=lambda path: load_dataset(path, split="train"),
        sample_processor=_process_c4_text,
    ),
    "c4_validation": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_hf_dataset, split="validation", name="en"),
        sample_processor=_process_c4_text,
    ),
    "wikitext_test": DatasetConfig(
        path="EleutherAI/wikitext_document_level",
        loader=partial(_load_hf_dataset, split="test", name="wikitext-2-raw-v1"),
        sample_processor=_process_wiki_text,
    ),
    "openmath": DatasetConfig(
        path="nvidia/OpenMathInstruct-2",
        loader=partial(_load_hf_dataset, split="train"),
        sample_processor=_process_openmath_text,
    ),
    "tulu": DatasetConfig(
        path="allenai/tulu-3-sft-mixture",
        loader=partial(_load_hf_dataset, split="train", columns=["messages"]),
        sample_processor=_process_tulu_text,
    ),
    "dolmino": DatasetConfig(
        path="bedio/dolmino-mix-1124-50B",
        loader=partial(_load_hf_dataset, split="train", num_proc=16, shuffle=True),
        sample_processor=_process_c4_text,
    ),
    "openhermes": DatasetConfig(
        path="teknium/OpenHermes-2.5",
        loader=partial(_load_openhermes_dataset, split="train"),
        sample_processor=_process_openhermes_text,
    ),
    "webinstructsub": DatasetConfig(
        path="TIGER-Lab/WebInstructSub",
        loader=partial(_load_hf_dataset, split="train"),
        sample_processor=_process_webinstructsub_text,
    ),
    "codefeedback": DatasetConfig(
        path="m-a-p/Code-Feedback",
        loader=partial(_load_hf_dataset, split="train"),
        sample_processor=_process_tulu_text,
    ),
    "ultrachat": DatasetConfig(
        path="HuggingFaceH4/ultrachat_200k",
        loader=partial(_load_hf_dataset, split="train_sft"),
        sample_processor=_process_tulu_text,
    ),
    # "smollm-corpus-cosmopedia":
    # DatasetConfig(
    #     path="HuggingFaceTB/smollm-corpus",
    #     loader=partial(
    #         _load_hf_dataset,
    #         split="train",
    #         name="cosmopedia-v2",
    #     ),
    #     sample_processor=_process_cosmopedia_text,
    # ),
    # "smollm-corpus-fineweb":
    # DatasetConfig(
    #     path="HuggingFaceTB/smollm-corpus",
    #     loader=partial(
    #         _load_hf_dataset,
    #         split="train",
    #         name="fineweb-edu-dedup",
    #     ),
    #     sample_processor=_process_tulu_text,
    # ),
    "pythonedu": DatasetConfig(
        path=PYTHON_EDU_CONTENT_DIR,
        loader=_load_python_edu_dataset,
        sample_processor=_process_c4_text,
    ),
    "gsm8k": DatasetConfig(
        path="amd/Instella-GSM8K-synthetic",
        loader=partial(
            _load_hf_dataset,
            split="train",
        ),
        sample_processor=_process_tulu_text,
    ),
    "mathematics": DatasetConfig(
        path=MATHEMATICS_DATASET_DIR,
        loader=_load_mathematics_dataset,
        sample_processor=_process_c4_text,
    ),
}


def _validate_dataset(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path."""
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.sample_processor


class HuggingFaceTextDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        seed: int | None = None,
        probabilities: list[float] | None = None,
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()
        dataset_names = dataset_name.split(",")
        if dataset_path is not None:
            dataset_paths = dataset_path.split(",")
            assert len(dataset_names) == len(dataset_paths)
        else:
            dataset_paths = [None] * len(dataset_names)

        datasets = []
        for name, path in zip(dataset_names, dataset_paths):
            path, dataset_loader, text_processor = _validate_dataset(name, path)
            ds = dataset_loader(path)
            ds = ds.map(text_processor)
            datasets.append(ds)

        if len(datasets) > 1:
            if probabilities is not None:
                assert len(probabilities) == len(datasets)
            ds = interleave_datasets(
                datasets,
                seed=seed,
                probabilities=probabilities,
                stopping_strategy="all_exhausted",
            )
            ds = ds.shuffle(seed=seed)
        else:
            ds = datasets[0]

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite

        # Variables for checkpointing
        self._sample_idx = 0
        self._token_buffer: list[int] = []

    def _get_data_iter(self):
        # For map-style datasets, resume by skipping to the correct index
        # For iterable-style datasets, the underlying iterator already points to the correct index
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))

        return iter(self._data)

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                # Use the dataset-specific text processor
                sample_text: str = sample["text"]  # type: ignore[index]
                sample_tokens = self._tokenizer.encode(
                    sample_text, add_bos=True, add_eos=True
                )
                self._token_buffer.extend(sample_tokens)
                self._sample_idx += 1

                while len(self._token_buffer) >= max_buffer_token_len:
                    x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self._token_buffer = self._token_buffer[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield {"input": input}, label

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")
                # Ensures re-looping a dataset loaded from a checkpoint works correctly
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        self._token_buffer = state_dict["token_buffer"]

        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        _state_dict: dict[str, Any] = {"token_buffer": self._token_buffer}

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            # Save the iterable dataset's state to later efficiently resume from it
            # https://huggingface.co/docs/datasets/v3.5.0/en/stream#save-a-dataset-checkpoint-and-resume-iteration
            _state_dict["data"] = self._data.state_dict()

        return _state_dict


class HuggingFaceTextDataLoader(ParallelAwareDataloader):
    """Configurable text dataloader that wraps HuggingFaceTextDataset.

    This dataloader can be used for both training and validation by
    configuring the appropriate dataset, seq_len, batch_size, etc.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset: str = "c4_test"
        """Dataset to use"""

        infinite: bool = True
        """Whether to loop the dataset infinitely"""

        seed: int | None = None
        """Random seed for shuffling the dataset"""

        probabilities: list[float] | None = None
        """Probabilities for interleaving the datasets"""

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        **kwargs,
    ):
        if config.dataset == "megatron":
            assert config.dataset_path
            hf_ds = build_megatron_blended_datasets(
                config.dataset_path,
                seq_len=seq_len,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
                infinite=config.infinite,
                probabilities=config.probabilities,
            )
        else:
            hf_ds = HuggingFaceTextDataset(
                dataset_name=config.dataset,
                dataset_path=config.dataset_path,
                tokenizer=tokenizer,
                seq_len=seq_len,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
                infinite=config.infinite,
                seed=config.seed,
                probabilities=config.probabilities,
            )

        dataloader_kwargs = {
            "num_workers": config.num_workers,
            "persistent_workers": config.persistent_workers,
            "pin_memory": config.pin_memory,
            "prefetch_factor": config.prefetch_factor,
            "batch_size": local_batch_size,
        }

        super().__init__(
            hf_ds,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            **dataloader_kwargs,
        )
