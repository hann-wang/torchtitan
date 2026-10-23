# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verified Llama 3 model recipes."""

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    DatasetMixConfig,
    GrainDataLoader,
    WeightedDataset,
)
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import AdamW, Optim, OptimizersContainer
from torchtitan.components.validator import Validator
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import MXFP8LinearConverter
from torchtitan.distributed.activation_remat import (
    DefaultActivationRematConfig,
    FullActivationRematConfig,
)
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.llama3 import build_model_config
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.observability.profiler import Profiler
from torchtitan.trainer import Trainer

from torchtitan_recipes.data import c4_document_capacity


def llama3_mxfp8_linear_converter_config() -> MXFP8LinearConverter.Config:
    """Build the MXFP8 policy shared by eager and GraphTrainer recipes."""
    return MXFP8LinearConverter.Config(
        linears_saving_inputs_for_backward_in_mxfp8=[
            "attention.qkv_linear.wqkv",
            "feed_forward.w2",
        ],
    )


def c4_validator_config(steps: int) -> Validator.Config:
    """Fixed-step C4 validation that keeps every DP rank in lockstep."""
    return Validator.Config(
        freq=500,
        steps=steps,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_validation"]),
            shuffle=False,
        ),
    )


def llama3_1b(seq_len: int | None = 8192) -> Trainer.Config:
    parallelism = ParallelismConfig()
    model_config = build_model_config(
        "1B", seq_len=seq_len, attn_backend="sdpa", parallelism=parallelism
    )
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./assets/hf/Llama-3.1-1B",
        profiler=Profiler.Config(enable_profiling=True, profile_freq=100),
        metrics=MetricsProcessor.Config(enable_tensorboard=True),
        model=model_config,
        parallelism=parallelism,
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=4e-5)]
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=1000,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        checkpointer=CheckpointManager.Config(interval=500),
        activation_remat=DefaultActivationRematConfig(),
        validator=c4_validator_config(steps=1200),
    )


INSTELLA_3B_DATA_MIX = {
    "tulu": 0.0116,
    "dolmino": 0.8510,
    "openhermes": 0.0069,
    "webinstructsub": 0.0249,
    "codefeedback": 0.0016,
    "ultrachat": 0.0125,
    "pythonedu": 0.0516,
    "mathematics": 0.0343,
    "gsm8k": 0.0056,
}
"""Instella-3B stage-2 document mixture weights."""


def instella_3b() -> Trainer.Config:
    parallelism = ParallelismConfig()
    model_config = build_model_config(
        "3B_instella", attn_backend="sdpa", parallelism=parallelism
    )
    max_context_length = model_config.max_context_length
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./assets/hf/Instella-3B-Stage1",
        model=model_config,
        parallelism=parallelism,
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=4e-5)]
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=8 * max_context_length,
            num_tokens_per_train_step=1024 * max_context_length,
            max_context_length=max_context_length,
            steps=13800,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(
                dataset=DatasetMixConfig(
                    datasets=tuple(
                        WeightedDataset(dataset=DATASETS[name], weight=weight)
                        for name, weight in INSTELLA_3B_DATA_MIX.items()
                    )
                )
            ),
        ),
        checkpointer=CheckpointManager.Config(interval=500),
        activation_remat=None,
        validator=c4_validator_config(steps=1200),
    )


def llama3_8b(seq_len: int | None = None) -> Trainer.Config:
    parallelism = ParallelismConfig()
    model_config = build_model_config("8B", seq_len=seq_len, parallelism=parallelism)
    num_tokens_per_microbatch_per_dp_rank = model_config.max_context_length
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./assets/hf/Llama-3.1-8B",
        profiler=Profiler.Config(enable_profiling=True, profile_freq=100),
        metrics=MetricsProcessor.Config(enable_tensorboard=True),
        model=model_config,
        parallelism=parallelism,
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=3e-4)]
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch_per_dp_rank,
            max_context_length=model_config.max_context_length,
            steps=1000,
        ),
        dataloader=GrainDataLoader.Config(
            max_num_documents=c4_document_capacity(
                num_tokens_per_microbatch_per_dp_rank
            ),
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        checkpointer=None,
        activation_remat=DefaultActivationRematConfig(),
        validator=None,
    )


def llama3_70b(seq_len: int | None = None) -> Trainer.Config:
    parallelism = ParallelismConfig(tensor_parallel_degree=8)
    model_config = build_model_config("70B", seq_len=seq_len, parallelism=parallelism)
    num_tokens_per_microbatch_per_dp_rank = 8 * model_config.max_context_length
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./assets/hf/Llama-3.1-70B",
        profiler=Profiler.Config(enable_profiling=True, profile_freq=100),
        metrics=MetricsProcessor.Config(enable_tensorboard=True),
        model=model_config,
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=1.5e-4)]
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch_per_dp_rank,
            max_context_length=model_config.max_context_length,
            steps=1000,
        ),
        dataloader=GrainDataLoader.Config(
            max_num_documents=c4_document_capacity(
                num_tokens_per_microbatch_per_dp_rank
            ),
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=FullActivationRematConfig(),
        validator=None,
    )
