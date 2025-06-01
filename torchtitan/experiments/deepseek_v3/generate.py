# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# torchrun --standalone --nproc-per-node 4 generate.py

# use inference.sh "Your Question Here?" to run inference with a single prompt.

import sys, os
from dataclasses import dataclass
from typing import List

import torch
import torch.distributed as dist

from checkpoint import load_weights_from_hf
from model import DeepseekForCausalLM
from model_config import deepseek_config_registry
from torch.distributed import get_process_group_ranks
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer

import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.tools import utils
from torchtitan.tools.utils import Color
from torchtitan.config_manager import JobConfig, ConfigManager
from torchtitan.distributed import ParallelDims, utils as dist_utils

# Uncomment the model you want to run.
model_id, model_path, config_file = "deepseek-ai/DeepSeek-V2-Lite-Chat", None, "train_configs/debug_model.toml"
# model_id, model_path, config_file = "deepseek-ai/deepseek-v3", None, "train_configs/deepseek_v3.toml"

if not model_path:
    model_path = model_id

def colorize_chat(text, user_color=None, assistant_color=None, output_color=None):
    """Parse and colorize chat output with optional colors for each role."""
    lines = text.split("\n")
    result = []

    current_role = None
    current_content = []

    def _process_current_content():
        if not current_role or not current_content:
            return None

        content = "\n".join(current_content)
        if current_role == "output":
            return (
                f"Output: {output_color}{content}{color.reset}"
                if output_color
                else f"Output: {content}"
            )
        else:
            try:
                prefix, rest = current_content[0].split(":", 1)
                role_color = user_color if current_role == "user" else assistant_color
                if role_color:
                    formatted = f"{prefix}:{role_color}{rest}{color.reset}"
                    if len(current_content) > 1:
                        formatted += (
                            f"{role_color}\n"
                            + "\n".join(current_content[1:])
                            + f"{color.reset}"
                        )
                    return formatted
            except ValueError:
                pass
        return content

    for line in lines:
        if line.startswith("Output:"):
            if processed := _process_current_content():
                result.append(processed)
            current_role = "output"
            content = line[len("Output:") :].strip()
            if output_color:
                content = f"Output: {output_color}{content}{color.reset}"
            else:
                content = f"Output: {content}"
            result.append(content)
            current_content = []

        elif line.startswith("User:"):
            if processed := _process_current_content():
                result.append(processed)
            current_role = "user"
            current_content = [line]

        elif line.startswith("Assistant:"):
            if processed := _process_current_content():
                result.append(processed)
            current_role = "assistant"
            current_content = [line]

        else:
            if current_content:
                current_content.append(line)
            elif line.strip() and current_role is None:
                # Handle system message at the beginning
                current_role = "output"
                if output_color:
                    result.append(f"Output: {output_color}{line.strip()}{color.reset}")
                else:
                    result.append(f"Output: {line.strip()}")

    # Process the last segment
    if processed := _process_current_content():
        result.append(processed)

    return "\n".join(result)


color = Color()


def create_model(device: torch.device):
    model_args = deepseek_config_registry[model_id]
    model_args.max_seq_len = 4096  # 16384

    with device:
        model = DeepseekForCausalLM(model_args)
        model.to(torch.bfloat16)
    load_weights_from_hf(model, model_path, device)
    model.eval()

    return model


def decode(tokenizer, x):
    output = tokenizer.decode(x[0])
    # Clean up the output by removing special tokens
    bos = tokenizer.bos_token
    output = output.replace(bos, "")
    # Truncate at end of sentence token
    eos_token = tokenizer.eos_token
    if eos_token and eos_token in output:
        output = output.split(eos_token)[0]
    colored_output = colorize_chat(
        output,
        user_color=color.green,
        assistant_color=color.cyan,
        output_color=color.blue,
    )
    return colored_output


def time_generation(func):
    """Decorator to time generation functions and display timing and token per second results."""

    def wrapper(*args, **kwargs):
        rank = dist.get_rank()

        # Setup timer
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # Call the original function
        result, tokens_generated = func(*args, **kwargs)

        # Record end time and calculate elapsed time
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)

        if rank == 0:
            print(
                f"\nGeneration time: {color.yellow}{elapsed_time / 1000:.2f} seconds{color.reset}"
            )
            print(f"Tokens generated: {color.blue}{tokens_generated}{color.reset}")
            print(
                f"Tokens per second: {color.green}{tokens_generated / (elapsed_time / 1000):.2f}\n{color.reset}"
            )

        return result

    return wrapper


@time_generation
@torch.inference_mode()
def generate(
    model_parts: List[torch.nn.Module],
    pp_schedule,
    tokenizer,
    parallel_dims: ParallelDims,
    messages: list[dict],
    n_tokens: int = 200,
    world_mesh: torch.distributed.device_mesh.DeviceMesh = None,
    device: torch.device = torch.device("cpu"),
    pp_has_first_stage: bool = False,
    pp_has_last_stage: bool = False,
):
    rank = dist.get_rank()
    x = tokenizer.apply_chat_template(
        [messages] * parallel_dims.pp,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    next_idx = x.shape[-1]
    x = torch.cat([x, torch.zeros(x.shape[0], n_tokens, dtype=torch.int64)], dim=-1)
    x = x.to(device)

    tokens_generated = 0
    eos_token_id = tokenizer.eos_token_id
    # Create tensor on device for comparison
    eos_tensor = torch.tensor([eos_token_id] * parallel_dims.pp, device=device)

    # Print initial progress indicator
    if rank == 0:
        print("Generating: ", end="", flush=True)

    for _ in range(n_tokens):
        if parallel_dims.pp_enabled:
            pp_group = world_mesh["pp"].get_group()
            pp_ranks = get_process_group_ranks(pp_group)
            if pp_has_first_stage:
                pp_schedule.step(x)
            elif pp_has_last_stage:
                preds = pp_schedule.step()
                if isinstance(preds, DTensor):
                    preds = preds.full_tensor()
                next_token = torch.argmax(preds[:, next_idx - 1], dim=-1)
                x[:, next_idx] = next_token
            else:
                pp_schedule.step()
            torch.distributed.broadcast(
                x,
                group=world_mesh["pp"].get_group(),
                group_src=pp_ranks[-1],
            )
            # Break if EOS token is generated (without .item())
            if torch.equal(x[:, next_idx], eos_tensor):
                tokens_generated += 1
                break
            next_idx += 1
        else:
            assert len(model_parts) == 1
            preds = model_parts[0](x)
            if isinstance(preds, DTensor):
                preds = preds.full_tensor()
            next_token = torch.argmax(preds[:, next_idx - 1], dim=-1)
            x[:, next_idx] = next_token
            next_idx += 1
            # Break if EOS token is generated (without .item())
            if torch.equal(next_token, eos_tensor):
                tokens_generated += 1
                break

        tokens_generated += 1

        # Print progress indicator every 20 tokens
        if rank == 0 and tokens_generated % 20 == 0:
            print(f"{color.yellow}:{color.reset}", end="", flush=True)

    # Print newline after progress indicator
    if rank == 0:
        print()
        colored_output = decode(tokenizer, x)
        print(f"Without CUDA Graph:\n{colored_output}")

    return x, tokens_generated


def apply_parallel(job_config: JobConfig, parallel_dims: ParallelDims,
                   init_device: torch.device):

    model = create_model(torch.device("cpu"))
    train_spec = train_spec_module.get_train_spec(job_config.model.name)
    model_args = train_spec.config[job_config.model.flavor]
    pp_schedule = None
    pp_has_first_stage = pp_has_last_stage = False

    # apply parallelisms and initialization
    if parallel_dims.pp_enabled:
        if not train_spec.pipelining_fn:
            raise RuntimeError(
                f"Pipeline Parallel is enabled but {train_spec.name} "
                f"does not support pipelining")

        # apply both PT-D Pipeline Parallel and SPMD-style PT-D techniques
        pp_schedule, model_parts, pp_has_first_stage, pp_has_last_stage = train_spec.pipelining_fn(
            model,
            world_mesh,
            parallel_dims,
            job_config,
            device,
            model_args,
            train_spec.parallelize_fn,
            None,
        )
        # when PP is enabled, `model` obj is no longer used after this point,
        # model_parts is used instead
        del model

        for m in model_parts:
            m.to(device=init_device)
            m.eval()
    else:
        # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
        model = train_spec.parallelize_fn(model, world_mesh,
                                               parallel_dims, job_config)

        model.to(device=init_device)
        model.eval()

        model_parts = [model]

    return model_parts, pp_schedule, pp_has_first_stage, pp_has_last_stage

if __name__ == "__main__":
    # Get user prompt from command line arguments
    user_prompt = "What is 2+2?"  # Default prompt
    if len(sys.argv) > 1:
        user_prompt = sys.argv[1]

    config_manager = ConfigManager()
    job_config = config_manager.parse_args(["--job.config_file", config_file])

    device_module, device_type = utils.device_module, utils.device_type
    is_distributed = 'LOCAL_RANK' in os.environ and 'WORLD_SIZE' in os.environ

    if is_distributed:
        device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        device = torch.device(f"{device_type}:0")
        world_size = 1
    device_module.set_device(device)

    parallelism_config = job_config.parallelism
    parallelism_config.pipeline_parallel_schedule = "GPipe"
    parallel_dims = ParallelDims(
        dp_shard=parallelism_config.data_parallel_shard_degree,
        dp_replicate=parallelism_config.data_parallel_replicate_degree,
        cp=parallelism_config.context_parallel_degree,
        tp=parallelism_config.tensor_parallel_degree,
        pp=parallelism_config.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=not parallelism_config.disable_loss_parallel,
    )
    if is_distributed:
        dist_utils.init_distributed(job_config)
        world_mesh = parallel_dims.build_mesh(device_type=device_type)
    else:
        world_mesh = None

    dist_utils.set_determinism(
        world_mesh,
        device,
        job_config.training.seed,
        job_config.training.deterministic,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model_parts, pp_schedule, pp_has_first_stage, pp_has_last_stage = apply_parallel(
        job_config,
        parallel_dims,
        device_type,
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_prompt},
    ]

    generate(
        model_parts,
        pp_schedule,
        tokenizer,
        parallel_dims,
        messages,
        world_mesh=world_mesh,
        device=device_type,
        pp_has_first_stage=pp_has_first_stage,
        pp_has_last_stage=pp_has_last_stage,
    )

    if dist.get_rank() == 0:
        print(f"\n{color.yellow}Closing inference mesh...{color.reset}")

    dist.destroy_process_group()
