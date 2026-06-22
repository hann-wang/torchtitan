#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from compressed_tensors.utils import match_named_modules
from torch import nn

from alto.config import Recipe
from alto.kernels.fp4.mxfp4.mxfp_linear import _to_mxfp4_then_scaled_mm
from alto.kernels.dispatch.tensor import MXFP4TrainingWeightWrapperTensor
from torchtitan.models.flux.flux_datasets import FluxDataLoader
from torchtitan.models.flux.config_registry import flux_dev
from torchtitan.models.flux.model.autoencoder import load_ae
from torchtitan.models.flux.model.hf_embedder import FluxEmbedder
from torchtitan.models.flux.utils import (
    create_position_encoding_for_latents,
    pack_latents,
    preprocess_data,
)


RECIPE_PATH = Path("/home/workspace/ALTO/alto/models/flux/configs/lpt_recipe.yaml")


def _make_inputs(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    img_tokens: int,
    txt_tokens: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    img = torch.randn(batch_size, img_tokens, 64, device=device, dtype=dtype, generator=generator)
    txt = torch.randn(batch_size, txt_tokens, 4096, device=device, dtype=dtype, generator=generator)
    y = torch.randn(batch_size, 768, device=device, dtype=dtype, generator=generator)
    timesteps = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)

    img_pos = torch.arange(img_tokens, device=device, dtype=dtype)
    txt_pos = torch.arange(txt_tokens, device=device, dtype=dtype)
    img_ids = torch.stack(
        (
            torch.zeros_like(img_pos),
            img_pos // 16,
            img_pos % 16,
        ),
        dim=-1,
    ).unsqueeze(0).expand(batch_size, -1, -1)
    txt_ids = torch.stack(
        (
            torch.zeros_like(txt_pos),
            torch.zeros_like(txt_pos),
            txt_pos,
        ),
        dim=-1,
    ).unsqueeze(0).expand(batch_size, -1, -1)

    return {
        "img": img,
        "img_ids": img_ids,
        "txt": txt,
        "txt_ids": txt_ids,
        "timesteps": timesteps,
        "y": y,
    }


def _make_dataset_inputs(
    *,
    trainer_config,
    device: torch.device,
    dtype: torch.dtype,
    dataset: str,
    dataset_path: str | None,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, object]]:
    trainer_config.dataloader.dataset = dataset
    trainer_config.dataloader.dataset_path = dataset_path
    trainer_config.dataloader.infinite = False
    trainer_config.dataloader.num_workers = 0
    trainer_config.dataloader.persistent_workers = False
    trainer_config.dataloader.prefetch_factor = None
    trainer_config.training.local_batch_size = batch_size

    dataloader = FluxDataLoader(
        trainer_config.dataloader,
        dp_world_size=1,
        dp_rank=0,
        local_batch_size=batch_size,
    )
    input_dict, labels = next(iter(dataloader))
    metadata = {
        "dataset": dataset,
        "sample_key": input_dict.get("sample_key"),
        "prompt": input_dict.get("prompt"),
    }

    model_args = trainer_config.model_spec.model
    autoencoder = load_ae(
        trainer_config.encoder.autoencoder_path,
        model_args.autoencoder_params,
        device=device,
        dtype=dtype,
        random_init=trainer_config.encoder.test_mode,
    )
    clip_encoder = FluxEmbedder(
        version=trainer_config.encoder.clip_encoder,
        random_init=trainer_config.encoder.test_mode,
    ).to(device=device, dtype=dtype)
    t5_encoder = FluxEmbedder(
        version=trainer_config.encoder.t5_encoder,
        random_init=trainer_config.encoder.test_mode,
    ).to(device=device, dtype=dtype)

    input_dict["image"] = labels
    with torch.no_grad():
        processed = preprocess_data(
            device=device,
            dtype=dtype,
            autoencoder=autoencoder,
            clip_encoder=clip_encoder,
            t5_encoder=t5_encoder,
            batch=input_dict,
        )

        img_encodings = processed["img_encodings"]
        bsz = img_encodings.shape[0]
        torch.manual_seed(seed)
        noise = torch.randn_like(img_encodings)
        timesteps = torch.rand((bsz,), device=device, dtype=dtype)
        sigmas = timesteps.view(-1, 1, 1, 1)
        latents = (1 - sigmas) * img_encodings + sigmas * noise

        bsz, _, latent_height, latent_width = latents.shape
        latent_pos_enc = create_position_encoding_for_latents(
            bsz,
            latent_height,
            latent_width,
            3,
        ).to(device=device, dtype=dtype)
        text_pos_enc = torch.zeros(
            bsz,
            processed["t5_encodings"].shape[1],
            3,
            device=device,
            dtype=dtype,
        )

        transformer_inputs = {
            "img": pack_latents(latents),
            "img_ids": latent_pos_enc,
            "txt": processed["t5_encodings"],
            "txt_ids": text_pos_enc,
            "timesteps": timesteps,
            "y": processed["clip_encodings"],
        }
        target = pack_latents(noise - img_encodings)

    del autoencoder, clip_encoder, t5_encoder
    torch.cuda.empty_cache()
    return transformer_inputs, target, metadata


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.float().reshape(-1)
    cand = candidate.float().reshape(-1)
    err = cand - ref

    mse = torch.mean(err * err)
    mae = torch.mean(torch.abs(err))
    ref_power = torch.mean(ref * ref)
    cos = F.cosine_similarity(ref, cand, dim=0)
    snr = 10.0 * torch.log10(ref_power / torch.clamp(mse, min=1e-30))

    return {
        "snr_db": float(snr.item()),
        "cos_sim": float(cos.item()),
        "mse": float(mse.item()),
        "mae": float(mae.item()),
        "ref_rms": float(torch.sqrt(ref_power).item()),
        "max_abs_err": float(torch.max(torch.abs(err)).item()),
    }


def _format_float(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        return str(value)
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.4e}"
    return f"{value:.6f}"


def _write_markdown(path: Path, rows: list[dict[str, object]], headers: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            values = []
            for key in headers:
                value = row[key]
                if isinstance(value, float):
                    values.append(_format_float(value))
                else:
                    values.append(str(value))
            f.write("| " + " | ".join(values) + " |\n")


def _prefixed_metrics(
    prefix: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float]:
    return {
        f"{prefix}_{key}": value
        for key, value in _metrics(reference, candidate).items()
    }


def _local_mxfp4_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    config,
) -> torch.Tensor:
    y = _to_mxfp4_then_scaled_mm(
        x,
        weight,
        use_2dblock_x=config.use_2dblock_x,
        use_2dblock_w=config.use_2dblock_w,
        use_sr_grad=config.use_sr_grad,
        use_dge=config.use_dge,
        clip_mode=config.clip_mode,
        use_hadamard=config.use_hadamard,
        use_macro_block_scaling=config.two_level_scaling == "blockwise",
    )
    if bias is not None:
        y = y + bias
    return y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/flux_layer_error")
    parser.add_argument("--analysis", choices=("forward", "backward"), default="forward")
    parser.add_argument("--input-source", choices=("random", "dataset"), default="random")
    parser.add_argument("--dataset", default="cc12m-test")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--img-tokens", type=int, default=64)
    parser.add_argument("--txt-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for MXFP4 layer analysis")

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)

    recipe = Recipe.create_instance(str(RECIPE_PATH))
    if len(recipe.modifiers) != 1:
        raise RuntimeError(f"Expected one LPT modifier, got {len(recipe.modifiers)}")
    modifier = recipe.modifiers[0]
    (config, targets), = modifier.resolved_config.items()
    if config.precision != "mxfp4":
        raise RuntimeError(f"Expected mxfp4 config, got {config.precision}")
    if config.use_hadamard or config.use_sr_grad:
        raise RuntimeError(
            "Recipe must have use_hadamard=false and use_sr_grad=false for this analysis; "
            f"got use_hadamard={config.use_hadamard}, use_sr_grad={config.use_sr_grad}"
        )

    previous_default_device = torch.get_default_device()
    torch.set_default_device(device)
    try:
        trainer_config = flux_dev()
        model = trainer_config.model_spec.model.build()
    finally:
        torch.set_default_device(previous_default_device)

    model.init_weights()
    model.to(device=device, dtype=dtype)
    model.eval()

    target_layers: dict[str, nn.Linear] = {}
    for name, module in match_named_modules(model, targets, modifier.ignore):
        if not isinstance(module, nn.Linear):
            continue
        if 32 in tuple(module.weight.shape):
            continue
        target_layers[name] = module

    rows: list[dict[str, object]] = []
    backward_records: dict[str, dict[str, object]] = {}
    hooks = []

    def make_hook(name: str):
        def hook(module: nn.Linear, inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
            x = inputs[0].detach()
            bf16_out = output.detach()
            if args.analysis == "forward":
                wrapped_weight = MXFP4TrainingWeightWrapperTensor(module.weight.detach(), config)
                mxfp4_out = F.linear(x, wrapped_weight, module.bias.detach() if module.bias is not None else None)
                metric_values = _metrics(bf16_out, mxfp4_out)
                rows.append(
                    {
                        "layer": name,
                        "input_shape": tuple(x.shape),
                        "output_shape": tuple(bf16_out.shape),
                        **metric_values,
                    }
                )
            if args.analysis == "backward":
                backward_records[name] = {
                    "module": module,
                    "input": x.cpu(),
                    "input_shape": tuple(x.shape),
                    "output_shape": tuple(bf16_out.shape),
                }
        return hook

    for name, module in target_layers.items():
        hooks.append(module.register_forward_hook(make_hook(name)))

    if args.input_source == "dataset":
        inputs, target, metadata = _make_dataset_inputs(
            trainer_config=trainer_config,
            device=device,
            dtype=dtype,
            dataset=args.dataset,
            dataset_path=args.dataset_path,
            batch_size=args.batch_size,
            seed=args.seed + 1,
        )
    else:
        inputs = _make_inputs(
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
            img_tokens=args.img_tokens,
            txt_tokens=args.txt_tokens,
            seed=args.seed + 1,
        )
        metadata = {
            "dataset": None,
            "sample_key": None,
            "prompt": None,
        }
        target = torch.zeros(
            args.batch_size,
            args.img_tokens,
            64,
            device=device,
            dtype=dtype,
        )

    if args.analysis == "backward":
        # Flux initializes several modulation/final-layer paths to zero, so
        # loss-based full-model gradients can be exactly zero for many early
        # transformer layers in an untrained model. Use the real dataset
        # activations, but inject a deterministic synthetic upstream gradient
        # per layer to isolate the local BF16-vs-MXFP4 backward operator error.
        with torch.no_grad():
            _ = model(**inputs)
        grad_generator = torch.Generator(device=device).manual_seed(args.seed + 2)
        backward_rows: list[dict[str, object]] = []
        for name, record in backward_records.items():
            module = record["module"]
            assert isinstance(module, nn.Linear)

            x_ref = record["input"].to(device=device, dtype=dtype).detach().requires_grad_(True)
            w_ref = module.weight.detach().clone().requires_grad_(True)
            b_ref = (
                module.bias.detach().clone().requires_grad_(True)
                if module.bias is not None
                else None
            )
            grad_output = torch.randn(
                tuple(record["output_shape"]),
                device=device,
                dtype=dtype,
                generator=grad_generator,
            )

            bf16_out = F.linear(x_ref, w_ref, b_ref)
            bf16_grads = torch.autograd.grad(
                bf16_out,
                tuple(t for t in (x_ref, w_ref, b_ref) if t is not None),
                grad_outputs=grad_output,
                retain_graph=False,
            )
            bf16_grad_input = bf16_grads[0]
            bf16_grad_weight = bf16_grads[1]

            x_mx = record["input"].to(device=device, dtype=dtype).detach().requires_grad_(True)
            w_mx = module.weight.detach().clone().requires_grad_(True)
            b_mx = (
                module.bias.detach().clone().requires_grad_(True)
                if module.bias is not None
                else None
            )
            mxfp4_out = _local_mxfp4_linear(x_mx, w_mx, b_mx, config)
            mxfp4_grads = torch.autograd.grad(
                mxfp4_out,
                tuple(t for t in (x_mx, w_mx, b_mx) if t is not None),
                grad_outputs=grad_output,
                retain_graph=False,
            )
            mxfp4_grad_input = mxfp4_grads[0]
            mxfp4_grad_weight = mxfp4_grads[1]

            backward_rows.append(
                {
                    "layer": name,
                    "input_shape": record["input_shape"],
                    "output_shape": record["output_shape"],
                    **_prefixed_metrics("grad_input", bf16_grad_input, mxfp4_grad_input),
                    **_prefixed_metrics("grad_weight", bf16_grad_weight, mxfp4_grad_weight),
                }
            )

            del (
                x_ref,
                w_ref,
                b_ref,
                bf16_out,
                bf16_grads,
                x_mx,
                w_mx,
                b_mx,
                mxfp4_out,
                mxfp4_grads,
                grad_output,
            )
        rows = backward_rows if args.analysis == "backward" else rows
    else:
        with torch.inference_mode():
            _ = model(**inputs)

    for hook in hooks:
        hook.remove()

    sort_key = "snr_db" if args.analysis == "forward" else "grad_weight_snr_db"
    rows.sort(key=lambda row: float(row[sort_key]))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "backward" if args.analysis == "backward" else "forward"
    csv_path = output_dir / f"flux_dev_mxfp4_layer_error_{suffix}.csv"
    md_path = output_dir / f"flux_dev_mxfp4_layer_error_{suffix}.md"
    metadata_path = output_dir / f"flux_dev_mxfp4_layer_error_{suffix}_metadata.txt"

    if args.analysis == "backward":
        fieldnames = [
            "layer",
            "input_shape",
            "output_shape",
            "grad_input_snr_db",
            "grad_input_cos_sim",
            "grad_input_mse",
            "grad_input_mae",
            "grad_input_ref_rms",
            "grad_input_max_abs_err",
            "grad_weight_snr_db",
            "grad_weight_cos_sim",
            "grad_weight_mse",
            "grad_weight_mae",
            "grad_weight_ref_rms",
            "grad_weight_max_abs_err",
        ]
    else:
        fieldnames = [
            "layer",
            "input_shape",
            "output_shape",
            "snr_db",
            "cos_sim",
            "mse",
            "mae",
            "ref_rms",
            "max_abs_err",
        ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_markdown(md_path, rows, fieldnames)
    with metadata_path.open("w", encoding="utf-8") as f:
        f.write(f"analysis={args.analysis}\n")
        if args.analysis == "backward":
            f.write("backward_grad_source=synthetic_random_upstream_grad\n")
        f.write(f"input_source={args.input_source}\n")
        f.write(f"dataset={metadata['dataset']}\n")
        f.write(f"sample_key={metadata['sample_key']}\n")
        f.write(f"prompt={metadata['prompt']}\n")
        for key, value in inputs.items():
            f.write(f"{key}_shape={tuple(value.shape)}\n")

    print(f"recipe={RECIPE_PATH}")
    print(f"use_hadamard={config.use_hadamard} use_sr_grad={config.use_sr_grad}")
    print(f"analysis={args.analysis}")
    print(f"input_source={args.input_source}")
    print(f"dataset={metadata['dataset']}")
    print(f"sample_key={metadata['sample_key']}")
    print(f"layers={len(rows)}")
    print(f"csv={csv_path}")
    print(f"markdown={md_path}")
    print(f"metadata={metadata_path}")
    print(f"worst_layers_by_{sort_key}:")
    for row in rows[:20]:
        if args.analysis == "backward":
            print(
                f"{row['layer']}\t"
                f"dW_snr={_format_float(float(row['grad_weight_snr_db']))}\t"
                f"dW_cos={_format_float(float(row['grad_weight_cos_sim']))}\t"
                f"dX_snr={_format_float(float(row['grad_input_snr_db']))}\t"
                f"dX_cos={_format_float(float(row['grad_input_cos_sim']))}"
            )
        else:
            print(
                f"{row['layer']}\t"
                f"snr_db={_format_float(float(row['snr_db']))}\t"
                f"cos={_format_float(float(row['cos_sim']))}\t"
                f"mse={_format_float(float(row['mse']))}\t"
                f"mae={_format_float(float(row['mae']))}"
            )


if __name__ == "__main__":
    main()
