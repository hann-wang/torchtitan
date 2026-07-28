# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import csv
import io
import itertools
import json
import math
import tarfile
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import PIL.Image
import torch
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.models.flux.tokenizer import build_flux_tokenizer, FluxTokenizer
from torchtitan.tools.logging import logger

from .configs import Encoder

MLPERF_COCO_VALIDATION_SAMPLES = 29_696
MLPERF_COCO_TIMESTEPS = 8


def _apply_pil_exif_safe_patches() -> None:
    """Prevent bad image metadata from crashing PIL/HF dataset decoding."""

    try:
        import PIL.ImageFile
        import PIL.ImageOps
    except Exception:
        return

    try:
        PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
    except Exception:
        pass

    image_cls = getattr(PIL.Image, "Image", None)
    if image_cls is not None:
        orig_getexif = getattr(image_cls, "getexif", None)
        if callable(orig_getexif) and not getattr(orig_getexif, "_tt_exif_safe", False):

            def _safe_getexif(self: Any) -> Any:
                try:
                    return orig_getexif(self)
                except Exception:
                    try:
                        exif_cls = getattr(PIL.Image, "Exif", None)
                        if exif_cls is not None:
                            return exif_cls()
                    except Exception:
                        pass
                    return {}

            setattr(_safe_getexif, "_tt_exif_safe", True)
            image_cls.getexif = _safe_getexif

    orig_exif_transpose = getattr(PIL.ImageOps, "exif_transpose", None)
    if callable(orig_exif_transpose) and not getattr(
        orig_exif_transpose, "_tt_exif_safe", False
    ):

        def _safe_exif_transpose(image: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return orig_exif_transpose(image, *args, **kwargs)
            except Exception:
                return image

        setattr(_safe_exif_transpose, "_tt_exif_safe", True)
        PIL.ImageOps.exif_transpose = _safe_exif_transpose


_apply_pil_exif_safe_patches()


_BAD_IMAGE_WARNING_SUBSTRINGS = (
    "Truncated File Read",
    "Corrupt EXIF data",
    "image file is truncated",
)


def _has_bad_image_warning(captured_warnings: list[warnings.WarningMessage]) -> bool:
    return any(
        any(substr in str(warning.message) for substr in _BAD_IMAGE_WARNING_SUBSTRINGS)
        for warning in captured_warnings
    )


def _process_cc12m_image(
    img: PIL.Image.Image,
    output_size: int = 256,
) -> torch.Tensor | None:
    """Process CC12M image to the desired size."""

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")

        try:
            width, height = img.size
            # Skip low resolution images
            if width < output_size or height < output_size:
                return None

            if width >= height:
                # resize height to be equal to output_size, then crop
                new_width, new_height = (
                    math.ceil(output_size / height * width),
                    output_size,
                )
                img = img.resize((new_width, new_height))
                left = torch.randint(0, new_width - output_size + 1, (1,)).item()
                resized_img = img.crop((left, 0, left + output_size, output_size))
            else:
                # resize width to be equal to output_size, the crop
                new_width, new_height = (
                    output_size,
                    math.ceil(output_size / width * height),
                )
                img = img.resize((new_width, new_height))
                lower = torch.randint(0, new_height - output_size + 1, (1,)).item()
                resized_img = img.crop((0, lower, output_size, lower + output_size))

            assert resized_img.size[0] == resized_img.size[1] == output_size

            # Convert grayscale images, and RGBA, CMYK images
            if resized_img.mode != "RGB":
                resized_img = resized_img.convert("RGB")

            # Force PIL to read image payload on CPU, so truncated/corrupt images
            # are rejected before the batch reaches GPU preprocessing.
            resized_img.load()

            # Normalize the image to [-1, 1]
            np_img = np.array(resized_img).transpose((2, 0, 1))
            tensor_img = torch.tensor(np_img).float() / 255.0 * 2.0 - 1.0
        except (OSError, ValueError, SyntaxError):
            return None

        if _has_bad_image_warning(captured_warnings):
            return None

    # NOTE: The following commented code is an alternative way
    # img_transform = transforms.Compose(
    #     [
    #         transforms.Resize(max(output_size, output_size)),
    #         transforms.CenterCrop((output_size, output_size)),
    #         transforms.ToTensor(),
    #     ]
    # )
    # tensor_img = img_transform(img)

    return tensor_img


def _cc12m_wds_data_processor(
    sample: dict[str, Any],
    t5_tokenizer: FluxTokenizer,
    clip_tokenizer: FluxTokenizer,
    output_size: int = 256,
) -> dict[str, Any]:
    """
    Preprocess CC12M dataset sample image and text for Flux model.

    Args:
        sample: A sample from dataset
        t5_tokenizer: T5 tokenizer
        clip_tokenizer: CLIP tokenizer
        output_size: The output image size

    """
    img = _process_cc12m_image(sample["jpg"], output_size=output_size)
    t5_tokens = t5_tokenizer.encode(sample["txt"])
    clip_tokens = clip_tokenizer.encode(sample["txt"])

    return {
        "image": img,
        "clip_tokens": clip_tokens,  # type: List[int]
        "t5_tokens": t5_tokens,  # type: List[int]
        "prompt": sample["txt"],  # type: str
        "sample_key": sample.get("__key__", "unknown"),  # type: str
    }


def _coco_data_processor(
    sample: dict[str, Any],
    t5_tokenizer: FluxTokenizer,
    clip_tokenizer: FluxTokenizer,
    output_size: int = 256,
) -> dict[str, Any]:
    """
    Preprocess COCO dataset sample image and text for Flux model.

    Args:
        sample: A sample from dataset
        t5_tokenizer: T5 tokenizer
        clip_tokenizer: CLIP tokenizer
        output_size: The output image size

    """
    img = _process_cc12m_image(sample["image"], output_size=output_size)
    prompt = sample["caption"]
    if isinstance(prompt, list):
        prompt = prompt[0]
    t5_tokens = t5_tokenizer.encode(prompt)
    clip_tokens = clip_tokenizer.encode(prompt)

    return {
        "image": img,
        "clip_tokens": clip_tokens,  # type: List[int]
        "t5_tokens": t5_tokens,  # type: List[int]
        "prompt": prompt,  # type: str
        "sample_key": str(sample.get("__key__", sample.get("id", "unknown"))),
    }


DATASETS = {
    "cc12m-wds": DatasetConfig(
        path="pixparse/cc12m-wds",
        loader=lambda path: load_dataset(path, split="train", streaming=True),
        sample_processor=_cc12m_wds_data_processor,
    ),
    "cc12m-test": DatasetConfig(
        path="tests/assets/cc12m_test",
        loader=lambda path: load_dataset(
            path, split="train", data_files={"train": "*.tar"}, streaming=True
        ),
        sample_processor=_cc12m_wds_data_processor,
    ),
    "coco-validation": DatasetConfig(
        path="howard-hou/COCO-Text",
        loader=lambda path: load_dataset(path, split="validation", streaming=True),
        sample_processor=_coco_data_processor,
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


class FluxDataset(IterableDataset, Stateful):
    """Dataset for FLUX text-to-image model.

    Args:
    dataset_name (str): Name of the dataset.
    dataset_path (str): Path to the dataset.
    model_transform (Transform): Callable that applies model-specific preprocessing to the sample.
    dp_rank (int): Data parallel rank.
    dp_world_size (int): Data parallel world size.
    infinite (bool): Whether to loop over the dataset infinitely.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        t5_tokenizer: BaseTokenizer,
        clip_tokenizer: BaseTokenizer,
        classifier_free_guidance_prob: float,
        img_size: int,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:

        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, data_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)

        self._t5_tokenizer = t5_tokenizer
        self._t5_empty_token = t5_tokenizer.encode("")
        self._clip_tokenizer = clip_tokenizer
        self._clip_empty_token = clip_tokenizer.encode("")
        self._data_processor = data_processor
        self.classifier_free_guidance_prob = classifier_free_guidance_prob
        self.img_size = img_size

        self.infinite = infinite

        # Variables for checkpointing
        self._sample_idx = 0
        self._all_samples: list[dict[str, Any]] = []

    def _get_data_iter(self):
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))

        return iter(self._data)

    def __iter__(self):
        dataset_iterator = self._get_data_iter()
        while True:
            # TODO: Add support for robust data loading and error handling.
            # Currently, we assume the dataset is well-formed and does not contain corrupted samples.
            # If a corrupted sample is encountered, the program will crash and throw an exception.
            # You can NOT try to catch the exception and continue, because the iterator within dataset
            # is not broken after raising an exception, so calling next() will throw StopIteration and might cause re-loop.
            try:
                sample = next(dataset_iterator)
            except StopIteration:
                # We are asumming the program hits here only when reaching the end of the dataset.
                if not self.infinite:
                    logger.warning(
                        f"Dataset {self.dataset_name} has run out of data. \
                         This might cause NCCL timeout if data parallelism is enabled."
                    )
                    break
                else:
                    # Reset offset for the next iteration if infinite
                    self._sample_idx = 0
                    logger.warning(f"Dataset {self.dataset_name} is being re-looped.")
                    dataset_iterator = self._get_data_iter()
                    if not isinstance(self._data, Dataset):
                        if hasattr(self._data, "set_epoch") and hasattr(
                            self._data, "epoch"
                        ):
                            self._data.set_epoch(self._data.epoch + 1)
                    continue

            # Use the dataset-specific preprocessor
            sample_dict = self._data_processor(
                sample,
                self._t5_tokenizer,
                self._clip_tokenizer,
                output_size=self.img_size,
            )

            # skip low quality image or image with color channel = 1
            if sample_dict["image"] is None:
                # pyrefly: ignore [missing-attribute]
                sample = sample.get("__key__", "unknown")
                logger.warning(
                    f"Low quality image {sample} is skipped in Flux Dataloader."
                )
                continue

            # Classifier-free guidance: Replace some of the strings with empty strings.
            # Distinct random seed is initialized at the beginning of training for each FSDP rank.
            # pyrefly: ignore [missing-attribute]
            dropout_prob = self.classifier_free_guidance_prob
            if dropout_prob > 0.0:
                if torch.rand(1).item() < dropout_prob:
                    sample_dict["t5_tokens"] = self._t5_empty_token
                if torch.rand(1).item() < dropout_prob:
                    sample_dict["clip_tokens"] = self._clip_empty_token

            self._sample_idx += 1

            labels = sample_dict.pop("image")

            yield sample_dict, labels

    def load_state_dict(self, state_dict):
        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        if isinstance(self._data, Dataset):
            return {"sample_idx": self._sample_idx}
        else:
            return {"data": self._data.state_dict()}


class FluxValidationDataset(FluxDataset):
    """
    Adds logic to generate timesteps for flux validation method described in SD3 paper

    Args:
    generate_timesteps (bool): Generate stratified timesteps in round-robin style for validation
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        t5_tokenizer: BaseTokenizer,
        clip_tokenizer: BaseTokenizer,
        classifier_free_guidance_prob: float,
        img_size: int,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        generate_timesteps: bool = True,
        infinite: bool = False,
    ) -> None:
        # Call parent constructor correctly
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            t5_tokenizer=t5_tokenizer,
            clip_tokenizer=clip_tokenizer,
            classifier_free_guidance_prob=classifier_free_guidance_prob,
            img_size=img_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=infinite,
        )

        # Initialize timestep generation for validation
        self.generate_timesteps = generate_timesteps
        if self.generate_timesteps:
            # Generate stratified timesteps as described in SD3 paper
            val_timesteps = [1 / 8 * (i + 0.5) for i in range(8)]
            self.timestep_cycle = itertools.cycle(val_timesteps)

    def __iter__(self):
        # Get parent iterator and add timesteps to each sample
        parent_iterator = super().__iter__()

        for sample_dict, labels in parent_iterator:
            # Add timestep to the sample dict if timestep generation is enabled
            if self.generate_timesteps:
                sample_dict["timestep"] = next(self.timestep_cycle)

            yield sample_dict, labels


class MLPerfCocoValidationDataset(IterableDataset, Stateful):
    """Finite, deterministic MLPerf Flux validation dataset.

    The MLPerf ``flux-1-coco`` download contains WebDataset tar shards and the
    accompanying ``val2014_30k.tsv`` manifest. The manifest is the source of
    truth for the fixed image/caption pairs and their assigned integer
    timesteps.
    """

    def __init__(
        self,
        *,
        dataset_path: str,
        manifest_path: str,
        t5_tokenizer: FluxTokenizer,
        clip_tokenizer: FluxTokenizer,
        img_size: int,
        dp_rank: int,
        dp_world_size: int,
    ) -> None:
        if dp_world_size <= 0:
            raise ValueError(f"dp_world_size must be positive, got {dp_world_size}")
        if MLPERF_COCO_VALIDATION_SAMPLES % dp_world_size != 0:
            raise ValueError(
                "MLPerf validation requires 29,696 samples to divide evenly across "
                f"the data-parallel world size, got {dp_world_size}"
            )

        self._t5_tokenizer = t5_tokenizer
        self._clip_tokenizer = clip_tokenizer
        self._img_size = img_size
        self._shard_metadata_by_image_id = self._index_shards(Path(dataset_path))
        records = self._load_manifest(Path(manifest_path))
        missing_image_ids = {
            row["image_id"] for row in records
        } - self._shard_metadata_by_image_id.keys()
        if missing_image_ids:
            sample_ids = sorted(missing_image_ids)[:5]
            raise ValueError(
                "MLPerf validation shards are missing manifest image IDs "
                f"(first five): {sample_ids}"
            )
        for row in records:
            shard_timestep = self._shard_metadata_by_image_id[row["image_id"]][1]
            manifest_timestep = row.get("timestep")
            if manifest_timestep not in (None, "") and int(manifest_timestep) != shard_timestep:
                raise ValueError(
                    "MLPerf validation timestep differs between manifest and shard "
                    f"metadata for image_id {row['image_id']}"
                )
            row["timestep"] = shard_timestep
        self._validate_timestep_counts(records)
        self._records = records[dp_rank::dp_world_size]
        expected_per_rank = MLPERF_COCO_VALIDATION_SAMPLES // dp_world_size
        if len(self._records) != expected_per_rank:
            raise RuntimeError(
                "MLPerf validation rank received an unexpected sample count: "
                f"expected {expected_per_rank}, got {len(self._records)}"
            )

    @staticmethod
    def _load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "MLPerf validation manifest is required but was not found: "
                f"{manifest_path}"
            )

        with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
            rows = list(csv.DictReader(manifest_file, delimiter="\t"))

        required_columns = {"image_id", "caption"}
        if not rows or not required_columns.issubset(rows[0]):
            raise ValueError(
                f"MLPerf validation manifest {manifest_path} must contain "
                f"{sorted(required_columns)}, got {list(rows[0]) if rows else []}"
            )
        if len(rows) != MLPERF_COCO_VALIDATION_SAMPLES:
            raise ValueError(
                "MLPerf validation manifest must contain exactly "
                f"{MLPERF_COCO_VALIDATION_SAMPLES} rows, got {len(rows)}"
            )

        image_ids: set[int] = set()
        for row in rows:
            try:
                image_id = int(row["image_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid image_id in {manifest_path}: {row}"
                ) from exc
            if image_id in image_ids:
                raise ValueError(
                    "MLPerf validation manifest must contain one caption per image; "
                    f"duplicate image_id {image_id}"
                )
            image_ids.add(image_id)
            row["image_id"] = image_id
        return rows

    @staticmethod
    def _index_shards(dataset_path: Path) -> dict[int, tuple[Path, int]]:
        if not dataset_path.is_dir():
            raise FileNotFoundError(
                "MLPerf validation WebDataset directory is required but was not "
                f"found: {dataset_path}"
            )
        shards = sorted(dataset_path.glob("*.tar"))
        if not shards:
            raise FileNotFoundError(
                f"No WebDataset tar shards were found in {dataset_path}"
            )

        metadata_by_image_id: dict[int, tuple[Path, int]] = {}
        for shard in shards:
            with tarfile.open(shard, "r") as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith(".json"):
                        continue
                    try:
                        image_id = int(Path(member.name).stem)
                    except ValueError as exc:
                        raise ValueError(
                            f"Unexpected metadata filename {member.name} in {shard}"
                        ) from exc
                    metadata_file = archive.extractfile(member)
                    if metadata_file is None:
                        raise ValueError(
                            f"Unable to read metadata {member.name} in {shard}"
                        )
                    try:
                        metadata = json.load(metadata_file)
                        timestep = int(metadata["timestep"])
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise ValueError(
                            f"Invalid MLPerf metadata {member.name} in {shard}"
                        ) from exc
                    if not 0 <= timestep < MLPERF_COCO_TIMESTEPS:
                        raise ValueError(
                            f"MLPerf validation timestep must be in [0, 7], got "
                            f"{timestep} for image_id {image_id}"
                        )
                    if image_id in metadata_by_image_id:
                        raise ValueError(
                            f"Duplicate image_id {image_id} across MLPerf shards"
                        )
                    metadata_by_image_id[image_id] = (shard, timestep)
        return metadata_by_image_id

    @staticmethod
    def _validate_timestep_counts(records: list[dict[str, Any]]) -> None:
        timestep_counts = [0] * MLPERF_COCO_TIMESTEPS
        for row in records:
            timestep_counts[row["timestep"]] += 1
        expected_per_timestep = (
            MLPERF_COCO_VALIDATION_SAMPLES // MLPERF_COCO_TIMESTEPS
        )
        if timestep_counts != [expected_per_timestep] * MLPERF_COCO_TIMESTEPS:
            raise ValueError(
                "MLPerf validation shards must have an equal number of samples "
                f"per timestep ({expected_per_timestep}), got {timestep_counts}"
            )

    def __iter__(self):
        open_shard: Path | None = None
        archive: tarfile.TarFile | None = None
        try:
            for row in self._records:
                image_id = row["image_id"]
                shard_metadata = self._shard_metadata_by_image_id.get(image_id)
                if shard_metadata is None:
                    raise RuntimeError(
                        f"MLPerf validation image_id {image_id} is absent from shards"
                    )
                shard = shard_metadata[0]
                if shard != open_shard:
                    if archive is not None:
                        archive.close()
                    archive = tarfile.open(shard, "r")
                    open_shard = shard

                image_member = archive.extractfile(f"{image_id:012d}.png")
                if image_member is None:
                    raise RuntimeError(
                        f"MLPerf validation image_id {image_id} has no PNG payload"
                    )
                with PIL.Image.open(io.BytesIO(image_member.read())) as image:
                    processed_image = _process_cc12m_image(
                        image, output_size=self._img_size
                    )
                if processed_image is None:
                    raise RuntimeError(
                        f"MLPerf validation image_id {image_id} could not be decoded"
                    )

                prompt = row["caption"]
                yield {
                    "clip_tokens": self._clip_tokenizer.encode(prompt),
                    "t5_tokens": self._t5_tokenizer.encode(prompt),
                    "prompt": prompt,
                    "sample_key": str(image_id),
                    "timestep": row["timestep"],
                }, processed_image
        finally:
            if archive is not None:
                archive.close()

    def state_dict(self) -> dict[str, Any]:
        # Validation is rebuilt from the fixed manifest for every evaluation.
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        del state_dict


class FluxDataLoader(ParallelAwareDataloader):
    """Configurable Flux dataloader for both training and validation.

    This dataloader wraps FluxDataset (or FluxValidationDataset when
    ``generate_timesteps`` is enabled) and can be used for both training
    and validation by configuring the appropriate dataset, batch_size, etc.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset: str = "cc12m-test"
        """Dataset to use"""

        infinite: bool = True
        """Whether to loop the dataset infinitely"""

        # TODO: In validation it should always be 0.0. Find a way to enforce this.
        classifier_free_guidance_prob: float = 0.0
        """Classifier-free guidance with probability `p` to dropout each text encoding independently.
        If `n` text encoders are used, the unconditional model is trained in `p ^ n` of all steps.
        For example, if `n = 2` and `p = 0.447`, the unconditional model is trained in 20% of all steps"""

        img_size: int = 256
        """Image width to sample"""

        generate_timesteps: bool = False
        """Generate stratified timesteps in round-robin style (for validation)"""

        mlperf_manifest_path: str | None = None
        """Path to MLPerf's val2014_30k.tsv manifest for fixed COCO validation."""

        # TODO: Remove after the tokenizer is properly built from the trainer. E.g.,
        # we can have a tokenizer container which holds the t5 and clip tokenizers.
        encoder: Encoder = field(default_factory=Encoder)
        """This is a hack to get the T5 and CLIP tokenizer asset paths. Ideally tokenizer should be
        built from the trainer, not inside dataloader. The reason we are doing this is because FLUX
        has two tokenizer instead of just one."""

        hf_assets_path: str = "./tests/assets/tokenizer"
        """Similar to above, this is a hack to get the test tokenizer asset paths."""

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        local_batch_size: int,
        **kwargs,
    ):

        t5_tokenizer, clip_tokenizer = build_flux_tokenizer(
            encoder_config=config.encoder,
            hf_assets_path=config.hf_assets_path,
        )

        if config.dataset == "mlperf-coco-validation":
            if config.dataset_path is None or config.mlperf_manifest_path is None:
                raise ValueError(
                    "mlperf-coco-validation requires both dataloader.dataset_path "
                    "(the flux-1-coco tar directory) and "
                    "dataloader.mlperf_manifest_path (val2014_30k.tsv)"
                )
            ds = MLPerfCocoValidationDataset(
                dataset_path=config.dataset_path,
                manifest_path=config.mlperf_manifest_path,
                t5_tokenizer=t5_tokenizer,
                clip_tokenizer=clip_tokenizer,
                img_size=config.img_size,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
            )
        elif config.generate_timesteps:
            ds = FluxValidationDataset(
                dataset_name=config.dataset,
                dataset_path=config.dataset_path,
                t5_tokenizer=t5_tokenizer,
                clip_tokenizer=clip_tokenizer,
                classifier_free_guidance_prob=config.classifier_free_guidance_prob,
                img_size=config.img_size,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
                generate_timesteps=True,
                infinite=config.infinite,
            )
        else:
            ds = FluxDataset(
                dataset_name=config.dataset,
                dataset_path=config.dataset_path,
                t5_tokenizer=t5_tokenizer,
                clip_tokenizer=clip_tokenizer,
                classifier_free_guidance_prob=config.classifier_free_guidance_prob,
                img_size=config.img_size,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
                infinite=config.infinite,
            )

        dataloader_kwargs = {
            "num_workers": config.num_workers,
            "persistent_workers": config.persistent_workers,
            "pin_memory": config.pin_memory,
            "prefetch_factor": config.prefetch_factor,
            "batch_size": local_batch_size,
        }

        super().__init__(
            ds,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            **dataloader_kwargs,
        )
