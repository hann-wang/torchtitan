import csv
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import PIL.Image
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from torchtitan.models.flux import flux_datasets
from torchtitan.models.flux.validate import compute_mlperf_validation_loss


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return [len(text)]


class TestMLPerfFluxValidation(unittest.TestCase):
    def test_fixed_manifest_is_evenly_sharded_and_preserves_timesteps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "val2014_30k.tsv"
            shard = root / "shard_00000.tar"
            image_bytes = io.BytesIO()
            PIL.Image.new("RGB", (256, 256)).save(image_bytes, format="PNG")

            with manifest.open("w", newline="", encoding="utf-8") as manifest_file:
                writer = csv.DictWriter(
                    manifest_file,
                    fieldnames=["image_id", "caption", "timestep"],
                    delimiter="\t",
                )
                writer.writeheader()
                for image_id in range(8):
                    writer.writerow(
                        {
                            "image_id": image_id,
                            "caption": f"caption {image_id}",
                        }
                    )

            with tarfile.open(shard, "w") as archive:
                for image_id in range(8):
                    payload = image_bytes.getvalue()
                    image_info = tarfile.TarInfo(f"{image_id:012d}.png")
                    image_info.size = len(payload)
                    archive.addfile(image_info, io.BytesIO(payload))
                    metadata = json.dumps(
                        {"id": image_id, "timestep": image_id}
                    ).encode()
                    metadata_info = tarfile.TarInfo(f"{image_id:012d}.json")
                    metadata_info.size = len(metadata)
                    archive.addfile(metadata_info, io.BytesIO(metadata))

            with (
                patch.object(
                    flux_datasets, "MLPERF_COCO_VALIDATION_SAMPLES", 8
                ),
                patch.object(flux_datasets, "MLPERF_COCO_TIMESTEPS", 8),
            ):
                rank_samples = []
                for rank in range(4):
                    dataset = flux_datasets.MLPerfCocoValidationDataset(
                        dataset_path=str(root),
                        manifest_path=str(manifest),
                        t5_tokenizer=_Tokenizer(),
                        clip_tokenizer=_Tokenizer(),
                        img_size=256,
                        dp_rank=rank,
                        dp_world_size=4,
                    )
                    loader = StatefulDataLoader(dataset, batch_size=2)
                    samples = []
                    for input_dict, labels in loader:
                        samples.extend(
                            (
                                {
                                    "timestep": timestep.item(),
                                    "sample_key": sample_key,
                                },
                                image,
                            )
                            for timestep, sample_key, image in zip(
                                input_dict["timestep"],
                                input_dict["sample_key"],
                                labels,
                            )
                        )
                    self.assertEqual(len(samples), 2)
                    rank_samples.extend(samples)

                self.assertEqual(len(rank_samples), 8)
                self.assertEqual(
                    sorted(sample[0]["timestep"] for sample in rank_samples),
                    list(range(8)),
                )
                self.assertEqual(
                    {sample[0]["sample_key"] for sample in rank_samples},
                    {str(image_id) for image_id in range(8)},
                )

    def test_mlperf_metric_averages_timestep_means(self):
        loss_sums = torch.tensor(
            [2.0, 8.0, 18.0, 32.0, 50.0, 72.0, 98.0, 128.0],
            dtype=torch.float64,
        )
        element_counts = torch.tensor([2.0] * 8, dtype=torch.float64)

        metric = compute_mlperf_validation_loss(loss_sums, element_counts)

        self.assertEqual(metric.item(), 25.5)


if __name__ == "__main__":
    unittest.main()
