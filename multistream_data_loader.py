"""Data loading for the RGB, edge, and LBP multi-signal model."""

from pathlib import Path
import hashlib

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.banknote.preprocessing.prepare_streams import (
    prepare_banknote_streams,
)


CLASS_LABELS = {
    "counterfeit": 0,
    "genuine": 1,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = ("train_set", "validation_set", "test_set")


def find_dataset_dir(data_root=Path("data/raw")):
    """Find the project's labelled dataset, or require an explicit path."""
    data_root = Path(data_root)
    matches = sorted(
        data_root.glob(
            "Ugandan banknote*/labelled_ugandan_banknote_dataset_clean"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one dataset folder under {data_root}, found {len(matches)}. "
            "Pass --dataset-dir with the exact labelled dataset directory."
        )
    return matches[0]


class MultiStreamBanknoteDataset(Dataset):
    """Load one image as aligned RGB, edge, and LBP tensors."""

    def __init__(self, dataset_dir, split, target_size=224):
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split: {split}")

        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.target_size = target_size
        self.samples = []

        split_dir = self.dataset_dir / "splits" / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        for class_name, label in CLASS_LABELS.items():
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")

            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((image_path, label))

        if not self.samples:
            raise ValueError(f"No supported images found in {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        streams = prepare_banknote_streams(
            str(image_path), target_size=self.target_size
        )

        rgb = torch.from_numpy(
            np.ascontiguousarray(streams["rgb"].transpose(2, 0, 1))
        ).float().div_(255.0)

        edges = torch.from_numpy(
            np.ascontiguousarray(streams["edges"][None, :, :])
        ).float().div_(255.0)

        texture = torch.from_numpy(
            np.ascontiguousarray(streams["texture"][None, :, :])
        ).float().div_(255.0)

        return {
            "rgb": rgb,
            "edges": edges,
            "texture": texture,
            "label": torch.tensor(label, dtype=torch.long),
        }


def create_multistream_loaders(
    dataset_dir, batch_size=8, seed=42, target_size=224
):
    """Create reproducibly shuffled training and unshuffled validation loaders."""
    train_dataset = MultiStreamBanknoteDataset(
        dataset_dir, "train_set", target_size
    )
    validation_dataset = MultiStreamBanknoteDataset(
        dataset_dir, "validation_set", target_size
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_dataset, validation_dataset, train_loader, validation_loader


def write_split_manifest(dataset_dir, output_path):
    """Record split membership and hashes without decoding the images."""
    import csv

    dataset_dir = Path(dataset_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "relative_path", "class", "label", "sha256"])

        for split in SPLIT_NAMES:
            split_dir = dataset_dir / "splits" / split
            if not split_dir.is_dir():
                raise FileNotFoundError(f"Missing split directory: {split_dir}")

            for class_name, label in CLASS_LABELS.items():
                class_dir = split_dir / class_name
                if not class_dir.is_dir():
                    raise FileNotFoundError(f"Missing class directory: {class_dir}")

                for image_path in sorted(class_dir.iterdir()):
                    if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                        digest = hashlib.sha256()
                        with image_path.open("rb") as image_file:
                            for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                                digest.update(chunk)
                        writer.writerow([
                            split,
                            image_path.relative_to(dataset_dir).as_posix(),
                            class_name,
                            label,
                            digest.hexdigest(),
                        ])
