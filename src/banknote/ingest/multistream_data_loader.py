"""Data loading for the RGB, edge, and LBP multi-signal model."""

from pathlib import Path
import csv
import hashlib

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from src.banknote.preprocessing.cache_streams import stream_cache_path

from src.banknote.preprocessing.prepare_streams import (
    prepare_banknote_streams,
)


CLASS_LABELS = {
    "counterfeit": 0,
    "genuine": 1,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = ("train_set", "validation_set", "test_set")
CONDITION_NAMES = ("soiled", "creased", "torn", "folded")


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

    def __init__(self, dataset_dir, split, target_size=224, cache_dir=None):
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split: {split}")

        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.target_size = target_size
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
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
                    cache_path = None
                    if self.cache_dir is not None:
                        cache_path = stream_cache_path(
                            image_path, self.cache_dir, target_size
                        )
                        if not cache_path.is_file():
                            raise FileNotFoundError(
                                f"Missing cached streams for {image_path}. "
                                "Precompute the selected split first."
                            )
                    self.samples.append((image_path, label, cache_path))

        if not self.samples:
            raise ValueError(f"No supported images found in {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, cache_path = self.samples[index]
        if cache_path is None:
            streams = prepare_banknote_streams(
                str(image_path), target_size=self.target_size
            )
        else:
            with np.load(cache_path, allow_pickle=False) as cached:
                streams = {
                    name: cached[name].copy()
                    for name in ("rgb", "edges", "texture")
                }

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


class MultiTaskMultiStreamBanknoteDataset(MultiStreamBanknoteDataset):
    """Multi-signal samples paired with four independent condition targets."""

    def __init__(
        self,
        dataset_dir,
        split,
        annotations_csv=Path("annotations/condition_labels_template.csv"),
        target_size=224,
        cache_dir=None,
    ):
        super().__init__(dataset_dir, split, target_size, cache_dir=cache_dir)

        project_root = Path(__file__).resolve().parents[3]
        annotations_csv = Path(annotations_csv)
        if not annotations_csv.is_absolute():
            annotations_csv = project_root / annotations_csv
        if not annotations_csv.is_file():
            raise FileNotFoundError(
                f"Condition annotation CSV not found: {annotations_csv}"
            )

        rows_by_id = {}
        with annotations_csv.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {
                "image_id", "image_path", "split", "authenticity_label",
                "soiled", "creased", "torn", "folded", "annotation_status",
            }
            missing_columns = required.difference(reader.fieldnames or ())
            if missing_columns:
                raise ValueError(
                    "Condition annotation CSV is missing columns: "
                    + ", ".join(sorted(missing_columns))
                )
            for row in reader:
                image_id = row["image_id"].strip()
                if not image_id:
                    raise ValueError("Condition annotation CSV contains a blank image_id")
                if image_id in rows_by_id:
                    raise ValueError(f"Duplicate condition annotation image_id: {image_id}")
                if row["annotation_status"].strip() != "complete":
                    raise ValueError(f"Condition annotation is not complete: {image_id}")

                values, masks = [], []
                for name in CONDITION_NAMES:
                    value = row[name].strip().lower()
                    if value == "yes":
                        values.append(1.0)
                        masks.append(1.0)
                    elif value == "no":
                        values.append(0.0)
                        masks.append(1.0)
                    elif value == "uncertain":
                        # Unknown for this tag, not a negative example.
                        values.append(0.0)
                        masks.append(0.0)
                    else:
                        raise ValueError(
                            f"Invalid {name} tag {value!r} for {image_id}; "
                            "expected yes, no, or uncertain"
                        )
                rows_by_id[image_id] = {
                    "row": row,
                    "values": values,
                    "masks": masks,
                }

        split_rows = {
            image_id: item for image_id, item in rows_by_id.items()
            if item["row"]["split"].strip() == split
        }
        dataset_ids = {image_path.stem for image_path, _, _ in self.samples}
        missing = sorted(dataset_ids.difference(split_rows))
        extra = sorted(set(split_rows).difference(dataset_ids))
        if missing or extra:
            raise ValueError(
                f"Condition labels do not match {split}: "
                f"missing={len(missing)}, extra={len(extra)}"
            )

        self.condition_targets = {}
        self.condition_masks = {}
        self.condition_counts = {
            name: {"yes": 0, "no": 0, "uncertain": 0}
            for name in CONDITION_NAMES
        }
        for image_path, authenticity_label, _ in self.samples:
            image_id = image_path.stem
            item = split_rows[image_id]
            row = item["row"]
            expected_authenticity = (
                "counterfeit" if authenticity_label == 0 else "genuine"
            )
            if row["authenticity_label"].strip() != expected_authenticity:
                raise ValueError(
                    f"Authenticity mismatch between image folder and annotations: {image_id}"
                )
            csv_path = Path(row["image_path"].replace("\\", "/"))
            if csv_path.parts[-4:] != (
                "splits", split, expected_authenticity, image_path.name
            ):
                raise ValueError(
                    f"Image path/split mismatch in condition annotations: {image_id}"
                )
            self.condition_targets[image_id] = item["values"]
            self.condition_masks[image_id] = item["masks"]
            for index, name in enumerate(CONDITION_NAMES):
                tag = row[name].strip().lower()
                self.condition_counts[name][tag] += 1

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        image_path = self.samples[index][0]
        image_id = image_path.stem
        sample["condition_targets"] = torch.tensor(
            self.condition_targets[image_id], dtype=torch.float32
        )
        sample["condition_mask"] = torch.tensor(
            self.condition_masks[image_id], dtype=torch.float32
        )
        return sample


def create_multitask_multistream_loaders(
    dataset_dir,
    annotations_csv=Path("annotations/condition_labels_template.csv"),
    batch_size=8,
    seed=42,
    target_size=224,
    cache_dir=None,
):
    """Create train and validation loaders; the held-out test split is untouched."""
    train_dataset = MultiTaskMultiStreamBanknoteDataset(
        dataset_dir, "train_set", annotations_csv, target_size, cache_dir
    )
    validation_dataset = MultiTaskMultiStreamBanknoteDataset(
        dataset_dir, "validation_set", annotations_csv, target_size, cache_dir
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


def create_multistream_loaders(
    dataset_dir, batch_size=8, seed=42, target_size=224, cache_dir=None
):
    """Create reproducibly shuffled training and unshuffled validation loaders."""
    train_dataset = MultiStreamBanknoteDataset(
        dataset_dir, "train_set", target_size, cache_dir=cache_dir
    )
    validation_dataset = MultiStreamBanknoteDataset(
        dataset_dir, "validation_set", target_size, cache_dir=cache_dir
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
