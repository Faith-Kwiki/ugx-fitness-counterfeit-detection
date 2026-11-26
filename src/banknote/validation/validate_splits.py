"""Check split class counts and detect exact image duplicates across splits."""

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

from src.banknote.ingest.multistream_data_loader import (
    CLASS_LABELS,
    IMAGE_EXTENSIONS,
    SPLIT_NAMES,
    find_dataset_dir,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_splits(dataset_dir):
    dataset_dir = Path(dataset_dir)
    image_hashes = defaultdict(list)
    counts = {}

    for split in SPLIT_NAMES:
        counts[split] = {}
        split_dir = dataset_dir / "splits" / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        for class_name in CLASS_LABELS:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            images = [
                path for path in sorted(class_dir.iterdir())
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if not images:
                raise ValueError(f"No images found in {class_dir}")
            counts[split][class_name] = len(images)
            for path in images:
                image_hashes[sha256_file(path)].append(
                    (split, class_name, path.relative_to(dataset_dir).as_posix())
                )

    cross_split = []
    label_conflicts = []
    within_split = []
    for entries in image_hashes.values():
        splits = {entry[0] for entry in entries}
        labels = {entry[1] for entry in entries}
        if len(labels) > 1:
            label_conflicts.append(entries)
        if len(splits) > 1:
            cross_split.append(entries)
        elif len(entries) > 1:
            within_split.append(entries)

    print("Image counts by split and class:")
    for split, classes in counts.items():
        print(
            f"{split}: counterfeit={classes['counterfeit']}, "
            f"genuine={classes['genuine']}, total={sum(classes.values())}"
        )
    print(f"Exact duplicate groups within a split: {len(within_split)}")
    if label_conflicts:
        print("Exact duplicate groups assigned conflicting labels:")
        for group in label_conflicts:
            for entry in group:
                print("  ", entry)
        raise ValueError("Resolve exact duplicate images with conflicting labels.")
    if cross_split:
        print("Exact duplicate image groups crossing split boundaries:")
        for group in cross_split:
            for entry in group:
                print("  ", entry)
        raise ValueError("Exact duplicate images cross train/validation/test splits.")
    print("No exact image duplicates cross split boundaries.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    dataset_dir = (
        args.dataset_dir
        if args.dataset_dir is not None
        else find_dataset_dir(args.data_root)
    )
    validate_splits(dataset_dir)


if __name__ == "__main__":
    main()
