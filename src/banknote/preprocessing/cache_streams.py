"""Precompute deterministic RGB, edge, and texture streams for faster training.

Run from the project root:
    python -m src.banknote.preprocessing.cache_streams --dataset-dir "data/raw/..."

The default caches only training and validation images. Add test_set explicitly
after model selection is complete.
"""

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np

from src.banknote.preprocessing.prepare_streams import prepare_banknote_streams


CACHE_VERSION = 1
CLASS_LABELS = {"counterfeit": 0, "genuine": 1}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = ("train_set", "validation_set", "test_set")


def stream_cache_path(image_path, cache_dir, target_size):
    """Return a cache path tied to source bytes and preprocessing settings."""
    digest = hashlib.sha256()
    with Path(image_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    cache_key = hashlib.sha256(
        f"v{CACHE_VERSION}:{target_size}:{digest.hexdigest()}".encode("ascii")
    ).hexdigest()
    return Path(cache_dir) / f"{cache_key}.npz"


def precompute_stream_cache(
    dataset_dir,
    cache_dir,
    target_size=224,
    splits=("train_set", "validation_set"),
):
    dataset_dir = Path(dataset_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split: {split}")
        split_dir = dataset_dir / "splits" / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        cached_count = 0
        reused_count = 0
        for class_name in CLASS_LABELS:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            for image_path in sorted(class_dir.iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                output_path = stream_cache_path(
                    image_path, cache_dir, target_size
                )
                if output_path.is_file():
                    reused_count += 1
                    continue
                streams = prepare_banknote_streams(
                    str(image_path), target_size=target_size
                )
                temporary = output_path.with_suffix(".npz.tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(
                        handle,
                        rgb=streams["rgb"],
                        edges=streams["edges"],
                        texture=streams["texture"],
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, output_path)
                cached_count += 1
        print(
            f"{split}: created {cached_count} cache files; "
            f"reused {reused_count}"
        )
    print(f"Cache directory: {cache_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/processed/streams_v1_224"),
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLIT_NAMES,
        default=("train_set", "validation_set"),
    )
    args = parser.parse_args()
    if args.image_size < 1:
        raise ValueError("image-size must be positive")
    precompute_stream_cache(
        args.dataset_dir,
        args.cache_dir,
        target_size=args.image_size,
        splits=tuple(args.splits),
    )


if __name__ == "__main__":
    main()
