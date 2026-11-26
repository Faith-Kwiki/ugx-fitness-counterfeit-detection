"""Reproducible RGB + edges + LBP training with run-isolated artifacts.

Run from the project root:
    python -m src.banknote.training.train_multisignal --run-id concat-seed42

This script uses training and validation data only. It deliberately does not
load or evaluate the test split.
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.banknote.ingest.multistream_data_loader import (
    create_multistream_loaders,
    find_dataset_dir,
    write_split_manifest,
)
from src.banknote.models.multisignal_cnn import MultiSignalCNN


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/multisignal"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional precomputed RGB/edge/texture cache directory.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ask cuDNN for repeatable operations where available.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def atomic_json_write(path, payload):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def atomic_checkpoint_write(path, payload):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for batch in loader:
        rgb = batch["rgb"].to(device)
        edges = batch["edges"].to(device)
        texture = batch["texture"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(rgb, edges, texture)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        count = labels.size(0)
        loss_sum += loss.item() * count
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += count

    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate_validation(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    for batch in loader:
        rgb = batch["rgb"].to(device)
        edges = batch["edges"].to(device)
        texture = batch["texture"].to(device)
        labels = batch["label"].to(device)

        logits = model(rgb, edges, texture)
        loss = criterion(logits, labels)
        count = labels.size(0)
        loss_sum += loss.item() * count
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += count

    return loss_sum / total, correct / total


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 1:
        raise ValueError("epochs, batch-size, and image-size must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")

    set_seed(args.seed)
    dataset_dir = (
        Path(args.dataset_dir)
        if args.dataset_dir is not None
        else find_dataset_dir(args.data_root)
    ).resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{timestamp}_seed{args.seed}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("run-id may contain only letters, numbers, dot, underscore, and hyphen")

    train_dataset, validation_dataset, train_loader, validation_loader = (
        create_multistream_loaders(
            dataset_dir,
            batch_size=args.batch_size,
            seed=args.seed,
            target_size=args.image_size,
            cache_dir=args.cache_dir,
        )
    )

    run_dir = Path(args.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "history.csv").write_text(
        "epoch,train_loss,train_accuracy,val_loss,val_accuracy,is_best\n",
        encoding="utf-8",
    )
    write_split_manifest(dataset_dir, run_dir / "split_manifest.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiSignalCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    config = {
        "run_id": run_id,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "image_size": args.image_size,
        "preprocessing_cache": (
            str(args.cache_dir.resolve()) if args.cache_dir is not None else None
        ),
        "dataset_folder_name": dataset_dir.name,
        "train_images": len(train_dataset),
        "validation_images": len(validation_dataset),
        "python_version": sys.version.split()[0],
        "selection_rule": "lowest validation cross-entropy loss",
        "device": str(device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "split_manifest": "split_manifest.csv",
        "test_set_used_for_training_or_selection": False,
    }
    atomic_json_write(run_dir / "config.json", config)

    summary = {
        "run_id": run_id,
        "status": "running",
        "last_completed_epoch": 0,
        "best_epoch": None,
        "best_val_loss": None,
        "best_val_accuracy": None,
    }
    summary_path = run_dir / "summary.json"
    atomic_json_write(summary_path, summary)

    best_val_loss = float("inf")
    history_path = run_dir / "history.csv"
    best_path = run_dir / "best_checkpoint.pth"
    latest_path = run_dir / "latest_checkpoint.pth"

    print(f"Run: {run_id}")
    print(f"Dataset: {dataset_dir.name}")
    print(f"Device: {device}; seed: {args.seed}; epochs: {args.epochs}")
    print(f"Artifacts: {run_dir.resolve()}")

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_accuracy = evaluate_validation(
                model, validation_loader, criterion, device
            )
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "is_best": is_best,
            }
            with history_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=row.keys())
                writer.writerow(row)
                handle.flush()
                os.fsync(handle.fileno())

            checkpoint = {
                "format_version": 1,
                "run_id": run_id,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "config": config,
            }
            atomic_checkpoint_write(latest_path, checkpoint)
            if is_best:
                atomic_checkpoint_write(best_path, checkpoint)
                summary["best_epoch"] = epoch
                summary["best_val_loss"] = val_loss
                summary["best_val_accuracy"] = val_accuracy

            summary["last_completed_epoch"] = epoch
            atomic_json_write(summary_path, summary)
            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"train loss {train_loss:.4f}, acc {train_accuracy:.4f} | "
                f"val loss {val_loss:.4f}, acc {val_accuracy:.4f}"
                f"{' | best checkpoint updated' if is_best else ''}"
            )

        summary["status"] = "completed"
        atomic_json_write(summary_path, summary)
        print(f"Training complete. Best checkpoint: {best_path.resolve()}")
        print(f"History: {history_path.resolve()}")
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        atomic_json_write(summary_path, summary)
        print(
            "Training interrupted. Completed epochs, history, and checkpoints "
            f"are preserved in {run_dir.resolve()}"
        )
        raise
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        atomic_json_write(summary_path, summary)
        raise


if __name__ == "__main__":
    main()
