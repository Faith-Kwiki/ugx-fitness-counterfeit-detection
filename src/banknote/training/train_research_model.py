"""Run-isolated training for RGB, ResNet-18, and hybrid candidates.

This trainer uses train and validation splits only. It never loads test images.
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW

from src.banknote.ingest.multistream_data_loader import (
    create_multistream_loaders,
    find_dataset_dir,
    write_split_manifest,
)
from src.banknote.models.research_models import build_research_model


MODEL_NAMES = (
    "rgb_cnn",
    "resnet18_frozen",
    "resnet18_finetune",
    "hybrid_concat",
    "hybrid_attention",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/research"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--backbone-learning-rate", type=float, default=0.00001)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def atomic_json_write(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_write(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".pth.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def make_optimizer(model, args):
    if args.model == "resnet18_finetune":
        backbone = model.backbone
        return AdamW(
            [
                {"params": backbone.layer3.parameters(), "lr": args.backbone_learning_rate},
                {"params": backbone.layer4.parameters(), "lr": args.backbone_learning_rate},
                {"params": backbone.fc.parameters(), "lr": args.learning_rate},
            ],
            weight_decay=0.01,
        )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if args.model in {"resnet18_frozen", "hybrid_concat", "hybrid_attention"}:
        return AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    return Adam(parameters, lr=args.learning_rate)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            rgb = batch["rgb"].to(device)
            edges = batch["edges"].to(device)
            texture = batch["texture"].to(device)
            labels = batch["label"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(rgb, edges, texture)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            size = labels.size(0)
            total_loss += loss.item() * size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += size
    return total_loss / count, correct / count


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 1:
        raise ValueError("epochs, batch-size, and image-size must be positive")
    if args.learning_rate <= 0 or args.backbone_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        raise ValueError("run-id may contain only letters, numbers, dot, underscore, and hyphen")

    set_seed(args.seed)
    dataset_dir = (
        Path(args.dataset_dir) if args.dataset_dir is not None
        else find_dataset_dir(args.data_root)
    ).resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    train_set, validation_set, train_loader, validation_loader = create_multistream_loaders(
        dataset_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        target_size=args.image_size,
        cache_dir=args.cache_dir,
    )
    run_dir = args.runs_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_split_manifest(dataset_dir, run_dir / "split_manifest.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_research_model(args.model, pretrained=True).to(device)
    if args.init_checkpoint is not None:
        initial = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        model.load_state_dict(initial["model_state_dict"])
    optimizer = make_optimizer(model, args)
    criterion = nn.CrossEntropyLoss()

    config = {
        "run_id": args.run_id,
        "model": args.model,
        "architecture": model.__class__.__name__,
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "optimizer": optimizer.__class__.__name__,
        "scheduler": None,
        "class_mapping": {"counterfeit": 0, "genuine": 1},
        "pretrained_weights": (
            "ResNet18_Weights.DEFAULT"
            if args.model in {
                "resnet18_frozen", "resnet18_finetune",
                "hybrid_concat", "hybrid_attention"
            }
            else None
        ),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "preprocessing_cache": (
            str(args.cache_dir.resolve()) if args.cache_dir is not None else None
        ),
        "fusion": (
            "attention" if args.model == "hybrid_attention"
            else "concatenation" if args.model == "hybrid_concat"
            else None
        ),
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "backbone_learning_rate": args.backbone_learning_rate,
        "image_size": args.image_size,
        "dataset_folder_name": dataset_dir.name,
        "train_images": len(train_set),
        "validation_images": len(validation_set),
        "init_checkpoint": (
            str(args.init_checkpoint.resolve()) if args.init_checkpoint else None
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "selection_rule": "lowest validation cross-entropy loss",
        "test_set_used_for_training_or_selection": False,
    }
    atomic_json_write(run_dir / "config.json", config)
    summary = {
        "run_id": args.run_id,
        "status": "running",
        "last_completed_epoch": 0,
        "best_epoch": None,
        "best_val_loss": None,
        "best_val_accuracy": None,
    }
    summary_path = run_dir / "summary.json"
    atomic_json_write(summary_path, summary)
    history_path = run_dir / "history.csv"
    history_path.write_text(
        "epoch,train_loss,train_accuracy,val_loss,val_accuracy,epoch_seconds,is_best\n",
        encoding="utf-8",
    )
    best_path = run_dir / "best_checkpoint.pth"
    latest_path = run_dir / "latest_checkpoint.pth"
    best_val_loss = float("inf")

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            train_loss, train_accuracy = run_epoch(
                model, train_loader, criterion, device, optimizer
            )
            val_loss, val_accuracy = run_epoch(
                model, validation_loader, criterion, device
            )
            epoch_seconds = time.perf_counter() - epoch_start
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                summary["best_epoch"] = epoch
                summary["best_val_loss"] = val_loss
                summary["best_val_accuracy"] = val_accuracy
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "epoch_seconds": epoch_seconds,
                "is_best": is_best,
            }
            with history_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=row.keys())
                writer.writerow(row)
                handle.flush()
                os.fsync(handle.fileno())
            checkpoint = {
                "format_version": 1,
                "run_id": args.run_id,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "config": config,
            }
            atomic_torch_write(latest_path, checkpoint)
            if is_best:
                atomic_torch_write(best_path, checkpoint)
            summary["last_completed_epoch"] = epoch
            atomic_json_write(summary_path, summary)
            print(
                f"Epoch {epoch:02d}/{args.epochs}: train loss {train_loss:.4f}, "
                f"accuracy {train_accuracy:.4f}; val loss {val_loss:.4f}, "
                f"accuracy {val_accuracy:.4f}; {epoch_seconds:.2f}s"
            )
        summary["status"] = "completed"
        atomic_json_write(summary_path, summary)
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        atomic_json_write(summary_path, summary)
        raise
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        atomic_json_write(summary_path, summary)
        raise


if __name__ == "__main__":
    main()
