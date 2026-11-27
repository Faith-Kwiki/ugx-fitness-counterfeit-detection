"""Train the shared multi-signal model with authenticity and four condition heads.

Run from the repository root:
    python -m src.banknote.training.train_multitask_multisignal --run-id condition-seed42

Only train and validation images are loaded. The held-out test split is not
used for training or model selection. "uncertain" condition tags are masked,
not converted to negative labels.
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.banknote.ingest.multistream_data_loader import (
    CONDITION_NAMES,
    create_multitask_multistream_loaders,
    find_dataset_dir,
    write_split_manifest,
)
from src.banknote.models.multitask_multisignal_cnn import MultiTaskMultiSignalCNN


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--annotations-csv",
        type=Path,
        default=Path("annotations/condition_labels_template.csv"),
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/multitask"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--condition-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--max-pos-weight",
        type=float,
        default=20.0,
        help="Cap for train-derived positive class weights on rare condition tags.",
    )
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


def make_positive_weights(train_dataset, maximum, device):
    weights = []
    for name in CONDITION_NAMES:
        counts = train_dataset.condition_counts[name]
        positives = counts["yes"]
        negatives = counts["no"]
        weight = negatives / positives if positives else 1.0
        weights.append(min(weight, maximum))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def calculate_loss(outputs, labels, targets, masks, criterion, pos_weight, condition_weight):
    authenticity_loss = criterion(outputs["authenticity_logits"], labels)
    element_losses = F.binary_cross_entropy_with_logits(
        outputs["condition_logits"],
        targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    known_per_tag = masks.sum(dim=0)
    loss_per_tag = (element_losses * masks).sum(dim=0) / known_per_tag.clamp_min(1.0)
    has_known_labels = (known_per_tag > 0).to(loss_per_tag.dtype)
    condition_loss = (loss_per_tag * has_known_labels).sum() / has_known_labels.sum().clamp_min(1.0)
    total_loss = authenticity_loss + condition_weight * condition_loss
    return total_loss, authenticity_loss, condition_loss


def train_epoch(model, loader, optimizer, device, pos_weight, condition_weight):
    model.train()
    criterion = nn.CrossEntropyLoss()
    sums = {"loss": 0.0, "authenticity_loss": 0.0, "condition_loss": 0.0}
    correct = total = 0

    for batch in loader:
        rgb = batch["rgb"].to(device)
        edges = batch["edges"].to(device)
        texture = batch["texture"].to(device)
        labels = batch["label"].to(device)
        targets = batch["condition_targets"].to(device)
        masks = batch["condition_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(rgb, edges, texture)
        loss, authenticity_loss, condition_loss = calculate_loss(
            outputs, labels, targets, masks, criterion, pos_weight, condition_weight
        )
        loss.backward()
        optimizer.step()

        count = labels.size(0)
        sums["loss"] += loss.item() * count
        sums["authenticity_loss"] += authenticity_loss.item() * count
        sums["condition_loss"] += condition_loss.item() * count
        correct += (outputs["authenticity_logits"].argmax(dim=1) == labels).sum().item()
        total += count

    return {
        **{name: value / total for name, value in sums.items()},
        "authenticity_accuracy": correct / total,
    }


def binary_metrics(targets, scores):
    targets = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = scores >= 0.5
    positives = targets == 1
    negatives = ~positives
    tp = int(np.sum(predictions & positives))
    fp = int(np.sum(predictions & negatives))
    fn = int(np.sum(~predictions & positives))
    tn = int(np.sum(~predictions & negatives))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else None

    average_precision = None
    positive_count = int(positives.sum())
    if positive_count:
        order = np.argsort(-scores, kind="stable")
        ranked_targets = targets[order]
        ranked_precision = np.cumsum(ranked_targets) / np.arange(1, len(targets) + 1)
        average_precision = float(ranked_precision[ranked_targets == 1].mean())

    return {
        "known_count": int(len(targets)),
        "positive_count": positive_count,
        "negative_count": int(negatives.sum()),
        "threshold": 0.5,
        "accuracy": float((tp + tn) / len(targets)) if len(targets) else None,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity) if specificity is not None else None,
        "average_precision": average_precision,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def validation_metrics(model, loader, device, pos_weight, condition_weight):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    sums = {"loss": 0.0, "authenticity_loss": 0.0, "condition_loss": 0.0}
    total = 0
    auth_targets, auth_scores, auth_predictions = [], [], []
    condition_targets = [[] for _ in CONDITION_NAMES]
    condition_scores = [[] for _ in CONDITION_NAMES]

    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            edges = batch["edges"].to(device)
            texture = batch["texture"].to(device)
            labels = batch["label"].to(device)
            targets = batch["condition_targets"].to(device)
            masks = batch["condition_mask"].to(device)
            outputs = model(rgb, edges, texture)
            loss, authenticity_loss, condition_loss = calculate_loss(
                outputs, labels, targets, masks, criterion, pos_weight, condition_weight
            )

            count = labels.size(0)
            sums["loss"] += loss.item() * count
            sums["authenticity_loss"] += authenticity_loss.item() * count
            sums["condition_loss"] += condition_loss.item() * count
            total += count

            auth_probabilities = outputs["authenticity_logits"].softmax(dim=1)
            auth_targets.extend(labels.cpu().numpy().tolist())
            auth_scores.extend(auth_probabilities[:, 0].cpu().numpy().tolist())
            auth_predictions.extend(outputs["authenticity_logits"].argmax(dim=1).cpu().numpy().tolist())

            probabilities = outputs["condition_logits"].sigmoid().cpu().numpy()
            target_values = targets.cpu().numpy()
            known_values = masks.cpu().numpy().astype(bool)
            for column in range(len(CONDITION_NAMES)):
                condition_targets[column].extend(target_values[known_values[:, column], column].astype(int).tolist())
                condition_scores[column].extend(probabilities[known_values[:, column], column].tolist())

    counterfeit = binary_metrics(
        [1 if label == 0 else 0 for label in auth_targets], auth_scores
    )
    conditions = {
        name: binary_metrics(condition_targets[column], condition_scores[column])
        for column, name in enumerate(CONDITION_NAMES)
    }
    return {
        **{name: value / total for name, value in sums.items()},
        "authenticity_accuracy": float(
            np.mean(np.asarray(auth_predictions) == np.asarray(auth_targets))
        ),
        "counterfeit": counterfeit,
        "conditions": conditions,
    }


def history_fields():
    fields = [
        "epoch", "train_loss", "train_authenticity_loss", "train_condition_loss",
        "train_authenticity_accuracy", "val_loss", "val_authenticity_loss",
        "val_condition_loss", "val_authenticity_accuracy", "val_counterfeit_f1",
    ]
    for name in CONDITION_NAMES:
        fields.extend(
            f"val_{name}_{metric}"
            for metric in ("known_count", "positive_count", "precision", "recall", "f1", "average_precision")
        )
    fields.extend(("epoch_seconds", "is_best"))
    return fields


def main():
    args = parse_args()
    if min(args.epochs, args.batch_size, args.image_size) < 1:
        raise ValueError("epochs, batch-size, and image-size must be positive")
    if args.learning_rate <= 0 or args.condition_loss_weight < 0 or args.max_pos_weight < 1:
        raise ValueError("learning-rate must be positive, condition weight nonnegative, and max-pos-weight at least 1")

    set_seed(args.seed)
    project_root = Path(__file__).resolve().parents[3]
    dataset_dir = (
        Path(args.dataset_dir)
        if args.dataset_dir is not None
        else find_dataset_dir(args.data_root)
    ).resolve()
    annotations_csv = Path(args.annotations_csv)
    if not annotations_csv.is_absolute():
        annotations_csv = project_root / annotations_csv
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not annotations_csv.is_file():
        raise FileNotFoundError(f"Condition annotation CSV not found: {annotations_csv}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"multitask-{timestamp}-seed{args.seed}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("run-id may contain only letters, numbers, dot, underscore, and hyphen")

    train_dataset, validation_dataset, train_loader, validation_loader = (
        create_multitask_multistream_loaders(
            dataset_dir,
            annotations_csv=annotations_csv,
            batch_size=args.batch_size,
            seed=args.seed,
            target_size=args.image_size,
            cache_dir=args.cache_dir,
        )
    )

    run_dir = Path(args.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_split_manifest(dataset_dir, run_dir / "split_manifest.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskMultiSignalCNN().to(device)
    pos_weight = make_positive_weights(train_dataset, args.max_pos_weight, device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    counts = train_dataset.condition_counts
    config = {
        "run_id": run_id,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "image_size": args.image_size,
        "architecture": "MultiTaskMultiSignalCNN",
        "shared_signals": ["rgb", "edges", "texture"],
        "output_heads": {
            "authenticity": ["counterfeit", "genuine"],
            "conditions": list(CONDITION_NAMES),
        },
        "condition_values": {"no": 0, "yes": 1, "uncertain": "masked"},
        "condition_loss_weight": args.condition_loss_weight,
        "condition_positive_weights": {
            name: float(pos_weight[index].item())
            for index, name in enumerate(CONDITION_NAMES)
        },
        "max_pos_weight": args.max_pos_weight,
        "training_condition_counts": counts,
        "train_images": len(train_dataset),
        "validation_images": len(validation_dataset),
        "dataset_folder_name": dataset_dir.name,
        "annotations_csv": str(annotations_csv.resolve()),
        "preprocessing_cache": str(args.cache_dir.resolve()) if args.cache_dir else None,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "optimizer": "Adam",
        "selection_rule": "lowest combined validation loss",
        "test_set_used_for_training_or_selection": False,
        "split_manifest": "split_manifest.csv",
    }
    atomic_json_write(run_dir / "config.json", config)

    summary = {
        "run_id": run_id,
        "status": "running",
        "last_completed_epoch": 0,
        "best_epoch": None,
        "best_val_loss": None,
        "latest_validation": None,
    }
    summary_path = run_dir / "summary.json"
    atomic_json_write(summary_path, summary)
    history_path = run_dir / "history.csv"
    fields = history_fields()
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()

    best_val_loss = float("inf")
    best_path = run_dir / "best_checkpoint.pth"
    latest_path = run_dir / "latest_checkpoint.pth"
    print(f"Run: {run_id}\nDataset: {dataset_dir}\nDevice: {device}\nArtifacts: {run_dir.resolve()}")

    try:
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            train = train_epoch(
                model, train_loader, optimizer, device, pos_weight,
                args.condition_loss_weight,
            )
            validation = validation_metrics(
                model, validation_loader, device, pos_weight,
                args.condition_loss_weight,
            )
            epoch_seconds = time.perf_counter() - started
            is_best = validation["loss"] < best_val_loss
            if is_best:
                best_val_loss = validation["loss"]

            row = {
                "epoch": epoch,
                "train_loss": train["loss"],
                "train_authenticity_loss": train["authenticity_loss"],
                "train_condition_loss": train["condition_loss"],
                "train_authenticity_accuracy": train["authenticity_accuracy"],
                "val_loss": validation["loss"],
                "val_authenticity_loss": validation["authenticity_loss"],
                "val_condition_loss": validation["condition_loss"],
                "val_authenticity_accuracy": validation["authenticity_accuracy"],
                "val_counterfeit_f1": validation["counterfeit"]["f1"],
                "epoch_seconds": epoch_seconds,
                "is_best": is_best,
            }
            for name in CONDITION_NAMES:
                for metric in ("known_count", "positive_count", "precision", "recall", "f1", "average_precision"):
                    row[f"val_{name}_{metric}"] = validation["conditions"][name][metric]
            with history_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writerow(row)
                handle.flush()
                os.fsync(handle.fileno())

            checkpoint = {
                "format_version": 1,
                "run_id": run_id,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation": validation,
                "config": config,
            }
            atomic_checkpoint_write(latest_path, checkpoint)
            if is_best:
                atomic_checkpoint_write(best_path, checkpoint)
                summary["best_epoch"] = epoch
                summary["best_val_loss"] = validation["loss"]
            summary["last_completed_epoch"] = epoch
            summary["latest_validation"] = validation
            atomic_json_write(summary_path, summary)
            tag_scores = ", ".join(
                f"{name} F1={validation['conditions'][name]['f1']:.3f} "
                f"(n={validation['conditions'][name]['known_count']})"
                for name in CONDITION_NAMES
            )
            print(
                f"Epoch {epoch:02d}/{args.epochs} | train loss {train['loss']:.4f} | "
                f"val loss {validation['loss']:.4f}, authenticity acc "
                f"{validation['authenticity_accuracy']:.3f} | {tag_scores}"
                f"{' | best checkpoint updated' if is_best else ''}"
            )

        summary["status"] = "completed"
        atomic_json_write(summary_path, summary)
        print(f"Training complete. Best checkpoint: {best_path.resolve()}")
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        atomic_json_write(summary_path, summary)
        print(f"Training interrupted; completed epochs are saved under {run_dir.resolve()}")
        raise
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        atomic_json_write(summary_path, summary)
        raise


if __name__ == "__main__":
    main()
