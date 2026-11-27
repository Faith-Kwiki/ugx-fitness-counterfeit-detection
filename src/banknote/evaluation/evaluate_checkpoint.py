"""Evaluate a completed multi-task checkpoint on validation or held-out test data.

Test evaluation requires --confirm-final-selection. Training and model selection
must be complete before running the test split.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.banknote.ingest.multistream_data_loader import (
    CONDITION_NAMES,
    MultiTaskMultiStreamBanknoteDataset,
    find_dataset_dir,
)
from src.banknote.models.multitask_multisignal_cnn import MultiTaskMultiSignalCNN
from src.banknote.training.train_multitask_multisignal import binary_metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--annotations-csv",
        type=Path,
        default=Path("annotations/condition_labels_template.csv"),
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--split", choices=("validation_set", "test_set"), default="test_set"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--confirm-final-selection", action="store_true")
    parser.add_argument("--allow-rerun", action="store_true")
    return parser.parse_args()


def binary_authenticity_metrics(targets, scores):
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
    specificity = tn / (tn + fp) if tn + fp else 0.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else None

    auc = None
    positive_count = int(positives.sum())
    negative_count = int(negatives.sum())
    if positive_count and negative_count:
        order = np.argsort(scores, kind="stable")
        sorted_scores = scores[order]
        ranks = np.empty(len(scores), dtype=np.float64)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and sorted_scores[end] == sorted_scores[start]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2.0
            start = end
        rank_sum = ranks[positives].sum()
        auc = float(
            (rank_sum - positive_count * (positive_count + 1) / 2)
            / (positive_count * negative_count)
        )

    return {
        "known_count": int(len(targets)),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "threshold": 0.5,
        "accuracy": float(np.mean(predictions == positives)) if len(targets) else None,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(2 * precision * recall / (precision + recall))
        if precision + recall
        else 0.0,
        "balanced_accuracy": float((recall + specificity) / 2),
        "matthews_correlation_coefficient": float(mcc) if mcc is not None else None,
        "roc_auc": auc,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def write_confusion_matrix(path, matrix):
    image = np.full((420, 520, 3), 255, dtype=np.uint8)
    cv2.putText(image, "Counterfeit authenticity", (105, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (35, 45, 55), 2)
    cv2.putText(image, "Predicted", (218, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (35, 45, 55), 1)
    cv2.putText(image, "Actual", (28, 242), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (35, 45, 55), 1)
    labels = [["Genuine", "Counterfeit"], ["Genuine", "Counterfeit"]]
    values = [[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]]
    x0, y0, cell = 150, 110, 150
    for row in range(2):
        for col in range(2):
            x, y = x0 + col * cell, y0 + row * cell
            color = (228, 240, 231) if row == col else (240, 231, 225)
            cv2.rectangle(image, (x, y), (x + cell, y + cell), color, -1)
            cv2.rectangle(image, (x, y), (x + cell, y + cell), (90, 100, 110), 1)
            cv2.putText(image, str(values[row][col]), (x + 56, y + 82), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (25, 35, 45), 2)
            cv2.putText(image, labels[row][col], (x + 35, y + 118), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (45, 55, 65), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write confusion matrix image: {path}")


def main():
    args = parse_args()
    if args.split == "test_set" and not args.confirm_final_selection:
        raise ValueError(
            "Test evaluation requires --confirm-final-selection after choosing "
            "a completed checkpoint from validation results."
        )
    if args.split == "test_set" and args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    project_root = Path(__file__).resolve().parents[3]
    run_dir = args.run_dir if args.run_dir.is_absolute() else project_root / args.run_dir
    run_dir = run_dir.resolve()
    summary_path = run_dir / "summary.json"
    checkpoint_path = run_dir / "best_checkpoint.pth"
    if not summary_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Completed run files are missing in {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed":
        raise ValueError(f"Run is not completed: status={summary.get('status')!r}")
    if not summary.get("best_epoch"):
        raise ValueError("Run summary has no selected best epoch.")

    report_path = run_dir / (
        "test_evaluation.json" if args.split == "test_set" else "validation_reassessment.json"
    )
    if report_path.exists() and not args.allow_rerun:
        raise FileExistsError(
            f"Refusing to overwrite {report_path}; use --allow-rerun deliberately."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    dataset_dir = (
        args.dataset_dir
        if args.dataset_dir is not None
        else find_dataset_dir(args.data_root)
    ).resolve()
    annotations_csv = args.annotations_csv
    if not annotations_csv.is_absolute():
        annotations_csv = project_root / annotations_csv
    cache_dir = args.cache_dir
    if cache_dir is not None and not cache_dir.is_absolute():
        cache_dir = project_root / cache_dir

    dataset = MultiTaskMultiStreamBanknoteDataset(
        dataset_dir,
        args.split,
        annotations_csv=annotations_csv,
        target_size=int(config.get("image_size", 224)),
        cache_dir=cache_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    model = MultiTaskMultiSignalCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    authenticity_targets = []
    counterfeit_scores = []
    condition_targets = [[] for _ in CONDITION_NAMES]
    condition_scores = [[] for _ in CONDITION_NAMES]
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            edges = batch["edges"].to(device)
            texture = batch["texture"].to(device)
            labels = batch["label"].to(device)
            targets = batch["condition_targets"].to(device)
            masks = batch["condition_mask"].to(device).bool()
            outputs = model(rgb, edges, texture)
            probabilities = outputs["authenticity_logits"].softmax(dim=1)[:, 0]
            authenticity_targets.extend((labels == 0).int().cpu().tolist())
            counterfeit_scores.extend(probabilities.cpu().tolist())
            condition_probabilities = outputs["condition_logits"].sigmoid().cpu().numpy()
            target_values = targets.cpu().numpy()
            known_values = masks.cpu().numpy()
            for column in range(len(CONDITION_NAMES)):
                condition_targets[column].extend(
                    target_values[known_values[:, column], column].astype(int).tolist()
                )
                condition_scores[column].extend(
                    condition_probabilities[known_values[:, column], column].tolist()
                )

    authenticity = binary_authenticity_metrics(authenticity_targets, counterfeit_scores)
    if args.split == "test_set" and (
        authenticity["positive_count"] == 0 or authenticity["negative_count"] == 0
    ):
        raise ValueError("The held-out test split must contain both authenticity classes.")
    conditions = {
        name: binary_metrics(condition_targets[index], condition_scores[index])
        for index, name in enumerate(CONDITION_NAMES)
    }
    try:
        checkpoint_reference = checkpoint_path.relative_to(project_root).as_posix()
    except ValueError:
        checkpoint_reference = str(checkpoint_path)
    result = {
        "run_id": config.get("run_id", run_dir.name),
        "checkpoint": checkpoint_reference,
        "selected_epoch": checkpoint.get("epoch", summary["best_epoch"]),
        "selection_rule": config.get("selection_rule", "best validation checkpoint"),
        "split": args.split,
        "test_set_used_for_training_or_selection": False,
        "image_count": len(dataset),
        "device": str(device),
        "threshold": 0.5,
        "authenticity_counterfeit_positive": authenticity,
        "condition_labels_are_human_spot_reviewed": False,
        "condition_metrics_note": (
            "Test condition labels have not received a human spot-review."
            if args.split == "test_set"
            else "Only the 39-image spot-review sample has human-reviewed labels; "
            "other validation condition labels come from the earlier annotation pass."
        ),
        "conditions": conditions,
    }
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.split == "test_set":
        write_confusion_matrix(run_dir / "test_confusion_matrix.png", authenticity["confusion_matrix"])
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved evaluation report: {report_path}")


if __name__ == "__main__":
    main()
