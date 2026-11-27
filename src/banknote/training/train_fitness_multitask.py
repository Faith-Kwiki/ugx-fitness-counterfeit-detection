"""Train and evaluate the preliminary joint fitness outcome model.

Fitness labels are low-confidence prelabels derived from the existing image
condition annotations. Treat every metric as agreement with that proxy rule,
not as a Bank of Uganda decision or validated real-world performance.
"""

import argparse
import csv
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.banknote.ingest.multistream_data_loader import (
    CONDITION_NAMES,
    MultiTaskMultiStreamBanknoteDataset,
    find_dataset_dir,
    write_split_manifest,
)
from src.banknote.models.fitness_multitask_multisignal_cnn import (
    FITNESS_OUTCOMES,
    FITNESS_STATUSES,
    SEVERITY_NAMES,
    FitnessMultiTaskMultiSignalCNN,
    joint_outcome_predictions,
    joint_outcome_probabilities,
)


OUTCOME_TO_INDEX = {name: index for index, name in enumerate(FITNESS_OUTCOMES)}
FITNESS_STATUS_TO_INDEX = {"genuine_fit": 0, "genuine_unfit": 1}
SEVERITY_TO_INDEX = {name: index for index, name in enumerate(SEVERITY_NAMES)}
OUTCOME_DISPLAY = {
    "counterfeit": "counterfeit",
    "genuine_fit": "genuine and fit for circulation",
    "genuine_unfit": "genuine and unfit for circulation",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--annotations-csv",
        type=Path,
        default=Path("annotations/prototype_fitness_prelabels.csv"),
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/fitness"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--condition-loss-weight", type=float, default=0.25)
    parser.add_argument("--severity-loss-weight", type=float, default=0.25)
    parser.add_argument("--authenticity-loss-weight", type=float, default=0.25)
    parser.add_argument("--fitness-loss-weight", type=float, default=0.25)
    parser.add_argument("--max-class-weight", type=float, default=10.0)
    return parser.parse_args()


class FitnessOutcomeDataset(Dataset):
    """Existing aligned streams paired with the usable prototype prelabels."""

    def __init__(self, dataset_dir, split, annotations_csv, target_size, cache_dir):
        self.base = MultiTaskMultiStreamBanknoteDataset(
            dataset_dir,
            split,
            annotations_csv=annotations_csv,
            target_size=target_size,
            cache_dir=cache_dir,
        )
        with Path(annotations_csv).open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"fitness_outcome", *(f"{name}_severity" for name in CONDITION_NAMES)}
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"Prelabel CSV is missing columns: {', '.join(sorted(missing))}")
            rows = {row["image_id"].strip(): row for row in reader}

        usable = []
        self.labels = {}
        for sample in self.base.samples:
            image_path = sample[0]
            image_id = image_path.stem
            row = rows.get(image_id)
            if row is None:
                raise ValueError(f"No prototype fitness label for {image_id}")
            if row["split"].strip() != split:
                raise ValueError(f"Split mismatch for {image_id}")
            outcome = row["fitness_outcome"].strip()
            if outcome == "review_required":
                continue
            if outcome not in OUTCOME_TO_INDEX:
                raise ValueError(f"Invalid fitness outcome {outcome!r} for {image_id}")
            if row["prelabel_confidence"].strip() != "low":
                raise ValueError(f"Expected a low-confidence prelabel for {image_id}")

            severity_values = []
            severity_masks = []
            for name in CONDITION_NAMES:
                value = row[f"{name}_severity"].strip().lower()
                if value in SEVERITY_TO_INDEX:
                    severity_values.append(SEVERITY_TO_INDEX[value])
                    severity_masks.append(1.0)
                elif value == "uncertain":
                    severity_values.append(0)
                    severity_masks.append(0.0)
                else:
                    raise ValueError(f"Invalid {name} severity {value!r} for {image_id}")
            self.labels[image_id] = {
                "outcome": OUTCOME_TO_INDEX[outcome],
                "outcome_name": outcome,
                "fitness": FITNESS_STATUS_TO_INDEX.get(outcome, -1),
                "severity": severity_values,
                "severity_mask": severity_masks,
                "authenticity": sample[1],
            }
            usable.append(sample)
        self.base.samples = usable
        if not self.base.samples:
            raise ValueError(f"No usable fitness prelabels in {split}")

        self.outcome_counts = {name: 0 for name in FITNESS_OUTCOMES}
        self.fitness_counts = {name: 0 for name in FITNESS_STATUSES}
        self.severity_counts = {name: 0 for name in SEVERITY_NAMES}
        self.condition_counts = {name: {"yes": 0, "no": 0} for name in CONDITION_NAMES}
        for image_path, _, _ in self.base.samples:
            image_id = image_path.stem
            label = self.labels[image_id]
            self.outcome_counts[label["outcome_name"]] += 1
            if label["outcome_name"] in FITNESS_STATUS_TO_INDEX:
                self.fitness_counts[FITNESS_STATUSES[label["fitness"]]] += 1
            for index, name in enumerate(CONDITION_NAMES):
                self.condition_counts[name]["yes" if self.base.condition_targets[image_id][index] else "no"] += 1
                if label["severity_mask"][index]:
                    severity_name = SEVERITY_NAMES[label["severity"][index]]
                    self.severity_counts[severity_name] += 1

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        image_id = self.base.samples[index][0].stem
        label = self.labels[image_id]
        sample.update(
            {
                "fitness_target": torch.tensor(label["outcome"], dtype=torch.long),
                "status_target": torch.tensor(label["fitness"], dtype=torch.long),
                "severity_targets": torch.tensor(label["severity"], dtype=torch.long),
                "severity_mask": torch.tensor(label["severity_mask"], dtype=torch.float32),
            }
        )
        return sample


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_weights(counts, class_names, maximum, device):
    values = np.array([counts.get(name, 0) for name in class_names], dtype=np.float64)
    nonzero = values > 0
    weights = np.ones_like(values)
    if nonzero.any():
        weights[nonzero] = values.sum() / (len(values) * values[nonzero])
    return torch.tensor(np.minimum(weights, maximum), dtype=torch.float32, device=device)


def condition_pos_weights(dataset, maximum, device):
    weights = []
    for name in CONDITION_NAMES:
        yes = dataset.condition_counts[name]["yes"]
        no = dataset.condition_counts[name]["no"]
        weights.append(min(no / yes, maximum) if yes else 1.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_loss(outputs, batch, device, outcome_weights, fitness_weights, severity_weights, condition_weights, args):
    outcome_targets = batch["fitness_target"].to(device)
    joint_probs = joint_outcome_probabilities(outputs["authenticity_logits"], outputs["fitness_logits"])
    outcome_loss = F.nll_loss(joint_probs.clamp_min(1e-8).log(), outcome_targets, weight=outcome_weights)
    authenticity_loss = F.cross_entropy(outputs["authenticity_logits"], batch["label"].to(device))
    genuine_mask = batch["label"].to(device) == 1
    status_targets = batch["status_target"].to(device)
    if genuine_mask.any():
        fitness_loss = F.cross_entropy(
            outputs["fitness_logits"][genuine_mask],
            status_targets[genuine_mask],
            weight=fitness_weights,
        )
    else:
        fitness_loss = outputs["fitness_logits"].sum() * 0.0

    condition_targets = batch["condition_targets"].to(device)
    condition_mask = batch["condition_mask"].to(device)
    condition_elements = F.binary_cross_entropy_with_logits(
        outputs["condition_logits"],
        condition_targets,
        pos_weight=condition_weights,
        reduction="none",
    )
    condition_known = condition_mask.sum(dim=0)
    condition_per_tag = (condition_elements * condition_mask).sum(dim=0) / condition_known.clamp_min(1.0)
    condition_has_labels = (condition_known > 0).to(condition_per_tag.dtype)
    condition_loss = (condition_per_tag * condition_has_labels).sum() / condition_has_labels.sum().clamp_min(1.0)

    severity_targets = batch["severity_targets"].to(device)
    severity_mask = batch["severity_mask"].to(device)
    severity_losses = []
    for index in range(len(CONDITION_NAMES)):
        element_loss = F.cross_entropy(
            outputs["severity_logits"][:, index, :],
            severity_targets[:, index],
            weight=severity_weights,
            reduction="none",
        )
        mask = severity_mask[:, index]
        severity_losses.append((element_loss * mask).sum() / mask.sum().clamp_min(1.0))
    severity_loss = torch.stack(severity_losses).mean()

    total = (
        outcome_loss
        + args.authenticity_loss_weight * authenticity_loss
        + args.fitness_loss_weight * fitness_loss
        + args.condition_loss_weight * condition_loss
        + args.severity_loss_weight * severity_loss
    )
    return total, {
        "outcome_loss": outcome_loss,
        "authenticity_loss": authenticity_loss,
        "fitness_loss": fitness_loss,
        "condition_loss": condition_loss,
        "severity_loss": severity_loss,
    }


def classification_metrics(targets, predictions, class_names):
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[target, prediction] += 1
    per_class = {}
    f1s = []
    recalls = []
    for index, name in enumerate(class_names):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "support": int(matrix[index, :].sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if matrix[index, :].sum():
            f1s.append(f1)
            recalls.append(recall)
    return {
        "count": int(len(targets)),
        "accuracy": float(np.mean(targets == predictions)) if len(targets) else None,
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "classes": per_class,
        "confusion_matrix": matrix.tolist(),
        "class_order": list(class_names),
    }


def run_epoch(model, loader, device, optimizer, outcome_weights, fitness_weights, severity_weights, condition_weights, args):
    training = optimizer is not None
    model.train(training)
    totals = {name: 0.0 for name in ("loss", "outcome_loss", "authenticity_loss", "fitness_loss", "condition_loss", "severity_loss")}
    count = 0
    outcome_true, outcome_pred = [], []
    severity_true = [[] for _ in CONDITION_NAMES]
    severity_pred = [[] for _ in CONDITION_NAMES]

    for batch in loader:
        rgb = batch["rgb"].to(device)
        edges = batch["edges"].to(device)
        texture = batch["texture"].to(device)
        with torch.set_grad_enabled(training):
            outputs = model(rgb, edges, texture)
            loss, components = compute_loss(
                outputs, batch, device, outcome_weights, fitness_weights, severity_weights, condition_weights, args
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = len(batch["fitness_target"])
        count += batch_size
        totals["loss"] += loss.item() * batch_size
        for name, value in components.items():
            totals[name] += value.item() * batch_size
        outcome_true.extend(batch["fitness_target"].numpy().tolist())
        outcome_pred.extend(
            joint_outcome_predictions(outputs["authenticity_logits"], outputs["fitness_logits"])
            .cpu().numpy().tolist()
        )
        severity_logits = outputs["severity_logits"].argmax(dim=2).cpu().numpy()
        severity_targets = batch["severity_targets"].numpy()
        severity_mask = batch["severity_mask"].numpy().astype(bool)
        for index in range(len(CONDITION_NAMES)):
            severity_true[index].extend(severity_targets[severity_mask[:, index], index].tolist())
            severity_pred[index].extend(severity_logits[severity_mask[:, index], index].tolist())

    return (
        {name: value / max(count, 1) for name, value in totals.items()},
        classification_metrics(outcome_true, outcome_pred, FITNESS_OUTCOMES),
        {
            name: classification_metrics(severity_true[index], severity_pred[index], SEVERITY_NAMES)
            for index, name in enumerate(CONDITION_NAMES)
        },
    )


def atomic_json(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def write_predictions(path, targets, predictions, probabilities, image_ids):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        fields = ["image_id", "true_outcome", "predicted_outcome", "predicted_display"]
        fields += [f"probability_{name}" for name in FITNESS_OUTCOMES]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for image_id, target, prediction, probs in zip(image_ids, targets, predictions, probabilities):
            writer.writerow({
                "image_id": image_id,
                "true_outcome": FITNESS_OUTCOMES[target],
                "predicted_outcome": FITNESS_OUTCOMES[prediction],
                "predicted_display": OUTCOME_DISPLAY[FITNESS_OUTCOMES[prediction]],
                **{f"probability_{name}": float(probs[index]) for index, name in enumerate(FITNESS_OUTCOMES)},
            })


def evaluate_test(model, dataset, args, device, outcome_weights, fitness_weights, severity_weights, condition_weights, run_dir):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_losses, test_outcomes, test_severity = run_epoch(
        model, loader, device, None, outcome_weights, fitness_weights, severity_weights, condition_weights, args
    )
    model.eval()
    targets, predictions, probabilities, image_ids = [], [], [], []
    with torch.no_grad():
        for batch_number, batch in enumerate(loader):
            outputs = model(batch["rgb"].to(device), batch["edges"].to(device), batch["texture"].to(device))
            probs = joint_outcome_probabilities(outputs["authenticity_logits"], outputs["fitness_logits"]).cpu().numpy()
            preds = joint_outcome_predictions(outputs["authenticity_logits"], outputs["fitness_logits"]).cpu().numpy()
            start = batch_number * args.batch_size
            ids = [dataset.base.samples[index][0].stem for index in range(start, start + len(preds))]
            targets.extend(batch["fitness_target"].numpy().tolist())
            predictions.extend(preds.tolist())
            probabilities.extend(probs.tolist())
            image_ids.extend(ids)
    metrics = classification_metrics(targets, predictions, FITNESS_OUTCOMES)
    metrics["severity_by_condition"] = test_severity
    metrics["test_losses"] = test_losses
    metrics["label_provenance"] = "low-confidence rule-based prelabels; not independent ground truth"
    metrics["review_required_excluded"] = True
    metrics["selected_using_test_set"] = False
    write_predictions(run_dir / "test_predictions.csv", targets, predictions, probabilities, image_ids)
    atomic_json(run_dir / "test_evaluation.json", metrics)
    return metrics


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 1:
        raise ValueError("epochs, batch-size, and image-size must be positive")
    if args.learning_rate <= 0 or args.max_class_weight < 1:
        raise ValueError("learning-rate must be positive and max-class-weight must be at least 1")
    set_seed(args.seed)

    project_root = Path(__file__).resolve().parents[3]
    dataset_dir = (Path(args.dataset_dir) if args.dataset_dir else find_dataset_dir(args.data_root)).resolve()
    annotations = Path(args.annotations_csv)
    if not annotations.is_absolute():
        annotations = project_root / annotations
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")
    if not annotations.is_file():
        raise FileNotFoundError(f"Prelabel CSV not found: {annotations}")
    run_id = args.run_id or f"fitness-multitask-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-seed{args.seed}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("run-id may contain only letters, numbers, dot, underscore, and hyphen")
    run_dir = Path(args.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_split_manifest(dataset_dir, run_dir / "split_manifest.csv")

    train_data = FitnessOutcomeDataset(dataset_dir, "train_set", annotations, args.image_size, args.cache_dir)
    val_data = FitnessOutcomeDataset(dataset_dir, "validation_set", annotations, args.image_size, args.cache_dir)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FitnessMultiTaskMultiSignalCNN().to(device)
    outcome_weights = class_weights(train_data.outcome_counts, FITNESS_OUTCOMES, args.max_class_weight, device)
    fitness_weights = class_weights(train_data.fitness_counts, FITNESS_STATUSES, args.max_class_weight, device)
    severity_weights = class_weights(train_data.severity_counts, SEVERITY_NAMES, args.max_class_weight, device)
    condition_weights = condition_pos_weights(train_data, args.max_class_weight, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    config = {
        "run_id": run_id,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "image_size": args.image_size,
        "architecture": "FitnessMultiTaskMultiSignalCNN",
        "shared_signals": ["rgb", "edges", "texture"],
        "output_heads": {
            "authenticity": ["counterfeit", "genuine"],
            "genuine_fitness": list(FITNESS_STATUSES),
            "joint_outcome_policy": "counterfeit if authenticity head predicts counterfeit; otherwise use the genuine fitness head",
            "conditions": list(CONDITION_NAMES),
            "condition_severity": {name: list(SEVERITY_NAMES) for name in CONDITION_NAMES},
        },
        "fitness_loss_class_weights": {
            name: float(fitness_weights[index].item()) for index, name in enumerate(FITNESS_STATUSES)
        },
        "outcome_loss_class_weights": {
            name: float(outcome_weights[index].item()) for index, name in enumerate(FITNESS_OUTCOMES)
        },
        "training_outcome_counts": train_data.outcome_counts,
        "training_genuine_fitness_counts": train_data.fitness_counts,
        "validation_outcome_counts": val_data.outcome_counts,
        "condition_loss_weight": args.condition_loss_weight,
        "severity_loss_weight": args.severity_loss_weight,
        "authenticity_loss_weight": args.authenticity_loss_weight,
        "fitness_loss_weight": args.fitness_loss_weight,
        "label_provenance": "low-confidence rule-based prelabels derived from existing condition tags and notes",
        "review_required_rows_excluded": True,
        "selection_rule": "lowest weighted validation joint-outcome loss; no test-set selection",
        "test_set_used_for_training_or_selection": False,
        "train_images": len(train_data),
        "validation_images": len(val_data),
        "dataset_dir": str(dataset_dir),
        "annotations_csv": str(annotations.resolve()),
        "cache_dir": str(args.cache_dir.resolve()) if args.cache_dir else None,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    atomic_json(run_dir / "config.json", config)
    history_path = run_dir / "history.csv"
    history_fields = [
        "epoch",
        "train_loss",
        "train_outcome_loss",
        "train_authenticity_loss",
        "train_fitness_loss",
        "train_condition_loss",
        "train_severity_loss",
        "val_loss",
        "val_outcome_loss",
        "val_authenticity_loss",
        "val_fitness_loss",
        "val_condition_loss",
        "val_severity_loss",
        "val_accuracy",
        "val_macro_f1",
        "val_balanced_accuracy",
        "epoch_seconds",
        "is_best",
    ]
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=history_fields).writeheader()

    best_loss = float("inf")
    best_path = run_dir / "best_checkpoint.pth"
    summary = {"status": "running", "run_id": run_id, "best_epoch": None, "best_validation_outcome_loss": None}
    print(f"Run {run_id}; device={device}; train={len(train_data)}; validation={len(val_data)}")
    print(f"Training outcomes: {train_data.outcome_counts}")

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_losses, _, _ = run_epoch(model, train_loader, device, optimizer, outcome_weights, fitness_weights, severity_weights, condition_weights, args)
        val_losses, val_metrics, val_severity = run_epoch(model, val_loader, device, None, outcome_weights, fitness_weights, severity_weights, condition_weights, args)
        is_best = val_losses["outcome_loss"] < best_loss
        if is_best:
            best_loss = val_losses["outcome_loss"]
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "config": config}, best_path)
            summary["best_epoch"] = epoch
            summary["best_validation_outcome_loss"] = best_loss
            summary["best_validation_outcome_metrics"] = val_metrics
            summary["best_validation_severity_metrics"] = val_severity
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_losses.items()},
            **{f"val_{key}": value for key, value in val_losses.items()},
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "epoch_seconds": time.perf_counter() - started,
            "is_best": is_best,
        }
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=history_fields).writerow(row)
        summary.update({"status": "running", "last_completed_epoch": epoch})
        atomic_json(run_dir / "summary.json", summary)
        print(
            f"Epoch {epoch:02d}/{args.epochs}: val outcome loss={val_losses['outcome_loss']:.4f}, "
            f"accuracy={val_metrics['accuracy']:.3f}, macro-F1={val_metrics['macro_f1']:.3f}, "
            f"balanced-accuracy={val_metrics['balanced_accuracy']:.3f}"
            + (" [best]" if is_best else "")
        )

    summary["status"] = "completed"
    atomic_json(run_dir / "summary.json", summary)

    # Load the validation-selected checkpoint before opening the held-out test set.
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_data = FitnessOutcomeDataset(dataset_dir, "test_set", annotations, args.image_size, args.cache_dir)
    test_metrics = evaluate_test(model, test_data, args, device, outcome_weights, fitness_weights, severity_weights, condition_weights, run_dir)
    summary["test_outcome_metrics"] = test_metrics
    summary["test_images"] = len(test_data)
    atomic_json(run_dir / "summary.json", summary)
    print(f"Test proxy metrics (held out from training/selection): {test_metrics}")
    print(f"Artifacts: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
