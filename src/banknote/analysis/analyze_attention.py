"""Summarize learned modality-attention weights on validation images only."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.banknote.ingest.multistream_data_loader import (
    MultiStreamBanknoteDataset,
    find_dataset_dir,
)
from src.banknote.models.research_models import build_research_model


SIGNALS = ("rgb", "edges", "texture")
LABELS = {0: "counterfeit", 1: "genuine"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--allow-rerun", action="store_true")
    return parser.parse_args()


def summarize(weights, labels):
    weights = np.asarray(weights)
    labels = np.asarray(labels)
    result = {
        "image_count": int(len(labels)),
        "signals": list(SIGNALS),
        "overall_mean": weights.mean(axis=0).tolist(),
        "overall_std": weights.std(axis=0).tolist(),
        "by_true_class_mean": {},
    }
    for label, name in LABELS.items():
        selected = weights[labels == label]
        if len(selected):
            result["by_true_class_mean"][name] = selected.mean(axis=0).tolist()
    return result


def main():
    args = parse_args()
    run_dir = args.run_dir
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("model") != "hybrid_attention":
        raise ValueError("This analysis requires a completed hybrid_attention run.")
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed":
        raise ValueError("Only completed runs can be analyzed.")
    checkpoint_path = run_dir / "best_checkpoint.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    output_json = run_dir / "validation_attention_summary.json"
    output_png = run_dir / "validation_attention_weights.png"
    if output_json.exists() and not args.allow_rerun:
        raise FileExistsError(
            f"{output_json} already exists. Use --allow-rerun for a repeat."
        )

    model = build_research_model("hybrid_attention", pretrained=False)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset_dir = (
        Path(args.dataset_dir) if args.dataset_dir is not None
        else find_dataset_dir(args.data_root)
    )
    validation_set = MultiStreamBanknoteDataset(
        dataset_dir,
        "validation_set",
        target_size=int(config.get("image_size", 224)),
        cache_dir=args.cache_dir,
    )
    loader = DataLoader(
        validation_set, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    all_weights = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            model(
                batch["rgb"].to(device),
                batch["edges"].to(device),
                batch["texture"].to(device),
            )
            all_weights.extend(model.last_attention_weights.cpu().tolist())
            all_labels.extend(batch["label"].tolist())

    result = {
        "run_id": config["run_id"],
        "data_split": "validation_set",
        "fusion": "learned scalar attention over RGB, edges, texture",
        **summarize(all_weights, all_labels),
    }
    output_json.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    groups = [np.asarray(all_weights)]
    names = ["All"]
    for label, name in LABELS.items():
        class_weights = np.asarray(all_weights)[np.asarray(all_labels) == label]
        if len(class_weights):
            groups.append(class_weights)
            names.append(name.capitalize())
    means = np.stack([group.mean(axis=0) for group in groups])
    positions = np.arange(len(SIGNALS))
    width = 0.8 / len(groups)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for index, (name, values) in enumerate(zip(names, means)):
        axis.bar(
            positions + index * width,
            values,
            width=width,
            label=name,
        )
    axis.set_xticks(positions + width * (len(groups) - 1) / 2)
    axis.set_xticklabels(SIGNALS)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Mean attention weight")
    axis.set_title("Validation modality attention")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_png, dpi=160)
    plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
