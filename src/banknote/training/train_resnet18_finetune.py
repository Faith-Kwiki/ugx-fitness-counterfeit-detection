
import os
import csv
import time
import random
import numpy as np
import torch
import torch.nn as nn

from torch.optim import AdamW

from src.banknote.models.resnet_baseline import BanknoteResNet18
from src.banknote.ingest.data_loader import create_data_loaders


# ==========================================
# 1. CONFIGURATION
# ==========================================

EPOCHS = 10
SEED = 42

BACKBONE_LR = 0.00001
CLASSIFIER_LR = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

FEATURE_CHECKPOINT = (
    "models/resnet18_feature_extractor_best.pth"
)

FINETUNE_CHECKPOINT = (
    "models/resnet18_finetuned_best.pth"
)

HISTORY_PATH = (
    "reports/resnet18_finetuning_history.csv"
)

MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(1, 3, 1, 1)

STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(1, 3, 1, 1)


# ==========================================
# 2. REPRODUCIBILITY
# ==========================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# 3. NORMALISATION
# ==========================================

def normalise_images(images):

    mean = MEAN.to(images.device)
    std = STD.to(images.device)

    return (images - mean) / std


# ==========================================
# 4. TRAINABLE LAYER CONFIGURATION
# ==========================================

def configure_finetuning(model):

    # Freeze every parameter first
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Unfreeze deeper ResNet layers
    for name, parameter in model.model.named_parameters():

        if name.startswith(("layer3.", "layer4.", "fc.")):
            parameter.requires_grad = True

    return model


# ==========================================
# 5. TRAINING FUNCTION
# ==========================================

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    # Keep frozen BatchNorm layers in evaluation mode
    model.model.bn1.eval()
    model.model.layer1.eval()
    model.model.layer2.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        images = normalise_images(images)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


# ==========================================
# 6. VALIDATION FUNCTION
# ==========================================

def validate(model, loader, criterion):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            images = normalise_images(images)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


# ==========================================
# 7. MAIN EXPERIMENT
# ==========================================

def main():

    set_seed(SEED)

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("\nRESNET-18 FINE-TUNING")
    print("=" * 50)

    print("Device:", DEVICE)
    print("Epochs:", EPOCHS)
    print("Backbone learning rate:", BACKBONE_LR)
    print("Classifier learning rate:", CLASSIFIER_LR)

    # Load datasets
    train_loader, val_loader, test_loader = (
        create_data_loaders()
    )

    # Initialise model
    model = BanknoteResNet18(
        num_classes=2,
        freeze_backbone=True
    )

    # Load the best feature-extraction checkpoint
    checkpoint = torch.load(
        FEATURE_CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("\nLoaded feature-extraction checkpoint.")
    print("Starting checkpoint epoch:", checkpoint["epoch"])

    # Unfreeze selected deeper layers
    model = configure_finetuning(model)

    model = model.to(DEVICE)

    # Verify trainable parameters
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    print(
        "\nTrainable parameters:",
        sum(p.numel() for p in trainable_parameters)
    )

    # Separate learning rates
    optimizer = AdamW(
        [
            {
                "params": model.model.layer3.parameters(),
                "lr": BACKBONE_LR
            },
            {
                "params": model.model.layer4.parameters(),
                "lr": BACKBONE_LR
            },
            {
                "params": model.model.fc.parameters(),
                "lr": CLASSIFIER_LR
            }
        ],
        weight_decay=0.01
    )

    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    history = []

    start_time = time.time()

    for epoch in range(EPOCHS):

        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion
        )

        epoch_time = time.time() - epoch_start

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_acc * 100:.2f}%")

        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc * 100:.2f}%")

        print(f"Epoch Time: {epoch_time:.2f} seconds")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "epoch_time_seconds": epoch_time
        })

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": val_loss,
                    "validation_accuracy": val_acc,
                    "seed": SEED
                },
                FINETUNE_CHECKPOINT
            )

            print("Best fine-tuned checkpoint saved!")

    total_time = time.time() - start_time

    with open(HISTORY_PATH, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys()
        )

        writer.writeheader()
        writer.writerows(history)

    print("\n" + "=" * 50)
    print("FINE-TUNING COMPLETED")

    print("Best validation loss:", best_val_loss)
    print("Total training time:", total_time)

    print("Checkpoint:", FINETUNE_CHECKPOINT)
    print("History:", HISTORY_PATH)


if __name__ == "__main__":
    main()
