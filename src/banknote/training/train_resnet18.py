
import os
import csv
import time
import random
import numpy as np
import torch
import torch.nn as nn

from torch.optim import Adam
from src.banknote.models.resnet_baseline import BanknoteResNet18
from src.banknote.ingest.data_loader import create_data_loaders


# -----------------------------------------
# 1. EXPERIMENT CONFIGURATION
# -----------------------------------------

EPOCHS = 10
LEARNING_RATE = 0.001
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "models/resnet18_feature_extractor_best.pth"
HISTORY_PATH = "reports/resnet18_feature_extractor_history.csv"


# ImageNet normalisation values
IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(1, 3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(1, 3, 1, 1)


# -----------------------------------------
# 2. REPRODUCIBILITY
# -----------------------------------------

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------
# 3. IMAGE NORMALISATION
# -----------------------------------------

def normalise_images(images):

    mean = IMAGENET_MEAN.to(images.device)
    std = IMAGENET_STD.to(images.device)

    return (images - mean) / std


# -----------------------------------------
# 4. TRAINING FUNCTION
# -----------------------------------------

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

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

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# -----------------------------------------
# 5. VALIDATION FUNCTION
# -----------------------------------------

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

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# -----------------------------------------
# 6. MAIN TRAINING PROCESS
# -----------------------------------------

def main():

    set_seed(SEED)

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("\nRESNET-18 FEATURE EXTRACTION")
    print("=" * 50)

    print("Device:", DEVICE)
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Random seed:", SEED)

    # Load datasets
    train_loader, val_loader, test_loader = (
        create_data_loaders()
    )

    print("\nDataset loaded successfully.")
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    # Initialise pretrained ResNet-18
    model = BanknoteResNet18(
        num_classes=2,
        freeze_backbone=True
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # Only train parameters that are not frozen
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")

    history = []

    start_time = time.time()

    # Training loop
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

        # Save the best model using validation loss
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
                MODEL_PATH
            )

            print("Best checkpoint saved!")

    total_time = time.time() - start_time

    # Save experiment history
    with open(HISTORY_PATH, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys()
        )

        writer.writeheader()
        writer.writerows(history)

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")

    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total training time: {total_time:.2f} seconds")

    print("Model saved to:", MODEL_PATH)
    print("History saved to:", HISTORY_PATH)


if __name__ == "__main__":
    main()
