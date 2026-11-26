
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.banknote.models.baseline_cnn import BanknoteCNN
from src.banknote.ingest.data_loader import create_data_loaders


# -------------------------------
# 1. CONFIGURATION
# -------------------------------

EPOCHS = 10
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "baseline_cnn_best.pth"


# -------------------------------
# 2. TRAINING FUNCTION
# -------------------------------

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_images = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear gradients from the previous batch
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(images)

        # Calculate classification loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Track performance
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = torch.argmax(outputs, dim=1)

        correct_predictions += (predictions == labels).sum().item()
        total_images += batch_size

    average_loss = total_loss / total_images
    accuracy = correct_predictions / total_images

    return average_loss, accuracy


# -------------------------------
# 3. VALIDATION FUNCTION
# -------------------------------

def validate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_images = 0

    # No gradients are needed during validation
    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = torch.argmax(outputs, dim=1)

            correct_predictions += (predictions == labels).sum().item()
            total_images += batch_size

    average_loss = total_loss / total_images
    accuracy = correct_predictions / total_images

    return average_loss, accuracy


# -------------------------------
# 4. MAIN TRAINING PROCESS
# -------------------------------

def main():

    print("\nBASELINE CNN TRAINING")
    print("=" * 45)

    print("Device:", DEVICE)
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)

    # Load training, validation and test data
    train_loader, validation_loader, test_loader = (
        create_data_loaders()
    )

    # Create the CNN
    model = BanknoteCNN().to(DEVICE)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimiser
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_validation_loss = float("inf")

    # Training loop
    for epoch in range(EPOCHS):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        validation_loss, validation_accuracy = validate(
            model,
            validation_loader,
            criterion
        )

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

        print(
            f"Training Loss: {train_loss:.4f} | "
            f"Training Accuracy: {train_accuracy * 100:.2f}%"
        )

        print(
            f"Validation Loss: {validation_loss:.4f} | "
            f"Validation Accuracy: {validation_accuracy * 100:.2f}%"
        )

        # Save the model with the lowest validation loss
        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            torch.save(
                model.state_dict(),
                BEST_MODEL_PATH
            )

            print("Best model checkpoint saved!")

    print("\nTRAINING COMPLETE")
    print("=" * 45)

    print("Best validation loss:", round(best_validation_loss, 4))
    print("Best model saved at:", BEST_MODEL_PATH)


if __name__ == "__main__":
    main()
