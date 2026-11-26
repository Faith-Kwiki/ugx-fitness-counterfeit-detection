
import torch

from src.banknote.ingest.data_loader import create_data_loaders


train_loader, validation_loader, test_loader = (
    create_data_loaders()
)

print("\nDATA LOADER TEST")
print("-" * 40)

print("Training batches:", len(train_loader))
print("Validation batches:", len(validation_loader))
print("Testing batches:", len(test_loader))

# Get one batch of training images
images, labels = next(iter(train_loader))

print("\nBATCH INFORMATION")
print("-" * 40)

print("Image tensor shape:", images.shape)
print("Label tensor shape:", labels.shape)

print("Image data type:", images.dtype)
print("Label data type:", labels.dtype)

print("Minimum pixel value:", images.min().item())
print("Maximum pixel value:", images.max().item())

print("\nCLASS DISTRIBUTION IN THIS BATCH")

print("Counterfeit:", (labels == 0).sum().item())
print("Genuine:", (labels == 1).sum().item())

print("\nData pipeline test complete!")
