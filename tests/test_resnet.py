
import torch

from src.banknote.models.resnet_baseline import BanknoteResNet18

def test_resnet18():

    print("\nRESNET-18 ARCHITECTURE TEST")
    print("-" * 40)

    model = BanknoteResNet18(
        num_classes=2,
        freeze_backbone=True
    )

    # Simulate 8 RGB images
    sample_images = torch.randn(8, 3, 224, 224)

    model.eval()

    with torch.no_grad():
        predictions = model(sample_images)

    print("Input shape:", sample_images.shape)
    print("Output shape:", predictions.shape)

    total_parameters = sum(
        p.numel() for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print("Total parameters:", total_parameters)
    print("Trainable parameters:", trainable_parameters)

    assert predictions.shape == (8, 2)

    print("\nResNet-18 test passed!")

if __name__ == "__main__":
    test_resnet18()