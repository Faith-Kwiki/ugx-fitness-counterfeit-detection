
import torch

from src.banknote.models.baseline_cnn import BanknoteCNN


def test_baseline_cnn():

    print("\nBASELINE CNN TEST")
    print("-" * 40)

    # Create the model
    model = BanknoteCNN()


    # Count all model parameters
    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )

    # Count parameters that will be learned during training
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Total parameters:", total_parameters)
    print("Trainable parameters:", trainable_parameters)

    # Simulate a batch of 8 RGB banknote images
    sample_images = torch.randn(8, 3, 224, 224)

    # Pass images through the model
    predictions = model(sample_images)

    print("Input shape:", sample_images.shape)
    print("Output shape:", predictions.shape)
    print("Output data type:", predictions.dtype)

    # Confirm the expected output
    assert predictions.shape == (8, 2)

    print("\nBaseline CNN architecture test passed!")


if __name__ == "__main__":
    test_baseline_cnn()
