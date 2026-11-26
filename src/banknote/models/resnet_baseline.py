
import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

class BanknoteResNet18(nn.Module):

    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()

        # Load ResNet-18 with pretrained ImageNet weights
        self.model = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # Freeze the pretrained feature extractor initially
        if freeze_backbone:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

        # Replace the original ImageNet classifier
        input_features = self.model.fc.in_features

        self.model.fc = nn.Linear(
            input_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)