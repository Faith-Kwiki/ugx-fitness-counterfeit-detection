
# ============================================================
# MULTI-SIGNAL CNN
# THREE CNN BRANCHES + FEATURE CONCATENATION
# ============================================================

import torch
import torch.nn as nn


# 1. REUSABLE CNN FEATURE EXTRACTOR

class CNNBranch(nn.Module):

    def __init__(self, input_channels):

        super().__init__()

        self.features = nn.Sequential(

            # First convolution block
            nn.Conv2d(
                input_channels, 16,
                kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Second convolution block
            nn.Conv2d(
                16, 32,
                kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Third convolution block
            nn.Conv2d(
                32, 64,
                kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Convert feature maps into a fixed-size vector
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):

        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        return x


# 2. THREE-BRANCH MULTI-SIGNAL CNN

class MultiSignalCNN(nn.Module):

    def __init__(self, num_classes=2):

        super().__init__()

        # RGB branch: 3 input channels
        self.rgb_branch = CNNBranch(
            input_channels=3
        )

        # Structural branch: 1 input channel
        self.edge_branch = CNNBranch(
            input_channels=1
        )

        # Texture branch: 1 input channel
        self.texture_branch = CNNBranch(
            input_channels=1
        )

        # Each branch produces 64 features
        self.fusion_size = 64 * 3

        # Classification head
        self.classifier = nn.Sequential(

            nn.Linear(self.fusion_size, 128),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, rgb, edges, texture):

        # Extract features independently
        rgb_features = self.rgb_branch(rgb)

        edge_features = self.edge_branch(edges)

        texture_features = self.texture_branch(texture)

        # Combine all three feature vectors
        combined_features = torch.cat(
            [
                rgb_features,
                edge_features,
                texture_features
            ],
            dim=1
        )

        # Predict authenticity
        output = self.classifier(combined_features)

        return output
