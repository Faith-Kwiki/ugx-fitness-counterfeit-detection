
import torch
import torch.nn as nn


class BanknoteCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Feature extraction: learn patterns from banknote images
        self.features = nn.Sequential(

            # Convolutional layer 1
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Convolutional layer 2
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Convolutional layer 3
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Reduce each feature map to one value
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification: genuine or counterfeit
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)

        return x
