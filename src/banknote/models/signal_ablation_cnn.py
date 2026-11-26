"""Small branch CNN used for controlled multi-signal ablations."""

import torch
import torch.nn as nn

from src.banknote.models.multisignal_cnn import CNNBranch


class SignalAblationCNN(nn.Module):
    """Concatenate features from only the selected input signals."""

    INPUT_CHANNELS = {"rgb": 3, "edges": 1, "texture": 1}
    ALLOWED_SIGNALS = ("rgb", "edges", "texture")

    def __init__(self, signals, num_classes=2):
        super().__init__()
        signals = tuple(signals)
        if not signals or len(set(signals)) != len(signals):
            raise ValueError("signals must be non-empty and contain no duplicates")
        unknown = set(signals) - set(self.ALLOWED_SIGNALS)
        if unknown:
            raise ValueError(f"Unknown signals: {sorted(unknown)}")

        self.signals = signals
        self.branches = nn.ModuleDict(
            {
                signal: CNNBranch(self.INPUT_CHANNELS[signal])
                for signal in signals
            }
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * len(signals), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, rgb, edges, texture):
        inputs = {"rgb": rgb, "edges": edges, "texture": texture}
        features = [
            self.branches[signal](inputs[signal])
            for signal in self.signals
        ]
        return self.classifier(torch.cat(features, dim=1))
