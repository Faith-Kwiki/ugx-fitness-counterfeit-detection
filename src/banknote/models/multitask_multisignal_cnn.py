"""Shared RGB/edge/texture model with authenticity and condition heads."""

import torch
import torch.nn as nn

from src.banknote.models.multisignal_cnn import CNNBranch


CONDITION_NAMES = ("soiled", "creased", "torn", "folded")


class MultiTaskMultiSignalCNN(nn.Module):
    """One fused representation feeding five independent prediction heads."""

    def __init__(self, num_authenticity_classes=2):
        super().__init__()
        self.rgb_branch = CNNBranch(input_channels=3)
        self.edge_branch = CNNBranch(input_channels=1)
        self.texture_branch = CNNBranch(input_channels=1)

        self.shared_features = nn.Sequential(
            nn.Linear(64 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.authenticity_head = nn.Linear(128, num_authenticity_classes)
        self.condition_heads = nn.ModuleDict(
            {name: nn.Linear(128, 1) for name in CONDITION_NAMES}
        )

    def forward(self, rgb, edges, texture):
        features = torch.cat(
            (
                self.rgb_branch(rgb),
                self.edge_branch(edges),
                self.texture_branch(texture),
            ),
            dim=1,
        )
        shared = self.shared_features(features)
        condition_logits = torch.cat(
            [self.condition_heads[name](shared) for name in CONDITION_NAMES],
            dim=1,
        )
        return {
            "authenticity_logits": self.authenticity_head(shared),
            "condition_logits": condition_logits,
        }
