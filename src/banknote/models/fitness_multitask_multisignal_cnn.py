"""Multi-signal model for joint outcome, authenticity, condition, and severity."""

import torch
import torch.nn as nn

from src.banknote.models.multisignal_cnn import CNNBranch


CONDITION_NAMES = ("soiled", "creased", "torn", "folded")
FITNESS_OUTCOMES = ("counterfeit", "genuine_fit", "genuine_unfit")
FITNESS_STATUSES = ("fit", "unfit")
SEVERITY_NAMES = ("none", "mild", "moderate", "severe")


def joint_outcome_probabilities(authenticity_logits, fitness_logits):
    """Compose authenticity and genuine-only fitness probabilities in class order."""
    authenticity = authenticity_logits.softmax(dim=1)
    fitness = fitness_logits.softmax(dim=1)
    return torch.stack(
        (
            authenticity[:, 0],
            authenticity[:, 1] * fitness[:, 0],
            authenticity[:, 1] * fitness[:, 1],
        ),
        dim=1,
    )


def joint_outcome_predictions(authenticity_logits, fitness_logits):
    """Return final labels, enforcing counterfeit priority over condition."""
    counterfeit = authenticity_logits.argmax(dim=1) == 0
    genuine_status = fitness_logits.argmax(dim=1) + 1
    return torch.where(counterfeit, torch.zeros_like(genuine_status), genuine_status)


class FitnessMultiTaskMultiSignalCNN(nn.Module):
    """Shared RGB, edge, and texture encoder with related output heads."""

    def __init__(self):
        super().__init__()
        self.rgb_branch = CNNBranch(input_channels=3)
        self.edge_branch = CNNBranch(input_channels=1)
        self.texture_branch = CNNBranch(input_channels=1)
        self.shared_features = nn.Sequential(
            nn.Linear(64 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.authenticity_head = nn.Linear(128, 2)
        # This head predicts circulation fitness conditional on authenticity.
        # The final three-way label is composed with authenticity at inference.
        self.fitness_head = nn.Linear(128, len(FITNESS_STATUSES))
        self.condition_heads = nn.ModuleDict(
            {name: nn.Linear(128, 1) for name in CONDITION_NAMES}
        )
        self.severity_heads = nn.ModuleDict(
            {name: nn.Linear(128, len(SEVERITY_NAMES)) for name in CONDITION_NAMES}
        )

    def forward(self, rgb, edges, texture):
        streams = torch.cat(
            (
                self.rgb_branch(rgb),
                self.edge_branch(edges),
                self.texture_branch(texture),
            ),
            dim=1,
        )
        shared = self.shared_features(streams)
        condition_logits = torch.cat(
            [self.condition_heads[name](shared) for name in CONDITION_NAMES], dim=1
        )
        severity_logits = torch.stack(
            [self.severity_heads[name](shared) for name in CONDITION_NAMES], dim=1
        )
        return {
            "authenticity_logits": self.authenticity_head(shared),
            "fitness_logits": self.fitness_head(shared),
            "condition_logits": condition_logits,
            "severity_logits": severity_logits,
        }
