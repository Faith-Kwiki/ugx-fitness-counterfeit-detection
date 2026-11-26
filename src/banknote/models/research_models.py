"""Run-ready RGB, ResNet-18, and multi-signal candidate models."""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from src.banknote.models.baseline_cnn import BanknoteCNN
from src.banknote.models.multisignal_cnn import CNNBranch, MultiSignalCNN
from src.banknote.models.signal_ablation_cnn import SignalAblationCNN


class RGBOnlyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = BanknoteCNN()

    def forward(self, rgb, edges, texture):
        return self.network(rgb)


class ResNet18Classifier(nn.Module):
    def __init__(self, mode="frozen", pretrained=True):
        super().__init__()
        if mode not in {"frozen", "finetune"}:
            raise ValueError("mode must be frozen or finetune")
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)
        self.mode = mode
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if mode == "frozen":
            for parameter in self.backbone.fc.parameters():
                parameter.requires_grad = True
        else:
            for name, parameter in self.backbone.named_parameters():
                if name.startswith(("layer3.", "layer4.", "fc.")):
                    parameter.requires_grad = True
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode=True):
        super().train(mode)
        if self.mode == "frozen":
            self.backbone.eval()
            self.backbone.fc.train(mode)
        else:
            self.backbone.bn1.eval()
            self.backbone.layer1.eval()
            self.backbone.layer2.eval()
        return self

    def forward(self, rgb, edges, texture):
        return self.backbone((rgb - self.mean) / self.std)


class HybridSignalModel(nn.Module):
    """Frozen pretrained RGB ResNet plus trainable edge and texture branches."""

    def __init__(self, fusion="concat", pretrained=True):
        super().__init__()
        if fusion not in {"concat", "attention"}:
            raise ValueError("fusion must be concat or attention")
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        feature_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.edge_branch = CNNBranch(1)
        self.texture_branch = CNNBranch(1)
        hidden_size = 256
        self.rgb_projection = nn.Sequential(
            nn.Linear(feature_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU()
        )
        self.edge_projection = nn.Sequential(
            nn.Linear(64, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU()
        )
        self.texture_projection = nn.Sequential(
            nn.Linear(64, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU()
        )
        self.fusion = fusion
        if fusion == "concat":
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size * 3, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 2),
            )
        else:
            self.attention = nn.Linear(hidden_size, 1)
            self.last_attention_weights = None
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 2),
            )
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, rgb, edges, texture):
        rgb_features = self.rgb_projection(
            self.backbone((rgb - self.mean) / self.std)
        )
        edge_features = self.edge_projection(self.edge_branch(edges))
        texture_features = self.texture_projection(self.texture_branch(texture))
        streams = torch.stack(
            (rgb_features, edge_features, texture_features), dim=1
        )
        if self.fusion == "concat":
            fused = streams.flatten(start_dim=1)
        else:
            weights = torch.softmax(self.attention(streams).squeeze(-1), dim=1)
            self.last_attention_weights = weights.detach()
            fused = torch.sum(streams * weights.unsqueeze(-1), dim=1)
        return self.classifier(fused)


def build_research_model(
    model_name, signals=("rgb", "edges", "texture"), pretrained=True
):
    if model_name == "rgb_cnn":
        return RGBOnlyCNN()
    if model_name == "resnet18_frozen":
        return ResNet18Classifier(mode="frozen", pretrained=pretrained)
    if model_name == "resnet18_finetune":
        return ResNet18Classifier(mode="finetune", pretrained=pretrained)
    if model_name == "hybrid_concat":
        return HybridSignalModel(fusion="concat", pretrained=pretrained)
    if model_name == "hybrid_attention":
        return HybridSignalModel(fusion="attention", pretrained=pretrained)
    if model_name == "multisignal_concat":
        return MultiSignalCNN(num_classes=2)
    if model_name == "signal_ablation":
        return SignalAblationCNN(signals=signals, num_classes=2)
    raise ValueError(f"Unknown model: {model_name}")
