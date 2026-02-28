import torch
import torch.nn as nn
import torchvision.models as models


class RobustClassifier(nn.Module):
    """
    ResNet-18 adapted for single-channel 28×28 greyscale images.

    Architecture changes from standard ResNet-18:
      - conv1: 3×3 kernel, stride 1, padding 1 (instead of 7×7)
      - maxpool: Identity (no early spatial downsampling)
      - fc: 10-class output

    Submission interface for Hackenza 2026.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        base = models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, 1, 28, 28]
        return self.model(x)

    def load_weights(self, path: str) -> None:
        state_dict = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state_dict)
