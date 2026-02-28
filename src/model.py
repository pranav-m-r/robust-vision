import torch.nn as nn
import torchvision.models as models


def get_model(num_classes: int = 10) -> nn.Module:
    """
    ResNet-18 adapted for single-channel 28×28 images.

    Changes from the standard architecture:
    - conv1: 3×3 kernel, stride 1, padding 1 (replaces the original 7×7)
    - maxpool: replaced with Identity (no spatial downsampling at input)
    - fc: output size set to num_classes
    """
    model = models.resnet18(weights=None)

    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
