import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block with optional down-sampling via stride."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class RobustClassifier(nn.Module):
    """
    ResNet-style classifier for 28x28 grey-scale images (10 classes).
    Heavy use of BatchNorm enables effective BNStats adaptation at test time.
    """

    def __init__(self):
        super().__init__()
        # --- stem ---
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # --- residual stages ---
        self.layer1 = nn.Sequential(ResBlock(64, 64), ResBlock(64, 64))       # 28x28
        self.layer2 = nn.Sequential(ResBlock(64, 128, stride=2),
                                    ResBlock(128, 128))                       # 14x14
        self.layer3 = nn.Sequential(ResBlock(128, 256, stride=2),
                                    ResBlock(256, 256))                       # 7x7
        # --- head ---
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        # x : [B, 1, 28, 28]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)  # logits [B, 10]

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"))
