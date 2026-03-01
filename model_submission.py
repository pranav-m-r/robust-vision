"""
model_submission.py — RobustClassifier Architecture
====================================================
ResNet-style CNN with Batch Normalization for Test-Time Adaptation.

Architecture choices:
  - Residual blocks: stabilise gradient flow, improve robustness.
  - BatchNorm at every conv layer: enables BN-stats adaptation and TENT at
    test time (Phase 3).
  - No dropout: dropout interferes with BN-stats adaptation (BN would see
    dropout-corrupted activations during the adaptation forward passes).
  - Kaiming initialisation: random weights only — no pretrained models.

Capacity: ~870 K parameters (32 → 64 → 128 channels).
Input : [B, 1, 28, 28]  greyscale, pixels in [0, 1]
Output: [B, 10]          raw logits (10 Fashion-MNIST classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Building block: two-conv residual block
# ------------------------------------------------------------------
class ResBlock(nn.Module):
    """
    Pre-activation style residual block (identity shortcut).
    Conv → BN → ReLU → Conv → BN  +  skip  → ReLU
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


# ------------------------------------------------------------------
# Main classifier
# ------------------------------------------------------------------
class RobustClassifier(nn.Module):
    """
    3-stage ResNet for 28×28 greyscale images (10 classes).

    Stage   Channels  Spatial   Blocks
    -----   --------  -------   ------
    stem        32     28×28       –
    stage1      32     28×28       2
    down1       64     14×14       –
    stage2      64     14×14       2
    down2      128      7×7        –
    stage3     128      7×7        2
    GAP       128      1×1        –
    FC         10        –         –
    """

    def __init__(self):
        super().__init__()

        # --- stem ---
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # --- stage 1 (28×28, 32 ch) ---
        self.stage1 = nn.Sequential(ResBlock(32), ResBlock(32))

        # --- downsample 28→14, 32→64 ---
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # --- stage 2 (14×14, 64 ch) ---
        self.stage2 = nn.Sequential(ResBlock(64), ResBlock(64))

        # --- downsample 14→7, 64→128 ---
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # --- stage 3 (7×7, 128 ch) ---
        self.stage3 = nn.Sequential(ResBlock(128), ResBlock(128))

        # --- head ---
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(128, 10)

        # random init (Kaiming)
        self._init_weights()

    # ---- weight initialisation (random, no pretrained) ----
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    # ---- forward pass ----
    def forward(self, x):
        # x: [B, 1, 28, 28]
        x = self.stem(x)      # [B,  32, 28, 28]
        x = self.stage1(x)    # [B,  32, 28, 28]
        x = self.down1(x)     # [B,  64, 14, 14]
        x = self.stage2(x)    # [B,  64, 14, 14]
        x = self.down2(x)     # [B, 128,  7,  7]
        x = self.stage3(x)    # [B, 128,  7,  7]
        x = self.pool(x)      # [B, 128,  1,  1]
        x = x.view(x.size(0), -1)
        return self.fc(x)     # [B, 10]

    # ---- required interface ----
    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"))
