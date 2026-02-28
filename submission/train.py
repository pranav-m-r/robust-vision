"""
train.py – Standalone reproducible training script for Hackenza 2026.

Usage:
    python train.py                              # defaults
    python train.py --data_path source_toxic.pt  # override source path

Produces weights.pth in the current directory.
No external dependencies beyond PyTorch and torchvision.
No pre-trained weights are loaded; the model is initialised from scratch.
"""

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms.v2 as transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def get_model(num_classes: int = 10) -> nn.Module:
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _build_train_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(28, padding=4, fill=0),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3), value=0),
    ])


class IndexedDataset(Dataset):
    def __init__(self, path: str, augmentation_factor: int = 1):
        data = torch.load(path, weights_only=True)
        self.images = data["images"].float()
        self.labels = data["labels"].long()
        self.augmentation_factor = max(1, augmentation_factor)
        self._base_len = len(self.images)
        self.transform = _build_train_transforms()

    def __len__(self) -> int:
        return self._base_len * self.augmentation_factor

    def __getitem__(self, idx: int):
        orig_idx = idx % self._base_len
        x = self.transform(self.images[orig_idx])
        y = self.labels[orig_idx]
        return x, y, orig_idx


# ---------------------------------------------------------------------------
# Truncated Loss (Lq with truncation)
# ---------------------------------------------------------------------------

class TruncatedLoss(nn.Module):
    def __init__(self, q: float = 0.7, k: float = 0.5, trainset_size: int = 50000):
        super().__init__()
        self.q = q
        self.k = k
        self.weight = nn.Parameter(
            torch.ones(trainset_size, 1), requires_grad=False
        )

    def forward(self, logits, targets, indexes):
        p = torch.clamp(F.softmax(logits, dim=1), min=1e-8)
        Yg = torch.gather(p, 1, targets.unsqueeze(1))
        Lq = (1 - Yg.pow(self.q)) / self.q
        Lqk = (1 - self.k ** self.q) / self.q
        return ((Lq - Lqk) * self.weight[indexes]).mean()

    def update_weight(self, logits, targets, indexes):
        with torch.no_grad():
            p = torch.clamp(F.softmax(logits, dim=1), min=1e-8)
            Yg = torch.gather(p, 1, targets.unsqueeze(1))
            Lq = (1 - Yg.pow(self.q)) / self.q
            Lqk = (1 - self.k ** self.q) / self.q
            self.weight[indexes] = (Lq < Lqk).float()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    data_path: str = "source_toxic.pt",
    output_path: str = "weights.pth",
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 2,
    eta_min: float = 1e-6,
    grad_clip_norm: float = 1.0,
    augmentation_factor: int = 1,
    q: float = 0.7,
    k: float = 0.5,
    seed: int = 42,
    num_classes: int = 10,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    dataset = IndexedDataset(data_path, augmentation_factor=augmentation_factor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = get_model(num_classes=num_classes).to(device)
    criterion = TruncatedLoss(q=q, k=k, trainset_size=len(dataset)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler: linear warmup → cosine annealing
    steps_per_epoch = len(loader)
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch

    warmup_scheduler = LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=total_steps - warmup_steps, T_mult=1, eta_min=eta_min
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # Train
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, labels, indexes in loader:
            images, labels, indexes = (
                images.to(device), labels.to(device), indexes.to(device)
            )

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels, indexes)
            loss.backward()

            if grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()
            scheduler.step()

            criterion.update_weight(outputs.detach(), labels, indexes)
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | LR: {cur_lr:.2e}")

    torch.save(model.state_dict(), output_path)
    print(f"Saved weights to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RobustClassifier on source_toxic.pt")
    parser.add_argument("--data_path", default="source_toxic.pt")
    parser.add_argument("--output_path", default="weights.pth")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--augmentation_factor", type=int, default=1)
    parser.add_argument("--q", type=float, default=0.7)
    parser.add_argument("--k", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        data_path=args.data_path,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        eta_min=args.eta_min,
        grad_clip_norm=args.grad_clip_norm,
        augmentation_factor=args.augmentation_factor,
        q=args.q,
        k=args.k,
        seed=args.seed,
    )
