import csv
import logging
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from src.loss import TruncatedLoss


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: TruncatedLoss,
    optimizer: Optimizer,
    device: torch.device,
    scheduler: Optional[LRScheduler] = None,
    grad_clip_norm: float = 0.0,
) -> float:
    """
    Run one full training epoch over the source loader.

    The source loader must yield (images, labels, indexes) triples
    (i.e. CustomPTDataset with return_index=True).

    Parameters
    ----------
    scheduler      : if provided, .step() is called once per batch (for
                     CosineAnnealingWarmRestarts / OneCycleLR-style schedulers).
    grad_clip_norm : if > 0, clips gradient norms before optimizer.step().

    Returns the mean batch loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    for images, labels, indexes in loader:
        images = images.to(device)
        labels = labels.to(device)
        indexes = indexes.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels, indexes)
        loss.backward()

        if grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        criterion.update_weight(outputs.detach(), labels, indexes)

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate top-1 accuracy (%) on the given loader.

    The loader must yield (images, labels) pairs.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100.0 * correct / total


def predict(
    model: nn.Module,
    loader: DataLoader,
    importance_weights: torch.Tensor,
    device: torch.device,
    results_dir: str,
    filename: str = "predictions.csv",
) -> None:
    """
    Run inference on an unlabelled loader and save predictions to a CSV file.

    The loader must yield (images, indices) pairs, i.e. a CustomPTDataset
    with is_labelled=False and return_index=True.

    The output CSV is written to <results_dir>/<filename> with columns:
        idx   – original dataset index
        label – predicted class label
    """
    model.eval()
    all_indices: list[int] = []
    all_preds: list[int] = []

    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device)
            probs = F.softmax(model(images), dim=1)
            corrected = probs * importance_weights.unsqueeze(0)
            preds = corrected.argmax(dim=1)

            all_indices.extend(indices.tolist())
            all_preds.extend(preds.cpu().tolist())

    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, filename)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "label"])
        writer.writerows(zip(all_indices, all_preds))

    logging.info("Predictions saved to: %s", csv_path)
