import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.loss import TruncatedLoss


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: TruncatedLoss,
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    """
    Run one full training epoch over the source loader.

    The source loader must yield (images, labels, indexes) triples
    (i.e. CustomPTDataset with return_index=True).

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
        optimizer.step()

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
