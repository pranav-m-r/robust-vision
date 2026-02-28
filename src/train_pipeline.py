import copy
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.dataset import CustomPTDataset
from src.loss import TruncatedLoss, SCELoss
from src.model import get_model
from src.task import train_epoch, evaluate


class TrainPipeline:
    """
    Training pipeline (improved):
    1. Trains on source_toxic.pt (full noisy dataset) with warmup + truncated loss.
    2. Monitors val_sanity.pt each epoch and saves the best model (+ EMA).
    3. Uses Adam with cosine-annealing LR schedule and weight decay.

    Key improvements over the original:
    - CE warmup for the first N epochs before truncation kicks in
    - EMA (Exponential Moving Average) of weights for better generalisation
    - Cosine-annealing LR schedule
    - Weight decay regularisation
    - Augmentation factor > 1 to see each image with diverse transforms
    - Best-model checkpointing based on clean val_sanity accuracy
    """

    def __init__(
        self,
        data: DictConfig,
        training: DictConfig,
        loss: DictConfig,
    ) -> None:
        self.data = data
        self.training = training
        self.loss = loss

        torch.manual_seed(training.seed)
        np.random.seed(training.seed)
        random.seed(training.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Using device: %s", self.device)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, results_dir: str = ".") -> None:
        train_loader, train_dataset, val_loader = self._build_dataloaders()
        model = self._train(train_loader, train_dataset, val_loader)

        model_path = os.path.join(results_dir, "base_model.pt")
        torch.save(model.state_dict(), model_path)
        logging.info("Base model saved to: %s", model_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_dataloaders(
        self,
    ) -> tuple[DataLoader, CustomPTDataset, DataLoader]:
        # Train on the full source_toxic.pt (no splitting)
        train_dataset = CustomPTDataset(
            self.data.train_path,
            return_index=True,
            split="train",
            is_labelled=True,
            augmentation_factor=self.training.augmentation_factor,
        )
        # Clean val_sanity.pt for monitoring (no augmentation)
        val_dataset = CustomPTDataset(
            self.data.val_path,
            return_index=False,
            split="test",          # no augmentation for clean eval
            is_labelled=True,
            augmentation_factor=1,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training.batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training.batch_size,
            shuffle=False,
        )

        return train_loader, train_dataset, val_loader

    def _train(
        self,
        train_loader: DataLoader,
        train_dataset: CustomPTDataset,
        val_loader: DataLoader,
    ) -> torch.nn.Module:
        model = get_model(num_classes=self.training.num_classes).to(self.device)

        # ── EMA model for better generalisation ──────────────────────────
        ema_model = copy.deepcopy(model)
        ema_decay = 0.999

        # ── Losses ───────────────────────────────────────────────────────
        # Truncated loss for the robust phase
        truncated_criterion = TruncatedLoss(
            q=self.loss.q,
            k=self.loss.k,
            trainset_size=train_dataset._base_len,  # base size, not augmented
        ).to(self.device)

        # Standard CE for warmup phase
        ce_criterion = nn.CrossEntropyLoss()

        warmup_epochs = getattr(self.loss, "warmup_epochs", 10)

        # ── Optimizer + scheduler ────────────────────────────────────────
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.training.lr,
            weight_decay=self.training.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.training.epochs,
            eta_min=1e-6,
        )

        best_val_acc = 0.0
        best_model_state = None

        for epoch in range(self.training.epochs):
            # Warmup: use CE (no truncation) for the first N epochs
            if epoch < warmup_epochs:
                loss = self._train_epoch_warmup(
                    model, train_loader, ce_criterion, optimizer
                )
                phase = "CE-warmup"
            else:
                loss = train_epoch(
                    model, train_loader, truncated_criterion, optimizer, self.device
                )
                phase = "Truncated"

            scheduler.step()

            # ── Update EMA ───────────────────────────────────────────────
            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
                for ema_b, b in zip(ema_model.buffers(), model.buffers()):
                    ema_b.data.copy_(b.data)

            # ── Validate ─────────────────────────────────────────────────
            val_acc = evaluate(model, val_loader, self.device)
            ema_acc = evaluate(ema_model, val_loader, self.device)

            current_lr = scheduler.get_last_lr()[0]
            logging.info(
                "Epoch %3d/%d [%s] | Loss: %.4f | Val: %.2f%% | "
                "EMA: %.2f%% | LR: %.2e",
                epoch + 1,
                self.training.epochs,
                phase,
                loss,
                val_acc,
                ema_acc,
                current_lr,
            )

            # ── Checkpoint best model ────────────────────────────────────
            pick_acc = max(val_acc, ema_acc)
            if pick_acc > best_val_acc:
                best_val_acc = pick_acc
                best_model_state = copy.deepcopy(
                    ema_model.state_dict() if ema_acc >= val_acc else model.state_dict()
                )
                logging.info(
                    "  ★ New best: %.2f%% (from %s)",
                    best_val_acc,
                    "EMA" if ema_acc >= val_acc else "model",
                )

        logging.info("Training complete – best val_sanity acc: %.2f%%", best_val_acc)

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        return model

    # ------------------------------------------------------------------

    def _train_epoch_warmup(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """One epoch of standard CE training (warmup phase)."""
        model.train()
        total_loss = 0.0

        for images, labels, _indexes in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)
