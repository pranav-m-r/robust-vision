import copy
import logging
import os
import random

import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.adaptation import (
    TemperatureScaledModel,
    compute_soft_confusion_matrix,
    estimate_target_priors_bbse,
    evaluate_with_prior_correction,
    find_optimal_temperature,
    get_source_priors,
    recalibrate_bn_robust,
)
from src.dataset import CustomPTDataset
from src.loss import TruncatedLoss
from src.model import get_model
from src.train import evaluate, train_epoch


class Pipeline:
    """
    Full robust-CV pipeline:
    1. Train on noisy source data with Truncated Loss.
    2. Temperature-scale the model on clean validation data.
    3. Recalibrate BatchNorm statistics on the target domain (robust median).
    4. Estimate label shift with BBSE and apply prior correction at inference.

    Instantiated by Hydra; every hyperparameter is passed through the config.
    """

    def __init__(
        self,
        data: DictConfig,
        training: DictConfig,
        loss: DictConfig,
        temperature: DictConfig,
    ) -> None:
        self.data = data
        self.training = training
        self.loss = loss
        self.temperature = temperature

        torch.manual_seed(training.seed)
        np.random.seed(training.seed)
        random.seed(training.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Using device: %s", self.device)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, results_dir: str = ".") -> None:
        source_loader, val_loader, target_loader, source_dataset, val_dataset = self._build_dataloaders()
        model = self._train(source_loader, val_loader, source_dataset)
        self._adapt_and_evaluate(model, val_loader, target_loader, val_dataset, results_dir)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_dataloaders(
        self,
    ) -> tuple[DataLoader, DataLoader, DataLoader, CustomPTDataset, CustomPTDataset]:
        source_dataset = CustomPTDataset(
            self.data.source_path,
            return_index=True,
            split="train",
            augmentation_factor=self.training.augmentation_factor,
        )
        val_dataset = CustomPTDataset(
            self.data.val_path,
            return_index=False,
            split="val",
            augmentation_factor=self.training.augmentation_factor,
        )
        target_dataset = CustomPTDataset(self.data.target_path, return_index=False, split="target")

        source_loader = DataLoader(source_dataset, batch_size=self.training.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.training.batch_size)
        target_loader = DataLoader(target_dataset, batch_size=self.training.batch_size)

        return source_loader, val_loader, target_loader, source_dataset, val_dataset

    def _train(
        self,
        source_loader: DataLoader,
        val_loader: DataLoader,
        source_dataset: CustomPTDataset,
    ) -> torch.nn.Module:
        model = get_model(num_classes=self.training.num_classes).to(self.device)
        criterion = TruncatedLoss(
            q=self.loss.q,
            k=self.loss.k,
            trainset_size=len(source_dataset),
        ).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.training.lr)

        for epoch in range(self.training.epochs):
            loss = train_epoch(model, source_loader, criterion, optimizer, self.device)
            val_acc = evaluate(model, val_loader, self.device)

            logging.info(
                "Epoch %d/%d | Loss: %.4f | Val Acc: %.2f%%",
                epoch + 1,
                self.training.epochs,
                loss,
                val_acc,
            )

        return model

    def _adapt_and_evaluate(
        self,
        model: torch.nn.Module,
        val_loader: DataLoader,
        target_loader: DataLoader,
        val_dataset: CustomPTDataset,
        results_dir: str,
    ) -> None:
        logging.info("=" * 70)
        logging.info("PIPELINE: Temp Scaling → Robust BN → BBSE → Prior Correction")
        logging.info("=" * 70)

        model_original = copy.deepcopy(model)
        # Source priors are estimated from the clean validation set
        source_priors = get_source_priors(
            val_dataset, self.training.num_classes
        ).to(self.device)

        # [0] Baseline ────────────────────────────────────────────────────────
        acc_baseline = evaluate(model_original, target_loader, self.device)
        logging.info("[0] Baseline: %.2f%%", acc_baseline)

        # [1] Temperature scaling ─────────────────────────────────────────────
        logging.info("\n[1] Temperature Scaling (calibrated on val_sanity)...")
        T = find_optimal_temperature(
            model_original,
            val_loader,
            self.device,
            t_min=self.temperature.t_min,
            t_max=self.temperature.t_max,
            steps=self.temperature.steps,
        )
        model_temp = TemperatureScaledModel(
            copy.deepcopy(model_original), temperature=T
        ).to(self.device)

        acc_temp = evaluate(model_temp, target_loader, self.device)
        logging.info("    After temp scaling: %.2f%%  (%.2f%%)", acc_temp, acc_temp - acc_baseline)

        # [2] Robust BN recalibration ─────────────────────────────────────────
        logging.info("\n[2] Robust BN Recalibration (median-of-means)...")
        model_adapted = recalibrate_bn_robust(model_temp, target_loader, self.device)

        acc_rbn = evaluate(model_adapted, target_loader, self.device)
        logging.info("    After robust BN recal: %.2f%%  (%.2f%%)", acc_rbn, acc_rbn - acc_baseline)

        # [3] BBSE label shift estimation ─────────────────────────────────────
        logging.info("\n[3] BBSE label shift estimation...")
        C_adapted = compute_soft_confusion_matrix(
            model_adapted, val_loader, self.training.num_classes, self.device
        )
        target_priors, importance_weights = estimate_target_priors_bbse(
            model=model_adapted,
            target_loader=target_loader,
            C=C_adapted,
            source_priors=source_priors,
            device=self.device,
            num_classes=self.training.num_classes,
        )

        logging.info("    Class-wise shift:")
        for c in range(self.training.num_classes):
            bar = "▲" if importance_weights[c] > 1.0 else "▼"
            logging.info(
                f"      Class {c}: π_s={source_priors[c]:.3f} → "
                f"π_t={target_priors[c]:.3f}  w={importance_weights[c]:.3f} {bar}"
            )

        # [4] Prior correction at inference ───────────────────────────────────
        logging.info("\n[4] Prior correction at inference...")
        acc_final = evaluate_with_prior_correction(
            model_adapted, target_loader, importance_weights, self.device
        )

        # Summary ─────────────────────────────────────────────────────────────
        logging.info("=" * 70)
        logging.info("  Baseline:                              %.2f%%", acc_baseline)
        logging.info(
            "  + Temperature Scaling:                 %.2f%%"
            "  (%.2f%%)", acc_temp, acc_temp - acc_baseline
        )
        logging.info(
            "  + Robust BN Recalibration:             %.2f%%"
            "  (%.2f%%)", acc_rbn, acc_rbn - acc_baseline
        )
        logging.info(
            "  + BBSE Prior Correction:               %.2f%%"
            "  (%.2f%%)", acc_final, acc_final - acc_baseline
        )
        logging.info("=" * 70)

        # Persist results ─────────────────────────────────────────────────────
        results_path = os.path.join(results_dir, "results.txt")
        with open(results_path, "w", encoding="utf-8") as f:
            f.write(f"Baseline:                    {acc_baseline:.2f}%\n")
            f.write(f"+ Temperature Scaling:       {acc_temp:.2f}%\n")
            f.write(f"+ Robust BN Recalibration:   {acc_rbn:.2f}%\n")
            f.write(f"+ BBSE Prior Correction:     {acc_final:.2f}%\n")
        logging.info("Results written to: %s", results_path)
