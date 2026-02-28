import copy
import logging
import os

import torch
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
from src.model import get_model
from src.task import evaluate, predict


class EvalPipeline:
    """
    Evaluation pipeline:
    1. Load base_model.pt from data.model_path (saved by TrainPipeline).
    2. Temperature-scale on clean val_sanity data (labelled, unbiased).
    3. Recalibrate BatchNorm statistics on the unlabelled test domain (robust median).
    4. Estimate label shift with BBSE (confusion matrix from val, target priors from test).
    5. Apply prior correction; save predictions for val_sanity and test domain.

    Instantiated by Hydra via conf/config_eval.yaml.
    """

    def __init__(
        self,
        data: DictConfig,
        training: DictConfig,
        temperature: DictConfig,
    ) -> None:
        self.data = data
        self.training = training
        self.temperature = temperature

        torch.manual_seed(training.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Using device: %s", self.device)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, results_dir: str = ".") -> None:
        val_loader, test_loader, val_dataset = self._build_dataloaders()
        model = self._load_model()
        self._adapt_and_evaluate(model, val_loader, test_loader, val_dataset, results_dir)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_dataloaders(
        self,
    ) -> tuple[DataLoader, DataLoader, CustomPTDataset]:
        # val_sanity: clean and labelled – used for calibration and accuracy metrics
        val_dataset = CustomPTDataset(
            self.data.val_path,
            return_index=False,
            split="val",
            is_labelled=True,
            augmentation_factor=1,
        )
        # static.pt: unlabelled – used for BN recalibration and BBSE prior estimation
        test_dataset = CustomPTDataset(
            self.data.test_path,
            return_index=False,
            split="test",
            is_labelled=False,
            augmentation_factor=1,
        )

        val_loader = DataLoader(val_dataset, batch_size=self.training.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.training.batch_size)

        return val_loader, test_loader, val_dataset

    def _load_model(self) -> torch.nn.Module:
        model = get_model(num_classes=self.training.num_classes).to(self.device)
        state_dict = torch.load(
            self.data.model_path, map_location=self.device, weights_only=True
        )
        model.load_state_dict(state_dict)
        logging.info("Loaded model from: %s", self.data.model_path)
        return model

    def _adapt_and_evaluate(
        self,
        model: torch.nn.Module,
        val_loader: DataLoader,
        test_loader: DataLoader,
        val_dataset: CustomPTDataset,
        results_dir: str,
    ) -> None:
        logging.info("=" * 70)
        logging.info("PIPELINE: Temp Scaling → Robust BN → BBSE → Prior Correction")
        logging.info("=" * 70)

        model_original = copy.deepcopy(model)
        # Source priors from val_sanity (10 per class → exactly uniform [0.1, …])
        source_priors = get_source_priors(
            val_dataset, self.training.num_classes
        ).to(self.device)

        # [0] Baseline ────────────────────────────────────────────────────────
        acc_baseline = evaluate(model_original, val_loader, self.device)
        logging.info("[0] Baseline (val_sanity): %.2f%%", acc_baseline)

        # [1] Temperature scaling ─────────────────────────────────────────────
        logging.info("\n[1] Temperature Scaling (calibrated on val_sanity)...")
        T = find_optimal_temperature(
            model_original,
            val_loader,
            self.device,
            t_min=self.temperature.t_min,
            t_max=self.temperature.t_max,
            steps=self.temperature.steps,
            noise_rate=self.training.noise_rate,  # val_sanity is clean – no forward correction
        )
        model_temp = TemperatureScaledModel(
            copy.deepcopy(model_original), temperature=T
        ).to(self.device)

        acc_temp = evaluate(model_temp, val_loader, self.device)
        logging.info(
            "    After temp scaling (val_sanity): %.2f%%  (%.2f%%)",
            acc_temp, acc_temp - acc_baseline,
        )

        # [2] BBSE label shift estimation ─────────────────────────────────────
        # Run BBSE on model_temp (before BN recal) so C and mu_t come from the
        # same model and C is estimated while accuracy is still at its highest.
        logging.info("\n[2] BBSE label shift estimation (on model_temp)...")
        C_temp = compute_soft_confusion_matrix(
            model_temp,
            val_loader,
            self.training.num_classes,
            self.device,
            noise_rate=self.training.noise_rate,  # val_sanity is clean – no forward correction
        )
        target_priors, importance_weights = estimate_target_priors_bbse(
            model=model_temp,
            target_loader=test_loader,
            C=C_temp,
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

        # [3] Robust BN recalibration ─────────────────────────────────────────
        # Adapts BN statistics to the test domain using unlabelled static.pt.
        # Happens after BBSE so BN recal does not corrupt the confusion matrix.
        logging.info("\n[3] Robust BN Recalibration (median-of-means on test domain)...")
        model_adapted = recalibrate_bn_robust(model_temp, test_loader, self.device)

        acc_rbn = evaluate(model_adapted, val_loader, self.device)
        logging.info(
            "    After robust BN recal (val_sanity): %.2f%%  (%.2f%%)",
            acc_rbn, acc_rbn - acc_baseline,
        )

        # [4] Prior correction at inference ───────────────────────────────────
        # Importance weights from BBSE (step 2) applied to the BN-adapted model.
        logging.info("\n[4] Prior correction at inference (evaluated on val_sanity)...")
        acc_final = evaluate_with_prior_correction(
            model_adapted, val_loader, importance_weights, self.device
        )

        # Summary ─────────────────────────────────────────────────────────────
        logging.info("=" * 70)
        logging.info("  Baseline (val_sanity):                 %.2f%%", acc_baseline)
        logging.info(
            "  + Temperature Scaling:                 %.2f%%"
            "  (%.2f%%)", acc_temp, acc_temp - acc_baseline
        )
        logging.info(
            "  + BBSE Prior Correction:               %.2f%%"
            "  (%.2f%%)", acc_final, acc_final - acc_baseline
        )
        logging.info(
            "  + Robust BN Recalibration:             %.2f%%"
            "  (%.2f%%)", acc_rbn, acc_rbn - acc_baseline
        )
        logging.info("=" * 70)

        # Persist results ─────────────────────────────────────────────────────
        results_path = os.path.join(results_dir, "results.txt")
        with open(results_path, "w", encoding="utf-8") as f:
            f.write(f"Baseline (val_sanity):             {acc_baseline:.2f}%\n")
            f.write(f"+ Temperature Scaling:             {acc_temp:.2f}%\n")
            f.write(f"+ BBSE Prior Correction:           {acc_final:.2f}%\n")
            f.write(f"+ Robust BN Recalibration:         {acc_rbn:.2f}%\n")
        logging.info("Results written to: %s", results_path)

        # Predictions on test domain (Kaggle submission) ──────────────────────
        test_predict_dataset = CustomPTDataset(
            self.data.test_path,
            return_index=True,
            split="test",
            is_labelled=False,
            augmentation_factor=1,
        )
        test_predict_loader = DataLoader(
            test_predict_dataset, batch_size=self.training.batch_size
        )
        predict(
            model_adapted, test_predict_loader, importance_weights, self.device,
            results_dir,
            filename="predictions.csv",
        )
