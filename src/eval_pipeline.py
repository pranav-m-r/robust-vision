import copy
import logging
import os

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.adaptation import (
    TemperatureScaledModel,
    estimate_target_priors_bbse,
    evaluate_with_prior_correction,
    find_optimal_temperature,
    get_source_priors,
    recalibrate_bn_robust,
    tent_adapt,
)
from src.dataset import CustomPTDataset
from src.model import get_model
from src.task import evaluate, predict


class EvalPipeline:
    """
    Evaluation pipeline:
    1. Load base_model.pt.
    2. Temperature-scale on clean val_sanity data.
    3. BBSE label shift estimation:
       - Confusion matrix from source_toxic.pt (noisy labels, denoised via T⁻¹)
       - Target priors μ_t from static.pt (unlabelled)
    4. Robust BN recalibration on the target domain (static.pt).
    5. SAR entropy minimisation on the target domain (BN affine params only).
    6. Apply prior correction; save predictions.
    """

    def __init__(
        self,
        data: DictConfig,
        training: DictConfig,
        temperature: DictConfig,
        adaptation: DictConfig | None = None,
    ) -> None:
        self.data = data
        self.training = training
        self.temperature = temperature
        self.adaptation = adaptation

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
            split="test",          # no augmentation on clean val
            is_labelled=True,
            augmentation_factor=1,
        )
        # static.pt / target: unlabelled – BN recalibration + EM prior estimation
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
        # ── Read on/off switches ─────────────────────────────────────────
        def _flag(name: str, default: bool = True) -> bool:
            return getattr(self.adaptation, name, default) if self.adaptation else default

        do_temp    = _flag("enable_temp_scaling")
        do_bn      = _flag("enable_bn_recal")
        do_tent    = _flag("enable_tent")
        do_prior_e = _flag("enable_prior_estimation")
        do_prior_c = _flag("enable_prior_correction")

        logging.info("=" * 70)
        logging.info("PIPELINE SWITCHES:  temp=%s  bn=%s  tent=%s  prior_est=%s  prior_corr=%s",
                      do_temp, do_bn, do_tent, do_prior_e, do_prior_c)
        logging.info("=" * 70)

        model_original = copy.deepcopy(model)
        # Source priors: source_toxic is balanced (stated in problem spec) → uniform
        source_priors = torch.full(
            (self.training.num_classes,), 1.0 / self.training.num_classes,
            device=self.device,
        )

        # [0] Baseline ────────────────────────────────────────────────────────
        acc_baseline = evaluate(model_original, val_loader, self.device)
        logging.info("[0] Baseline (val_sanity): %.2f%%", acc_baseline)

        # [1] Temperature scaling ─────────────────────────────────────────────
        if do_temp:
            logging.info("\n[1] Temperature Scaling (calibrated on val_sanity)...")
            T = find_optimal_temperature(
                model_original,
                val_loader,
                self.device,
                t_min=self.temperature.t_min,
                t_max=self.temperature.t_max,
                steps=self.temperature.steps,
                noise_rate=0.0,  # val_sanity is clean
            )
            model_current = TemperatureScaledModel(
                copy.deepcopy(model_original), temperature=T
            ).to(self.device)

            acc_temp = evaluate(model_current, val_loader, self.device)
            logging.info(
                "    After temp scaling (val_sanity): %.2f%%  (Δ %+.2f%%)",
                acc_temp, acc_temp - acc_baseline,
            )
        else:
            logging.info("\n[1] Temperature Scaling: SKIPPED")
            model_current = copy.deepcopy(model_original)
            acc_temp = acc_baseline

        # Track the best model state — revert any step that hurts accuracy
        best_acc = acc_temp
        best_model_state = copy.deepcopy(model_current.state_dict())

        # [2] Alpha-blended BN recalibration ──────────────────────────────────
        if do_bn:
            bn_alpha = getattr(self.adaptation, "bn_alpha", 0.1) if self.adaptation else 0.1
            logging.info("\n[2] Alpha-blended BN Recalibration (α=%.2f on target domain)...", bn_alpha)
            model_current = recalibrate_bn_robust(
                copy.deepcopy(model_current), test_loader, self.device, alpha=bn_alpha
            )

            acc_rbn = evaluate(model_current, val_loader, self.device)
            logging.info(
                "    After BN recal (val_sanity): %.2f%%  (Δ %+.2f%%)",
                acc_rbn, acc_rbn - acc_baseline,
            )

            # Safety: revert if BN recal hurts by > 2%
            if acc_rbn < best_acc - 2.0:
                logging.warning("    ⚠ BN recal HURT accuracy (%.2f%% → %.2f%%). REVERTING.", best_acc, acc_rbn)
                model_current.load_state_dict(best_model_state)
                acc_rbn = best_acc
            else:
                best_acc = acc_rbn
                best_model_state = copy.deepcopy(model_current.state_dict())
        else:
            logging.info("\n[2] BN Recalibration: SKIPPED")
            acc_rbn = best_acc

        # [3] TENT entropy minimisation ───────────────────────────────────────
        if do_tent:
            tent_lr = getattr(self.adaptation, "tent_lr", 5e-4) if self.adaptation else 5e-4
            tent_steps = getattr(self.adaptation, "tent_steps", 2) if self.adaptation else 2
            e_margin = getattr(self.adaptation, "entropy_margin", 0.5) if self.adaptation else 0.5

            logging.info("\n[3] TENT / SAR entropy minimisation (on target domain)...")
            model_current = tent_adapt(
                model_current,
                test_loader,
                self.device,
                lr=tent_lr,
                steps=tent_steps,
                num_classes=self.training.num_classes,
                entropy_margin=e_margin,
            )

            acc_tent = evaluate(model_current, val_loader, self.device)
            logging.info(
                "    After TENT (val_sanity): %.2f%%  (Δ %+.2f%%)",
                acc_tent, acc_tent - acc_baseline,
            )

            # Safety: revert if TENT hurts by > 2%
            if acc_tent < best_acc - 2.0:
                logging.warning("    ⚠ TENT HURT accuracy (%.2f%% → %.2f%%). REVERTING.", best_acc, acc_tent)
                model_current.load_state_dict(best_model_state)
                acc_tent = best_acc
            else:
                best_acc = acc_tent
        else:
            logging.info("\n[3] TENT: SKIPPED")
            acc_tent = best_acc

        # Use the current model for final predictions
        model_final = model_current

        # [4] Prior estimation (from ADAPTED model's predictions) ─────────────
        # Default: unit weights (no correction)
        importance_weights = torch.ones(self.training.num_classes, device=self.device)

        if do_prior_e:
            logging.info("\n[4] Label shift estimation (from adapted model's target predictions)...")
            target_priors, importance_weights = estimate_target_priors_bbse(
                model=model_final,
                target_loader=test_loader,
                C=None,
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
        else:
            logging.info("\n[4] Prior estimation: SKIPPED (using uniform weights)")

        # [5] Prior correction at inference ───────────────────────────────────
        if do_prior_c and do_prior_e:
            logging.info("\n[5] Prior correction at inference...")
            acc_final = evaluate_with_prior_correction(
                model_final, val_loader, importance_weights, self.device
            )
            logging.info(
                "    After prior correction (val_sanity): %.2f%%  (Δ %+.2f%%)",
                acc_final, acc_final - acc_baseline,
            )

            # Safety: if prior correction hurts val_sanity, note it
            if acc_final < best_acc - 3.0:
                logging.warning(
                    "    ⚠ Prior correction hurts val_sanity (%.2f%% → %.2f%%). "
                    "Keeping estimated weights for target predictions anyway.",
                    best_acc, acc_final,
                )
        else:
            if not do_prior_c:
                logging.info("\n[5] Prior correction: SKIPPED")
            elif not do_prior_e:
                logging.info("\n[5] Prior correction: SKIPPED (no prior estimation)")
            importance_weights = torch.ones(self.training.num_classes, device=self.device)
            acc_final = best_acc

        # Summary ─────────────────────────────────────────────────────────────
        logging.info("=" * 70)
        logging.info("  Baseline (val_sanity):                 %.2f%%", acc_baseline)
        logging.info(
            "  + Temperature Scaling:                 %.2f%%  (Δ %+.2f%%)%s",
            acc_temp, acc_temp - acc_baseline, "" if do_temp else "  [OFF]",
        )
        logging.info(
            "  + Alpha-blended BN Recalibration:      %.2f%%  (Δ %+.2f%%)%s",
            acc_rbn, acc_rbn - acc_baseline, "" if do_bn else "  [OFF]",
        )
        logging.info(
            "  + TENT Entropy Minimisation:           %.2f%%  (Δ %+.2f%%)%s",
            acc_tent, acc_tent - acc_baseline, "" if do_tent else "  [OFF]",
        )
        logging.info(
            "  + Prior Correction:                    %.2f%%  (Δ %+.2f%%)%s",
            acc_final, acc_final - acc_baseline,
            "" if (do_prior_c and do_prior_e) else "  [OFF]",
        )
        logging.info("=" * 70)

        # Persist results ─────────────────────────────────────────────────────
        results_path = os.path.join(results_dir, "results.txt")
        with open(results_path, "w", encoding="utf-8") as f:
            f.write(f"Baseline (val_sanity):             {acc_baseline:.2f}%\n")
            f.write(f"+ Temperature Scaling:             {acc_temp:.2f}%{'  [OFF]' if not do_temp else ''}\n")
            f.write(f"+ Alpha-blended BN Recalibration:  {acc_rbn:.2f}%{'  [OFF]' if not do_bn else ''}\n")
            f.write(f"+ TENT Entropy Minimisation:       {acc_tent:.2f}%{'  [OFF]' if not do_tent else ''}\n")
            f.write(f"+ Prior Correction:                {acc_final:.2f}%{'  [OFF]' if not (do_prior_c and do_prior_e) else ''}\n")
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
            model_final, test_predict_loader, importance_weights, self.device,
            results_dir,
            filename="predictions.csv",
        )
