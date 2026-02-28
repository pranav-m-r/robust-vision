"""
submit.py – Generate the final Kaggle submission CSV.

Runs the full EvalPipeline adaptation stack (temperature scaling, BBSE,
robust BN recalibration) and then predicts on:

  1. static.pt         → the main public-leaderboard target
  2. test_suite_public.pt → 24 anonymous scenarios for private evaluation

Produces a single  submission.csv  with columns:
    ID    – "<scenario>_<idx>"  (e.g. "static_0042", "scenario_03_0007")
    label – predicted class (0–9)

Usage:
    python -m src.submit                          # defaults from config.yaml
    python -m src.submit --model_path results/train/<ts>/base_model.pt
"""

import argparse
import copy
import csv
import logging
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.adaptation import (
    TemperatureScaledModel,
    compute_soft_confusion_matrix,
    estimate_target_priors_bbse,
    find_optimal_temperature,
    get_source_priors,
    recalibrate_bn_robust,
)
from src.dataset import CustomPTDataset
from src.model import get_model


def _predict_tensor(
    model: torch.nn.Module,
    images: torch.Tensor,
    importance_weights: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> list[int]:
    """Run prior-corrected inference on a raw image tensor."""
    model.eval()
    loader = DataLoader(TensorDataset(images), batch_size=batch_size)
    preds: list[int] = []

    with torch.no_grad():
        for (batch,) in loader:
            probs = F.softmax(model(batch.to(device)), dim=1)
            corrected = probs * importance_weights.unsqueeze(0)
            preds.extend(corrected.argmax(dim=1).cpu().tolist())

    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Kaggle submission CSV")
    parser.add_argument("--model_path", required=True, help="Path to base_model.pt")
    parser.add_argument("--val_path", default="data/val_sanity.pt",
                        help="Clean labelled val set for calibration")
    parser.add_argument("--test_path", default="data/static.pt",
                        help="Unlabelled target domain (static.pt)")
    parser.add_argument("--test_suite_path", default="data/test_suite_public.pt",
                        help="24-scenario test suite")
    parser.add_argument("--output_dir", default="submission",
                        help="Directory to write submission.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--t_min", type=float, default=0.1)
    parser.add_argument("--t_max", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model = get_model(num_classes=args.num_classes).to(device)
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    logging.info("Loaded model from %s", args.model_path)

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    val_dataset = CustomPTDataset(
        args.val_path, return_index=False, split="test",
        is_labelled=True, augmentation_factor=1,
    )
    test_dataset = CustomPTDataset(
        args.test_path, return_index=False, split="test",
        is_labelled=False, augmentation_factor=1,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # ------------------------------------------------------------------
    # Adaptation: Temp Scaling → BBSE → Robust BN
    # ------------------------------------------------------------------
    model_original = copy.deepcopy(model)
    source_priors = get_source_priors(val_dataset, args.num_classes).to(device)

    # [1] Temperature scaling
    logging.info("Temperature scaling...")
    T = find_optimal_temperature(model_original, val_loader, device,
                                  t_min=args.t_min, t_max=args.t_max)
    model_temp = TemperatureScaledModel(
        copy.deepcopy(model_original), temperature=T
    ).to(device)

    # [2] BBSE
    logging.info("BBSE label-shift estimation...")
    C = compute_soft_confusion_matrix(
        model_temp, val_loader, args.num_classes, device, noise_rate=0.0
    )
    target_priors, importance_weights = estimate_target_priors_bbse(
        model=model_temp, target_loader=test_loader, C=C,
        source_priors=source_priors, device=device,
        num_classes=args.num_classes,
    )
    logging.info("Importance weights: %s",
                 [f"{w:.3f}" for w in importance_weights.tolist()])

    # [3] Robust BN recalibration
    logging.info("Robust BN recalibration...")
    model_adapted = recalibrate_bn_robust(model_temp, test_loader, device)

    # ------------------------------------------------------------------
    # Predict: static.pt
    # ------------------------------------------------------------------
    logging.info("Predicting on static.pt (%d samples)...", len(test_dataset))
    static_data = torch.load(args.test_path, weights_only=True)
    static_images = static_data["images"].float()
    static_preds = _predict_tensor(
        model_adapted, static_images, importance_weights, device, args.batch_size
    )

    # ------------------------------------------------------------------
    # Predict: test_suite_public.pt (24 scenarios)
    # ------------------------------------------------------------------
    logging.info("Loading test suite from %s", args.test_suite_path)
    test_suite = torch.load(args.test_suite_path, weights_only=True)
    scenario_keys = sorted(test_suite.keys())
    logging.info("Found %d scenarios: %s", len(scenario_keys), scenario_keys)

    scenario_preds: dict[str, list[int]] = {}
    for key in scenario_keys:
        images = test_suite[key].float()
        preds = _predict_tensor(
            model_adapted, images, importance_weights, device, args.batch_size
        )
        scenario_preds[key] = preds
        logging.info("  %s: %d samples predicted", key, len(preds))

    # ------------------------------------------------------------------
    # Write submission.csv
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "submission.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Category"])

        # static.pt rows
        for idx, label in enumerate(static_preds):
            writer.writerow([f"static_{idx}", label])

        # test_suite scenarios
        for key in scenario_keys:
            for idx, label in enumerate(scenario_preds[key]):
                writer.writerow([f"{key}_{idx}", label])

    total = len(static_preds) + sum(len(v) for v in scenario_preds.values())
    logging.info("Submission CSV written to %s  (%d rows)", csv_path, total)


if __name__ == "__main__":
    main()
