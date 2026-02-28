"""
Generate submission.csv with the required ID format:
  - static_i   for elements in static.pt
  - scenario_XX_i  for elements in test_suite_public.pt (if available)

Applies the full adaptation pipeline (temp scaling → BN recal → TENT →
prior correction) before generating predictions.
"""

import copy
import os
import sys

import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from src.model import get_model
from src.adaptation import (
    TemperatureScaledModel,
    recalibrate_bn_robust,
    tent_adapt,
    estimate_target_priors_bbse,
)

# ── Paths ────────────────────────────────────────────────────────────────
MODEL_PATH = "results/train/2026-02-28_22-20-05/base_model.pt"
STATIC_PATH = "data/static.pt"
VAL_PATH = "data/val_sanity.pt"
SUITE_PATH = "data/test_suite_public.pt"    # 24 scenarios for private LB

# ── Hyperparameters (must match eval pipeline config) ────────────────────
NUM_CLASSES = 10
BATCH_SIZE = 64
BN_ALPHA = 0.1
TENT_LR = 5e-4
TENT_STEPS = 2
ENTROPY_MARGIN = 0.5
TEMP_T_MIN = 0.5
TEMP_T_MAX = 5.0
TEMP_STEPS = 200

# ── On/Off switches (match config.yaml → adaptation) ────────────────────
ENABLE_TEMP_SCALING     = True   # [1] Temperature scaling
ENABLE_BN_RECAL         = True   # [2] Alpha-blended BN recalibration
ENABLE_TENT             = True   # [3] TENT / SAR entropy minimisation
ENABLE_PRIOR_ESTIMATION = True   # [4] Label shift prior estimation
ENABLE_PRIOR_CORRECTION = True   # [5] Prior-corrected inference


def build_loader(images: torch.Tensor, batch_size: int = BATCH_SIZE) -> DataLoader:
    """Build a simple DataLoader from a tensor of images."""
    return DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=False)


def adapt_model(model, static_images, val_images, val_labels, device):
    """
    Run the adaptation pipeline (respecting on/off switches) and return
    (adapted_model, importance_weights).
    """
    from src.adaptation import find_optimal_temperature

    model_current = copy.deepcopy(model)

    # ── [1] Temperature scaling (calibrated on val_sanity) ───────────────
    if ENABLE_TEMP_SCALING:
        val_loader = DataLoader(
            TensorDataset(val_images, val_labels), batch_size=BATCH_SIZE
        )
        T = find_optimal_temperature(
            model_current, val_loader, device,
            t_min=TEMP_T_MIN, t_max=TEMP_T_MAX, steps=TEMP_STEPS,
            noise_rate=0.0,
        )
        model_current = TemperatureScaledModel(
            copy.deepcopy(model_current), temperature=T
        ).to(device)
        print(f"[1] Temperature scaling: T={T:.3f}")
    else:
        print("[1] Temperature scaling: SKIPPED")

    # ── [2] Alpha-blended BN recalibration ───────────────────────────────
    target_loader = build_loader(static_images)
    if ENABLE_BN_RECAL:
        model_current = recalibrate_bn_robust(
            copy.deepcopy(model_current), target_loader, device, alpha=BN_ALPHA
        )
        print(f"[2] BN recalibration: α={BN_ALPHA}")
    else:
        print("[2] BN recalibration: SKIPPED")

    # ── [3] TENT entropy minimisation ────────────────────────────────────
    if ENABLE_TENT:
        model_current = tent_adapt(
            model_current, target_loader, device,
            lr=TENT_LR, steps=TENT_STEPS,
            num_classes=NUM_CLASSES, entropy_margin=ENTROPY_MARGIN,
        )
        print(f"[3] TENT: lr={TENT_LR}, steps={TENT_STEPS}, margin={ENTROPY_MARGIN}")
    else:
        print("[3] TENT: SKIPPED")

    # ── [4] Label shift estimation ───────────────────────────────────────
    importance_weights = torch.ones(NUM_CLASSES, device=device)
    if ENABLE_PRIOR_ESTIMATION:
        source_priors = torch.full(
            (NUM_CLASSES,), 1.0 / NUM_CLASSES, device=device
        )
        _, importance_weights = estimate_target_priors_bbse(
            model=model_current, target_loader=target_loader,
            C=None, source_priors=source_priors,
            device=device, num_classes=NUM_CLASSES,
        )
        print(f"[4] Importance weights: {importance_weights.cpu().tolist()}")
    else:
        print("[4] Prior estimation: SKIPPED (uniform weights)")

    # ── [5] Prior correction flag ────────────────────────────────────────
    if not ENABLE_PRIOR_CORRECTION:
        importance_weights = torch.ones(NUM_CLASSES, device=device)
        print("[5] Prior correction: SKIPPED (uniform weights)")

    return model_current, importance_weights


def predict_with_prior_correction(model, images, importance_weights, device):
    """Return class predictions with prior correction applied."""
    model.eval()
    all_preds = []
    loader = build_loader(images)
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            probs = F.softmax(model(batch), dim=1)
            corrected = probs * importance_weights.unsqueeze(0)
            preds = corrected.argmax(dim=1)
            all_preds.append(preds.cpu())
    return torch.cat(all_preds, dim=0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ───────────────────────────────────────────────────────
    model = get_model(num_classes=NUM_CLASSES).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model from: {MODEL_PATH}")

    # ── Load data ────────────────────────────────────────────────────────
    static = torch.load(STATIC_PATH, weights_only=False)
    static_images = static["images"]   # keep on CPU; moved to device in batches
    print(f"Static set: {static_images.shape}")

    val = torch.load(VAL_PATH, weights_only=False)
    val_images = val["images"]   # keep on CPU; loaders move to device as needed
    val_labels = val["labels"]

    # ── Adapt model ──────────────────────────────────────────────────────
    model_adapted, importance_weights = adapt_model(
        model, static_images, val_images, val_labels, device
    )

    # ── Generate predictions ─────────────────────────────────────────────
    results = []

    # 1. Static set (public LB)
    print("\nPredicting on static set...")
    static_preds = predict_with_prior_correction(
        model_adapted, static_images, importance_weights, device
    )
    for i, p in enumerate(static_preds):
        results.append({"ID": f"static_{i}", "Category": int(p)})
    print(f"  → {len(static_preds)} static predictions")

    # 2. Scenario suite (private LB) — only if file exists
    if os.path.exists(SUITE_PATH):
        print(f"\nLoading scenario suite from {SUITE_PATH}...")
        suite = torch.load(SUITE_PATH, weights_only=False)
        scenario_keys = sorted([k for k in suite.keys() if k.startswith("scenario")])
        print(f"  Found {len(scenario_keys)} scenarios")

        for skey in scenario_keys:
            scenario_images = suite[skey].to(device)
            preds = predict_with_prior_correction(
                model_adapted, scenario_images, importance_weights, device
            )
            for i, p in enumerate(preds):
                results.append({"ID": f"{skey}_{i}", "Category": int(p)})
            print(f"  {skey}: {len(preds)} predictions")
    else:
        print(f"\n⚠ Suite file not found at {SUITE_PATH} — skipping scenarios.")

    # ── Save ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv("submission.csv", index=False)
    print(f"\n✓ Saved submission.csv ({len(df)} rows)")
    print(df.head())
    print("...")
    print(df.tail())


if __name__ == "__main__":
    main()
