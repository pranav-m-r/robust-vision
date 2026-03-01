"""
evaluate_models.py — Test model weights against labeled eval data
==================================================================

Computes Accuracy, Macro-F1, and per-class metrics for each corruption scenario.

Usage:
    python evaluate_models.py                              # uses weights.pth
    python evaluate_models.py --weights weights_v2.pth     # use specific weights
    python evaluate_models.py --no-tta                     # skip TTA (raw model)

Prerequisites:
    Run generate_eval_data.py first to create eval_data/eval_suite.pt
"""

import argparse
import copy
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report
import numpy as np

# ── CHANGE IF MODEL CHANGES ──────────────────────────────────────────
# 1. Update the import below to your new model class.
#    e.g. from new_model import MyNewClassifier
from model_submission import RobustClassifier
from train import adapt_bn, tent_adapt, estimate_target_priors, predict_with_prior
# 2. If your new model has NO BatchNorm2d layers (e.g. uses LayerNorm,
#    GroupNorm, or no normalization), adapt_bn from train.py will
#    silently do nothing. Either add equivalent adaptation for your
#    norm type, or just run with --no-tta.
# 3. The eval data files (eval_clean.pt, eval_suite.pt) are
#    model-independent — no changes needed there.
# ─────────────────────────────────────────────────────────────────────

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
NUM_CLASSES = 10

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# ============================================================
# TTA: adapt_bn imported directly from train.py
# ============================================================


@torch.no_grad()
def predict(model, images):
    model.eval()
    all_preds = []
    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i:i + BATCH_SIZE].to(DEVICE)
        preds = model(batch).argmax(dim=1)
        all_preds.append(preds.cpu())
    return torch.cat(all_preds)


# ============================================================
# Evaluation
# ============================================================
def evaluate_scenario(model_state, images, labels, use_tta=True):
    """Evaluate one scenario. Returns (accuracy, macro_f1, preds)."""
    # ── CHANGE IF MODEL CHANGES ──────────────────────────────────────
    # Replace RobustClassifier() with your new model class.
    # Must accept [B, 1, 28, 28] input and return [B, 10] logits.
    # ─────────────────────────────────────────────────────────────────
    model = RobustClassifier()
    model.load_state_dict(model_state)
    model.to(DEVICE)

    if use_tta:
        adapt_bn(model, images, DEVICE)

    preds = predict(model, images)
    labels_np = labels.numpy()
    preds_np = preds.numpy()

    acc = (preds_np == labels_np).mean()
    macro_f1 = f1_score(labels_np, preds_np, average='macro', zero_division=0)

    return acc, macro_f1, preds_np


def simulate_kaggle_eval(model_state, C_true, log, use_tta=True,
                         eval_dir="eval_data_WA"):
    """
    Mirrors generate_submission() from train.py but evaluates against labeled data.
    Applies adapt_bn + BBSE prior correction (when use_tta=True) per scenario,
    then computes Accuracy and Macro-F1 against ground-truth labels.

    Returns dict of {name: {"accuracy": float, "macro_f1": float, "preds": ndarray}}
    including a "clean" entry for the clean baseline.
    """
    results = {}

    def _run(imgs, labels):
        # ── CHANGE IF MODEL CHANGES ──────────────────────────────────
        # Replace RobustClassifier() with your new model class.
        # ─────────────────────────────────────────────────────────────
        model = RobustClassifier()
        model.load_state_dict(copy.deepcopy(model_state))
        model.to(DEVICE)
        if use_tta:
            adapt_bn(model, imgs, DEVICE)
            model = tent_adapt(model, imgs)  # entropy minimisation on BN affine params
            p_target = estimate_target_priors(model, C_true, imgs, DEVICE)
            preds = predict_with_prior(model, imgs, DEVICE, p_target)
        else:
            preds = predict(model, imgs)
        preds_np  = preds.numpy()
        labels_np = labels.numpy()
        acc      = (preds_np == labels_np).mean()
        macro_f1 = f1_score(labels_np, preds_np, average="macro", zero_division=0)
        return acc, macro_f1, preds_np

    # ── clean baseline ──────────────────────────────────────────────
    log("\n--- CLEAN BASELINE ---")
    clean = torch.load(os.path.join(eval_dir, "eval_clean.pt"), weights_only=False)
    acc, f1, preds = _run(clean["images"], clean["labels"])
    results["clean"] = {"accuracy": acc, "macro_f1": f1, "preds": preds}
    log(f"  Accuracy: {acc:.4f}  Macro-F1: {f1:.4f}")

    # ── corruption scenarios ─────────────────────────────────────────
    log(f"\n--- CORRUPTION SCENARIOS ---")
    suite = torch.load(os.path.join(eval_dir, "eval_suite.pt"), weights_only=False)
    for name in sorted(suite.keys()):
        data = suite[name]
        acc, f1, preds = _run(data["images"], data["labels"])
        results[name] = {"accuracy": acc, "macro_f1": f1, "preds": preds}
        log(f"  {name:25s}  Acc={acc:.4f}  F1={f1:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights.pth", help="Path to weights file")
    parser.add_argument("--no-tta", action="store_true", help="Skip TTA and BBSE")
    parser.add_argument("--eval-dir", default="eval_data_WA",
                        help="Directory containing eval_clean.pt and eval_suite.pt")
    args = parser.parse_args()

    # Collect all output lines for both console and file
    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)

    log(f"Device: {DEVICE}")
    log(f"Weights: {args.weights}")
    log(f"TTA+BBSE: {'OFF' if args.no_tta else 'ON'}")
    log(f"Eval dir: {args.eval_dir}")
    log(f"{'='*70}")

    # Load model weights
    state = torch.load(args.weights, map_location="cpu", weights_only=False)

    # Load C_true for BBSE prior correction (from Phase 2 checkpoint)
    phase2_path = os.path.join("checkpoints", "phase_2_confusion.pt")
    if os.path.isfile(phase2_path):
        C_true = torch.load(phase2_path, map_location="cpu", weights_only=False)["C_true"]
        log(f"C_true  : loaded from {phase2_path}")
    else:
        C_true = torch.eye(NUM_CLASSES)
        log(f"C_true  : identity (no phase-2 checkpoint found – BBSE disabled)")

    # Run full submission pipeline against labeled eval data
    all_results = simulate_kaggle_eval(
        state, C_true, log, use_tta=not args.no_tta, eval_dir=args.eval_dir
    )

    # --- Summary over corruption scenarios (exclude clean baseline) ---
    scenario_results = {k: v for k, v in all_results.items() if k != "clean"}
    accs = [r["accuracy"] for r in scenario_results.values()]
    f1s  = [r["macro_f1"]  for r in scenario_results.values()]
    log(f"\n{'='*70}")
    log(f"SUMMARY")
    log(f"{'='*70}")
    log(f"  Mean Accuracy : {np.mean(accs):.4f}  (min={np.min(accs):.4f}, max={np.max(accs):.4f})")
    log(f"  Mean Macro-F1 : {np.mean(f1s):.4f}  (min={np.min(f1s):.4f}, max={np.max(f1s):.4f})")
    log(f"  Worst scenario: {min(scenario_results, key=lambda k: scenario_results[k]['macro_f1'])}")
    log(f"  Best scenario : {max(scenario_results, key=lambda k: scenario_results[k]['macro_f1'])}")

    # --- Detailed report for worst scenario ---
    worst = min(scenario_results, key=lambda k: scenario_results[k]["macro_f1"])
    log(f"\n--- Detailed report for WORST scenario: {worst} ---")
    suite = torch.load(os.path.join(args.eval_dir, "eval_suite.pt"), weights_only=False)
    report = classification_report(
        suite[worst]["labels"].numpy(), all_results[worst]["preds"],
        target_names=CLASS_NAMES, zero_division=0
    )
    log(report)

    # --- Save results to timestamped txt file ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"{timestamp}_eval.txt"
    with open(out_file, "w") as f:
        f.write("\n".join(lines))
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
