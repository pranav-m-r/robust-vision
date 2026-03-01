"""
train.py — Full Training & Adaptation Pipeline
================================================
Run:  python train.py

Produces:  weights.pth   (trained model weights)
           submission.csv (predictions for static.pt + 24 test scenarios)

Pipeline overview
-----------------
Phase 1  Robust Training
         Train on source_toxic.pt (30 % label noise) using:
           • Symmetric Cross-Entropy (SCE) loss — tolerates symmetric noise
           • Mixup — regularisation that smooths decision boundaries
           • Permitted augmentations: RandomCrop, HFlip, Rotation, Erasing
         Select best checkpoint via val_sanity.pt (100 clean samples).

Phase 2  Target Distribution Estimation
         Run the trained model on the unlabeled static.pt to estimate the
         target class distribution.  This is used for prior-adjusted
         inference (log-prior shift on logits).

Phase 3  Test-Time Adaptation (TTA)
         For each evaluation set (static + 24 scenarios):
           a) BN-stats alignment — reset BN running stats and recompute
              them from the target batch (one forward pass).
           b) TENT — fine-tune only BN affine parameters (γ, β) by
              minimising prediction entropy for a few steps.
         Then predict with prior-adjusted logits.

Constraints honoured
--------------------
✓ Random weight init only (Kaiming) — no pretrained models.
✓ No external data — trains strictly on source_toxic.pt.
✓ Augmentations are white-listed only (no noise/blur/AugMix/AutoAugment).
✓ Model architecture defined in model_submission.py → RobustClassifier.
"""

import os
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision.transforms as T
import pandas as pd

from model_submission import RobustClassifier

# ============================================================
# Configuration
# ============================================================
DATA_DIR     = "data"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES  = 10

# Phase 1 — training
EPOCHS       = 60
BATCH_SIZE   = 128
LR           = 1e-3
WEIGHT_DECAY = 5e-4
SCE_ALPHA    = 1.0       # weight for standard CE in SCE
SCE_BETA     = 0.5       # weight for reverse CE in SCE
MIXUP_ALPHA  = 0.3       # Beta distribution parameter for Mixup
GRAD_CLIP    = 1.0       # max gradient norm

# Phase 3 — TTA
BN_ADAPT_PASSES = 1      # forward passes for BN stat reset
TENT_STEPS      = 3      # entropy-minimisation steps
TENT_LR         = 5e-4   # learning rate for TENT


# ============================================================
# 1.  LOSS: Symmetric Cross-Entropy (SCE)
# ============================================================
class SCELoss(nn.Module):
    """
    Symmetric Cross-Entropy for learning with noisy labels.

    L_SCE = α · CE(p, y) + β · RCE(p, y)

    CE  = standard cross-entropy (robust to easy noise).
    RCE = reverse cross-entropy — bounds the contribution of
          mislabelled samples because log(one-hot) is clamped.

    Reference:
        Wang et al., "Symmetric Cross Entropy for Robust Learning
        with Noisy Labels", ICCV 2019.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5,
                 num_classes: int = 10):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.num_classes = num_classes

    def forward(self, logits, labels):
        # --- standard CE ---
        ce = F.cross_entropy(logits, labels)

        # --- reverse CE: -Σ p_i · log(y_i) ---
        p = F.softmax(logits, dim=1).clamp(min=1e-7, max=1.0)
        y = F.one_hot(labels, self.num_classes).float()
        y = y.clamp(min=1e-4, max=1.0)          # avoid log(0)
        rce = -(p * torch.log(y)).sum(dim=1).mean()

        return self.alpha * ce + self.beta * rce


# ============================================================
# 2.  DATASET with per-sample augmentations
# ============================================================
class AugmentedDataset(Dataset):
    """Wraps tensors and applies torchvision transforms per sample."""

    def __init__(self, images, labels, transform=None):
        self.images    = images       # [N, 1, 28, 28]
        self.labels    = labels       # [N]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img   = self.images[idx]      # [1, 28, 28]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_train_transform():
    """
    White-listed augmentations only:
      • RandomCrop 28 with 4-pixel padding (translation invariance)
      • RandomHorizontalFlip          (mirror invariance)
      • RandomRotation ±15°           (rotation invariance)
      • RandomErasing p=0.2           (occlusion invariance, Cutout-style)
    """
    return T.Compose([
        T.RandomCrop(28, padding=4),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


# ============================================================
# 3.  MIXUP helpers
# ============================================================
def mixup_data(x, y, alpha=0.3):
    """
    Apply Mixup: x̃ = λ·x_i + (1-λ)·x_j,  use both labels for loss.
    Mixup is naturally robust to label noise because it softens targets.
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss: λ·L(pred, y_a) + (1-λ)·L(pred, y_b)."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# 4.  EVALUATION helper
# ============================================================
@torch.no_grad()
def evaluate(model, images, labels):
    """Return accuracy on a small tensor set."""
    model.eval()
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)
    preds  = model(images).argmax(dim=1)
    return (preds == labels).float().mean().item()


# ============================================================
# 5.  PHASE 1 — Robust Training
# ============================================================
def train_robust(model, train_images, train_labels, val_images, val_labels):
    """
    Train the model on noisy source data.

    Key ingredients for noise robustness:
      • SCE loss (bounds gradient from mislabelled samples)
      • Mixup    (interpolated targets reduce memorisation of noise)
      • Weight decay + gradient clipping (stability)
      • Cosine-annealing LR schedule (smooth convergence)
      • Best-checkpoint selection on clean val set
    """
    model.to(DEVICE)
    criterion = SCELoss(alpha=SCE_ALPHA, beta=SCE_BETA,
                        num_classes=NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    dataset = AugmentedDataset(train_images, train_labels,
                               transform=get_train_transform())
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         shuffle=True, drop_last=True,
                         num_workers=2, pin_memory=True)

    best_acc   = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Mixup
            images, y_a, y_b, lam = mixup_data(images, labels, MIXUP_ALPHA)

            logits = model(images)
            loss   = mixup_criterion(criterion, logits, y_a, y_b, lam)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        # ---- checkpoint on clean val ----
        val_acc  = evaluate(model, val_images, val_labels)
        avg_loss = running_loss / len(loader)

        if val_acc >= best_acc:
            best_acc   = val_acc
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:3d}/{EPOCHS}  "
                  f"loss={avg_loss:.4f}  val_acc={val_acc:.2%}  "
                  f"best={best_acc:.2%}  lr={lr_now:.6f}")

    model.load_state_dict(best_state)
    print(f"\n  ✓ Phase 1 complete — best val accuracy: {best_acc:.2%}")
    return model


# ============================================================
# 6.  PHASE 2 — Target Distribution Estimation
# ============================================================
@torch.no_grad()
def estimate_target_distribution(model, target_images):
    """
    Estimate class prior of the unlabeled target domain.

    Strategy: use model predictions as a proxy for the true
    distribution.  Smooth slightly toward uniform to avoid
    extreme adjustments on rare classes.
    """
    model.eval()
    target_images = target_images.to(DEVICE)

    # predict in batches to be safe with memory
    all_preds = []
    for i in range(0, len(target_images), BATCH_SIZE):
        batch = target_images[i:i + BATCH_SIZE]
        preds = model(batch).argmax(dim=1)
        all_preds.append(preds)
    all_preds = torch.cat(all_preds)

    raw_dist = torch.bincount(all_preds, minlength=NUM_CLASSES).float()
    raw_dist = raw_dist / raw_dist.sum()

    # smooth: 90 % predicted + 10 % uniform (avoids zero-class issues)
    uniform  = torch.ones(NUM_CLASSES) / NUM_CLASSES
    smoothed = 0.9 * raw_dist.cpu() + 0.1 * uniform

    print(f"  Raw predicted distribution : {raw_dist.cpu().numpy().round(3)}")
    print(f"  Smoothed target distribution: {smoothed.numpy().round(3)}")
    return smoothed


# ============================================================
# 7.  PHASE 3 — Test-Time Adaptation
# ============================================================
def adapt_bn_stats(model, target_images, n_passes=1):
    """
    BN-statistics alignment.

    Reset BN running statistics and recompute them from the target
    data.  This aligns the feature normalisation to the target
    domain's data distribution (covariate shift correction).

    Uses cumulative moving average (momentum=None) so one full
    pass is sufficient.
    """
    adapted = copy.deepcopy(model)
    adapted.to(DEVICE)

    # Reset BN running stats → will recompute from target
    for m in adapted.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None        # cumulative moving average

    adapted.train()                   # BN uses batch stats & updates running stats
    loader = DataLoader(TensorDataset(target_images),
                        batch_size=BATCH_SIZE, shuffle=True)

    with torch.no_grad():
        for _ in range(n_passes):
            for (batch,) in loader:
                adapted(batch.to(DEVICE))

    adapted.eval()
    return adapted


def tent_adapt(model, target_images, steps=3, lr=5e-4):
    """
    TENT: Test-time Entropy minimisation.

    Fine-tune only BN affine parameters (γ, β) to minimise the
    entropy of model predictions on the target batch.  This
    encourages the model to make confident (low-entropy) predictions,
    which empirically improves accuracy on shifted domains.

    Reference:
        Wang et al., "Tent: Fully Test-Time Adaptation by Entropy
        Minimization", ICLR 2021.
    """
    adapted = copy.deepcopy(model)
    adapted.to(DEVICE)

    # Freeze everything
    for p in adapted.parameters():
        p.requires_grad_(False)

    # Un-freeze + train-mode only for BN layers
    bn_params = []
    for m in adapted.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()                     # use batch stats
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)
            bn_params += [m.weight, m.bias]

    optimizer = torch.optim.Adam(bn_params, lr=lr)
    loader    = DataLoader(TensorDataset(target_images),
                           batch_size=BATCH_SIZE, shuffle=True)

    for step in range(steps):
        for (batch,) in loader:
            batch  = batch.to(DEVICE)
            logits = adapted(batch)
            probs  = F.softmax(logits, dim=1)
            # Shannon entropy
            entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=1).mean()

            optimizer.zero_grad()
            entropy.backward()
            optimizer.step()

    adapted.eval()
    return adapted


# ============================================================
# 8.  INFERENCE with prior adjustment
# ============================================================
@torch.no_grad()
def predict_with_prior(model, images, target_dist=None):
    """
    Predict class labels, optionally adjusting logits for label shift.

    Prior adjustment:
        adjusted_logit_c = logit_c + log(P_target(c) / P_source(c))
    Source is balanced → P_source = 1/C for all classes.
    """
    model.eval()
    images = images.to(DEVICE)

    all_preds = []
    for i in range(0, len(images), BATCH_SIZE):
        batch  = images[i:i + BATCH_SIZE]
        logits = model(batch)

        if target_dist is not None:
            source_dist = torch.ones(NUM_CLASSES) / NUM_CLASSES
            log_prior   = torch.log(target_dist / source_dist).to(DEVICE)
            logits      = logits + log_prior

        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())

    return torch.cat(all_preds)


# ============================================================
# 9.  ADAPTATION + PREDICTION for one dataset
# ============================================================
def adapt_and_predict(base_state, images, target_dist=None):
    """
    Full TTA pipeline for a single target set:
      1. Load base weights
      2. Adapt BN running stats
      3. TENT entropy minimisation
      4. Predict with prior adjustment
    """
    model = RobustClassifier()
    model.load_state_dict(base_state)

    # Step 1: BN-stats alignment
    model = adapt_bn_stats(model, images, n_passes=BN_ADAPT_PASSES)

    # Step 2: TENT
    model = tent_adapt(model, images, steps=TENT_STEPS, lr=TENT_LR)

    # Step 3: predict
    preds = predict_with_prior(model, images, target_dist)
    return preds


# ============================================================
# 10.  SUBMISSION GENERATION
# ============================================================
def generate_submission(base_state, target_dist, static_path, suite_path):
    """
    Generate submission.csv with TTA applied independently per dataset.

    Each dataset gets its own BN adaptation — this is critical because
    different scenarios have different corruption types/severities.
    """
    results = []

    # --- static set ---
    print("  Predicting static set …")
    static = torch.load(static_path, weights_only=False)
    static_images = static["images"]
    preds = adapt_and_predict(base_state, static_images, target_dist)
    for i, p in enumerate(preds):
        results.append({"ID": f"static_{i}", "Category": int(p)})
    print(f"    → {len(static_images)} predictions")

    # --- 24 scenarios ---
    print("  Predicting test scenarios …")
    suite = torch.load(suite_path, weights_only=False)
    scenario_keys = sorted(k for k in suite if k.startswith("scenario"))

    for skey in scenario_keys:
        scenario_images = suite[skey]
        preds = adapt_and_predict(base_state, scenario_images, target_dist)
        for i, p in enumerate(preds):
            results.append({"ID": f"{skey}_{i}", "Category": int(p)})
        print(f"    {skey}: {len(scenario_images)} samples")

    df = pd.DataFrame(results)
    df.to_csv("submission.csv", index=False)
    print(f"\n  ✓ submission.csv saved — {len(results)} total predictions")


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Device: {DEVICE}")
    print(f"{'=' * 60}")

    # ---- load data ----
    source = torch.load(os.path.join(DATA_DIR, "source_toxic.pt"),
                        weights_only=False)
    val    = torch.load(os.path.join(DATA_DIR, "val_sanity.pt"),
                        weights_only=False)

    train_images = source["images"]   # [60000, 1, 28, 28]
    train_labels = source["labels"]   # [60000]
    val_images   = val["images"]      # [100, 1, 28, 28]
    val_labels   = val["labels"]      # [100]

    print(f"Training samples : {len(train_images):,} (30 % noisy labels)")
    print(f"Validation samples: {len(val_images):,} (clean)")

    # ==========================================================
    # PHASE 1 — Robust Training
    # ==========================================================
    print(f"\n{'=' * 60}")
    print("PHASE 1: ROBUST TRAINING  (Noise Decontamination)")
    print(f"{'=' * 60}")
    print(f"  Loss      : SCE (α={SCE_ALPHA}, β={SCE_BETA})")
    print(f"  Mixup     : α={MIXUP_ALPHA}")
    print(f"  Epochs    : {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  LR        : {LR} → cosine annealing")
    print()

    model = RobustClassifier()
    model = train_robust(model, train_images, train_labels,
                         val_images, val_labels)

    # Save base weights
    torch.save(model.state_dict(), "weights.pth")
    print("  Base weights saved → weights.pth")

    # ==========================================================
    # PHASE 2 — Distribution Estimation
    # ==========================================================
    print(f"\n{'=' * 60}")
    print("PHASE 2: TARGET DISTRIBUTION ESTIMATION")
    print(f"{'=' * 60}")

    static = torch.load(os.path.join(DATA_DIR, "static.pt"),
                        weights_only=False)
    target_dist = estimate_target_distribution(model, static["images"])

    # ==========================================================
    # PHASE 3 — TTA  +  Submission
    # ==========================================================
    print(f"\n{'=' * 60}")
    print("PHASE 3: TEST-TIME ADAPTATION  &  SUBMISSION")
    print(f"{'=' * 60}")
    print(f"  BN-stats passes : {BN_ADAPT_PASSES}")
    print(f"  TENT steps      : {TENT_STEPS}")
    print(f"  TENT lr         : {TENT_LR}")
    print()

    base_state = copy.deepcopy(model.state_dict())
    generate_submission(
        base_state, target_dist,
        os.path.join(DATA_DIR, "static.pt"),
        os.path.join(DATA_DIR, "test_suite_public.pt"),
    )

    # ==========================================================
    # Final sanity check
    # ==========================================================
    print(f"\n{'=' * 60}")
    print("FINAL SANITY CHECK")
    print(f"{'=' * 60}")
    final_acc = evaluate(model, val_images, val_labels)
    print(f"  Val accuracy (base model): {final_acc:.2%}")
    print(f"\n  Done!  Files produced:")
    print(f"    • weights.pth")
    print(f"    • submission.csv")


if __name__ == "__main__":
    main()
