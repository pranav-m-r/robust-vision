"""
train.py – Full training + TTA pipeline
========================================
Phase 1  :  Noise-robust training with GCE loss on source_toxic.pt
Phase 2  :  Confusion-matrix estimation & noise correction (known T)
Phase 3  :  Per-domain BNStats adaptation  +  BBSE prior estimation
Phase 4  :  Submission generation (static + 24 scenarios)

Required data folder structure
--------------------------------
<data-dir>/                  (default: "data", override with --data-dir)
    source_toxic.pt          – noisy labelled training images + labels
    static.pt                – static public-LB test images
    test_suite_public.pt     – 24-scenario private-LB suite
    val_sanity.pt            – clean validation set for sanity checks

CLI usage
----------
    python train.py                        # uses ./data/
    python train.py --data-dir /path/data  # custom data directory
"""

import os
import copy
import argparse
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T

from model_submission import RobustClassifier

# ================================================================
# Configuration
# ================================================================
SEED = 42

NUM_CLASSES = 10
NOISE_RATE = 0.3  # known symmetric label-noise rate
BATCH_SIZE = 256

EPOCHS = 100
LR = 0.1
WEIGHT_DECAY = 5e-4

GCE_Q = 0.7  # GCE truncation parameter
SCE_ALPHA = 1.0  # SCE CE weight
SCE_BETA = 0.5  # SCE RCE weight

BBSE_REG = 2e-2 # BBSE confusion matrix regularisation

TENT_LR = 2e-3 # TENT adaptation learning rate
TENT_STEPS = 10 # TENT adaptation steps

WARMUP_EPOCHS = 5
VAL_RATIO = 0.1  # fraction of source kept for val

DATA_DIR = "data"
WEIGHTS_PATH = "weights.pth"
SUBMISSION_PATH = "submission.csv"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_INTERVAL = 10  # save a checkpoint every N epochs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# Reproducibility
# ================================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================================================================
# Checkpointing helpers
# ================================================================
def _ckpt_path(name: str) -> str:
    """Return full path for a named checkpoint file."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, name)


def save_training_checkpoint(
    epoch, model, optimizer, scheduler, best_acc, best_state, path
):
    """Persist full training state so we can resume later."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_acc,
            "best_state": best_state,
        },
        path,
    )


def load_training_checkpoint(path, model, optimizer, scheduler, device):
    """Restore training state from a checkpoint; returns (start_epoch, best_acc, best_state)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt["best_acc"], ckpt["best_state"]


def save_phase_checkpoint(phase_name: str, payload: dict):
    """Save an arbitrary dict marking a phase as complete."""
    path = _ckpt_path(f"phase_{phase_name}.pt")
    torch.save(payload, path)
    print(f"  [Checkpoint] Phase '{phase_name}' saved → {path}")


def load_phase_checkpoint(phase_name: str):
    """Load a phase checkpoint if it exists, else return None."""
    path = _ckpt_path(f"phase_{phase_name}.pt")
    if os.path.isfile(path):
        print(f"  [Checkpoint] Resuming from phase '{phase_name}' → {path}")
        return torch.load(path, map_location="cpu")
    return None


# ================================================================
# GCE Loss  (Zhang & Sabuncu 2018)
# ================================================================
class GCELoss(nn.Module):
    """L_q(f(x), y) = (1 - f_y(x)^q) / q   (q → 0 gives CE, q → 1 gives MAE)."""

    def __init__(self, q: float = 0.7):
        super().__init__()
        self.q = q

    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        p_y = p.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(1e-7, 1.0)
        loss = (1.0 - p_y**self.q) / self.q
        return loss.mean()


# ================================================================
# SCE Loss  (Wang et al., ICCV 2019)
# ================================================================
class SCELoss(nn.Module):
    """
    Symmetric Cross-Entropy for learning with noisy labels.

    L_SCE = α · CE(p, y) + β · RCE(p, y)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, num_classes: int = 10):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels)
        p = F.softmax(logits, dim=1).clamp(min=1e-7, max=1.0)
        y = F.one_hot(labels, self.num_classes).float()
        y = y.clamp(min=1e-4, max=1.0)
        rce = -(p * torch.log(y)).sum(dim=1).mean()
        return self.alpha * ce + self.beta * rce


# ================================================================
# Dataset with permitted augmentations only
# ================================================================
class AugmentedDataset(torch.utils.data.Dataset):
    """Wraps (images, labels) tensors; applies transforms when train=True."""

    def __init__(self, images, labels, train: bool = True):
        self.images = images
        self.labels = labels
        self.transform = (
            T.Compose(
                [
                    T.RandomCrop(28, padding=4, padding_mode="reflect"),
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(15),
                    T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
                ]
            )
            if train
            else None
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


# ================================================================
# Training helpers
# ================================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for imgs, labs in loader:
        preds = model(imgs.to(device)).argmax(1).cpu()
        correct += (preds == labs).sum().item()
        total += labs.size(0)
    return 100.0 * correct / total


def train_model(model, train_loader, val_loader, device):
    # can swap to GCELoss(q=GCE_Q)
    criterion = SCELoss(alpha=SCE_ALPHA, beta=SCE_BETA, num_classes=NUM_CLASSES)
    optimizer = optim.SGD(
        model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY
    )
    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[WARMUP_EPOCHS]
    )

    best_acc, best_state = 0.0, None
    start_epoch = 0

    # --- attempt to resume from latest training checkpoint ---
    ckpt_file = _ckpt_path("train_latest.pt")
    if os.path.isfile(ckpt_file):
        start_epoch, best_acc, best_state = load_training_checkpoint(
            ckpt_file, model, optimizer, scheduler, device
        )
        start_epoch += 1  # we already completed that epoch
        print(
            f"  [Checkpoint] Resuming training from epoch {start_epoch} "
            f"(best_acc={best_acc:.1f}%)"
        )

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = running_correct = running_total = 0

        for imgs, labs in train_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            logits = model(imgs)
            loss = criterion(logits, labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            running_correct += (logits.argmax(1) == labs).sum().item()
            running_total += imgs.size(0)

        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            val_acc = evaluate(model, val_loader, device)
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch+1:3d}/{EPOCHS} | "
                f"LR {lr:.5f} | "
                f"Loss {running_loss/running_total:.4f} | "
                f"Train {100*running_correct/running_total:.1f}% | "
                f"Val(noisy) {val_acc:.1f}%"
            )
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())

        # --- periodic checkpoint ---
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0 or epoch == EPOCHS - 1:
            save_training_checkpoint(
                epoch,
                model,
                optimizer,
                scheduler,
                best_acc,
                best_state,
                ckpt_file,
            )
            print(f"  [Checkpoint] Epoch {epoch+1} saved → {ckpt_file}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ================================================================
# Phase 2 – Confusion matrix estimation + noise correction
# ================================================================
@torch.no_grad()
def compute_noisy_confusion_matrix(model, loader, device, K=NUM_CLASSES):
    """C_noisy[noisy_label j, predicted k] = P(pred=k | noisy_label=j)."""
    C = torch.zeros(K, K)
    model.eval()
    for imgs, labs in loader:
        preds = model(imgs.to(device)).argmax(1).cpu()
        for j, k in zip(labs, preds):
            C[j.item(), k.item()] += 1
    return C / C.sum(dim=1, keepdim=True).clamp(min=1)


def noise_transition_matrix(eta: float, K: int = NUM_CLASSES):
    """Symmetric noise: T[i,j] = P(noisy=j | true=i)."""
    T = torch.full((K, K), eta / (K - 1))
    T.fill_diagonal_(1.0 - eta)
    return T


def correct_confusion_matrix(C_noisy, T):
    """
    Assuming uniform source prior:
        C_noisy = T^T @ C_true   (for symmetric T, T^T = T)
    => C_true  = T^{-1} @ C_noisy
    """
    C_true = torch.inverse(T) @ C_noisy
    C_true = C_true.clamp(min=1e-6)
    C_true = C_true / C_true.sum(dim=1, keepdim=True)
    return C_true


# ================================================================
# Phase 3 – BNStats adaptation
# ================================================================
def adapt_bn(model, images, device, batch_size=256):
    """
    Reset BN running stats and re-estimate them on *target* data
    using cumulative moving average (momentum=None).
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()
            m.momentum = None  # cumulative average

    model.train()
    loader = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch,) in loader:
            model(batch.to(device))
    model.eval()


def tent_adapt(model, target_images, steps=TENT_STEPS, lr=TENT_LR):
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
            m.train()  # use batch stats
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)
            bn_params += [m.weight, m.bias]

    optimizer = torch.optim.Adam(bn_params, lr=lr)
    loader = DataLoader(
        TensorDataset(target_images), batch_size=BATCH_SIZE, shuffle=True
    )

    for step in range(steps):
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            logits = adapted(batch)
            probs = F.softmax(logits, dim=1)
            # Shannon entropy
            entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=1).mean()

            optimizer.zero_grad()
            entropy.backward()
            optimizer.step()

    adapted.eval()
    return adapted


# ================================================================
# Phase 4 – BBSE  (Black-Box Shift Estimation – Lipton et al. 2018)
# ================================================================
@torch.no_grad()
def estimate_target_priors(
    model, C_true, images, device, batch_size=256, K=NUM_CLASSES
):
    """
    mu_target = C_true^T @ p_target
    => p_target = (C_true^T)^{-1} @ mu_target
    """
    model.eval()
    all_preds = []
    loader = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=False)
    for (batch,) in loader:
        all_preds.append(model(batch.to(device)).argmax(1).cpu())
    all_preds = torch.cat(all_preds)

    # empirical prediction distribution on target
    mu = torch.zeros(K)
    for k in range(K):
        mu[k] = (all_preds == k).float().mean()

    # BBSE inversion (regularise to avoid singular matrix)
    C_T = C_true.T.clone()
    C_T += BBSE_REG * torch.eye(K)  # stronger reg to stabilise inversion
    p_target = torch.inverse(C_T) @ mu
    p_target = p_target.clamp(min=0.02, max=0.40)  # prevent runaway class priors
    p_target = p_target / p_target.sum()
    return p_target


@torch.no_grad()
def predict_with_prior(model, images, device, p_target, batch_size=256, K=NUM_CLASSES):
    """
    Label-shift correction:
        adjusted_logits = logits + log(p_target / p_source)
    with p_source = 1/K  (uniform source prior).
    """
    p_source = torch.ones(K) / K
    log_ratio = torch.log(p_target / p_source).to(device)

    model.eval()
    all_preds = []
    loader = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=False)
    for (batch,) in loader:
        logits = model(batch.to(device))
        adjusted = logits + log_ratio.unsqueeze(0)
        all_preds.append(adjusted.argmax(1).cpu())
    return torch.cat(all_preds)


# ================================================================
# Submission generation
# ================================================================
def generate_submission(
    model, C_true, base_state, device, static_path, suite_path, out_path=SUBMISSION_PATH
):
    results = []

    # ---- static set (public LB) ----
    print("\n[Submission] Static set ...")
    static_imgs = torch.load(static_path, map_location="cpu")["images"]

    model.load_state_dict(copy.deepcopy(base_state))
    model.to(device)
    adapt_bn(model, static_imgs, device)

    p_static = estimate_target_priors(model, C_true, static_imgs, device)
    preds = predict_with_prior(model, static_imgs, device, p_static)
    print(f"  priors: {p_static.numpy().round(4)}")
    for i, p in enumerate(preds):
        results.append({"ID": f"static_{i}", "Category": int(p)})

    # ---- 24 scenario suite (private LB) ----
    print("[Submission] Scenario suite ...")
    suite = torch.load(suite_path, map_location="cpu")
    for skey in sorted(k for k in suite if k.startswith("scenario")):
        scenario_imgs = suite[skey]

        # reload base weights → adapt BN to this scenario
        model.load_state_dict(copy.deepcopy(base_state))
        model.to(device)
        adapt_bn(model, scenario_imgs, device)

        p_scen = estimate_target_priors(model, C_true, scenario_imgs, device)
        preds = predict_with_prior(model, scenario_imgs, device, p_scen)
        for i, p in enumerate(preds):
            results.append({"ID": f"{skey}_{i}", "Category": int(p)})
        print(
            f"  {skey}: {len(scenario_imgs)} imgs, " f"priors={p_scen.numpy().round(3)}"
        )

    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"\n[Submission] Saved {len(df)} rows → {out_path}")
    return df


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Robust classifier training + TTA pipeline"
    )
    parser.add_argument(
        "--data-dir",
        default=DATA_DIR,
        metavar="DIR",
        help="Root folder containing source_toxic.pt, static.pt, "
        "test_suite_public.pt, val_sanity.pt (default: %(default)s)",
    )
    args = parser.parse_args()
    data_dir = args.data_dir

    seed_everything(SEED)
    print(f"Device : {DEVICE}")
    print(f"Data   : {data_dir}")
    print(
        f"Config : EPOCHS={EPOCHS}  LR={LR}  GCE_Q={GCE_Q}  "
        f"BS={BATCH_SIZE}  NOISE={NOISE_RATE}"
    )

    # ---- load source data ----
    print("\n[Phase 0] Loading data ...")
    src = torch.load(os.path.join(data_dir, "source_toxic.pt"), map_location="cpu")
    all_imgs, all_labs = src["images"], src["labels"]
    print(
        f"  Source : {all_imgs.shape}   label dist : "
        f"{torch.bincount(all_labs, minlength=NUM_CLASSES).tolist()}"
    )

    # ---- train / val split ----
    n = len(all_imgs)
    n_val = int(n * VAL_RATIO)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))
    tr_imgs, tr_labs = all_imgs[idx[n_val:]], all_labs[idx[n_val:]]
    vl_imgs, vl_labs = all_imgs[idx[:n_val]], all_labs[idx[:n_val]]
    print(f"  Train : {len(tr_imgs)}   Val : {len(vl_imgs)}")

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        AugmentedDataset(tr_imgs, tr_labs, train=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=g,
    )
    val_loader = DataLoader(
        AugmentedDataset(vl_imgs, vl_labs, train=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ---- Phase 1: train with GCE ----
    phase1 = load_phase_checkpoint("1_training")
    if phase1 is not None:
        print("\n[Phase 1] Skipped (loaded from checkpoint)")
        model = RobustClassifier().to(DEVICE)
        model.load_state_dict(phase1["model_state"])
    else:
        print("\n[Phase 1] Training with GCE loss ...")
        model = RobustClassifier().to(DEVICE)
        model = train_model(model, train_loader, val_loader, DEVICE)
        torch.save(model.state_dict(), WEIGHTS_PATH)
        print(f"  Saved base weights → {WEIGHTS_PATH}")
        save_phase_checkpoint(
            "1_training",
            {
                "model_state": model.state_dict(),
            },
        )

    # ---- sanity check ----
    san = torch.load(os.path.join(data_dir, "val_sanity.pt"), map_location="cpu")
    san_loader = DataLoader(TensorDataset(san["images"], san["labels"]), batch_size=100)
    san_acc = evaluate(model, san_loader, DEVICE)
    print(f"\n[Sanity] val_sanity accuracy : {san_acc:.1f}%")

    # ---- Phase 2: confusion matrix ----
    phase2 = load_phase_checkpoint("2_confusion")
    if phase2 is not None:
        print("\n[Phase 2] Skipped (loaded from checkpoint)")
        C_true = phase2["C_true"]
    else:
        print("\n[Phase 2] Confusion matrix estimation ...")
        C_noisy = compute_noisy_confusion_matrix(model, val_loader, DEVICE)
        T = noise_transition_matrix(NOISE_RATE)
        C_true = correct_confusion_matrix(C_noisy, T)
        print(f"  C_noisy diag : {C_noisy.diag().numpy().round(3)}")
        print(f"  C_true  diag : {C_true.diag().numpy().round(3)}")
        save_phase_checkpoint(
            "2_confusion",
            {
                "C_true": C_true,
            },
        )

    # ---- Phase 3–4: BNStats + BBSE + submission ----
    print("\n[Phase 3] Generating submission (BNStats + BBSE) ...")
    base_state = copy.deepcopy(model.state_dict())
    generate_submission(
        model,
        C_true,
        base_state,
        DEVICE,
        os.path.join(data_dir, "static.pt"),
        os.path.join(data_dir, "test_suite_public.pt"),
    )

    print("\n✓ Done – train.py completed successfully.")


if __name__ == "__main__":
    main()
