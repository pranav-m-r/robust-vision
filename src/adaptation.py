import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class TemperatureScaledModel(nn.Module):
    """
    Wraps a model with a fixed scalar temperature applied to its logits.

    Corrects overconfidence introduced by noisy-source training without
    modifying any weights.  T > 1 softens predictions; T < 1 sharpens them.

    Reference: Guo et al., "On Calibration of Modern Neural Networks",
    ICML 2017. https://arxiv.org/abs/1706.04599
    """

    def __init__(self, model: nn.Module, temperature: float):
        super().__init__()
        self.model = model
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) / self.temperature


def find_optimal_temperature(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    t_min: float = 0.1,
    t_max: float = 10.0,
    steps: int = 200,
) -> float:
    """
    Grid-search for the temperature T that minimises NLL on the validation set.

    T > 1 → softer predictions (fixes overconfident model).
    Uses only the clean labeled validation set; no target labels are needed.

    Reference: Guo et al., ICML 2017. https://arxiv.org/abs/1706.04599
    """
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            all_logits.append(model(images.to(device)).cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    best_T, best_nll = 1.0, float("inf")
    for T in torch.linspace(t_min, t_max, steps):
        T = T.item()
        nll = F.cross_entropy(all_logits / T, all_labels).item()
        if nll < best_nll:
            best_nll, best_T = nll, T

    logging.info(f"    Optimal T={best_T:.3f}  (NLL={best_nll:.4f})")
    return best_T


def recalibrate_bn_robust(
    model: nn.Module,
    target_loader: DataLoader,
    device: torch.device,
) -> nn.Module:
    """
    Robust batch-norm recalibration via median-of-batch-means.

    Runs the model in train() mode so each forward pass normalises activations
    with actual batch statistics.  Running stats are then set to the median
    across batches, which has a 50 % breakdown point and is therefore robust
    to batches dominated by impulse/corruption noise.

    Reference: Schneider et al., "Improving robustness against common
    corruptions by covariate shift adaptation", NeurIPS 2020.
    https://arxiv.org/abs/2006.16971
    """
    model.train()
    model.requires_grad_(False)

    bn_modules = [
        m
        for m in model.modules()
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
    ]

    # momentum=0 keeps running stats frozen during forward passes so we can
    # set them manually to the median afterwards.
    for module in bn_modules:
        module.momentum = 0

    all_means: list[list[torch.Tensor]] = [[] for _ in bn_modules]
    all_vars: list[list[torch.Tensor]] = [[] for _ in bn_modules]
    hooks = []

    def make_hook(idx: int):
        def hook(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
            x = input[0].detach()
            if x.dim() == 4:  # Conv BN: [B, C, H, W]
                m = x.mean(dim=[0, 2, 3])
                v = x.var(dim=[0, 2, 3], unbiased=False)
            else:  # Linear BN: [B, C]
                m = x.mean(dim=0)
                v = x.var(dim=0, unbiased=False)
            all_means[idx].append(m.cpu())
            all_vars[idx].append(v.cpu())

        return hook

    for i, module in enumerate(bn_modules):
        hooks.append(module.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        for batch_data in target_loader:
            images = batch_data[0] if isinstance(batch_data, (tuple, list)) else batch_data
            model(images.to(device))

    for h in hooks:
        h.remove()

    with torch.no_grad():
        for i, module in enumerate(bn_modules):
            module.running_mean.copy_(
                torch.stack(all_means[i]).median(dim=0).values.to(device)
            )
            module.running_var.copy_(
                torch.stack(all_vars[i]).median(dim=0).values.clamp(min=1e-8).to(device)
            )

    model.eval()
    return model


def compute_soft_confusion_matrix(
    model: nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute the soft confusion matrix C where:
        C[i, j] = E[ p_θ(i | x) | y = j ]

    Rows correspond to predicted class, columns to true class.
    Returns a tensor of shape (num_classes, num_classes).
    """
    model.eval()

    class_prob_sums = torch.zeros(num_classes, num_classes, device=device)
    class_counts = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = F.softmax(logits, dim=1)  # (B, num_classes)

            for i in range(images.size(0)):
                true_class = labels[i]
                class_prob_sums[:, true_class] += probs[i]
                class_counts[true_class] += 1

    for j in range(num_classes):
        if class_counts[j] > 0:
            class_prob_sums[:, j] /= class_counts[j]

    return class_prob_sums


def get_source_priors(dataset: Dataset, num_classes: int = 10) -> torch.Tensor:
    """
    Compute the source class distribution π_s from a labeled dataset.

    Returns a probability vector of shape (num_classes,).
    """
    labels = dataset.labels
    priors = torch.zeros(num_classes)
    for c in range(num_classes):
        priors[c] = (labels == c).sum().item()
    return priors / labels.size(0)


def estimate_target_priors_bbse(
    model: nn.Module,
    target_loader: DataLoader,
    C: torch.Tensor,
    source_priors: torch.Tensor,
    device: torch.device,
    num_classes: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Black Box Shift Estimation (BBSE) for label shift correction.

    Uses soft predicted class probabilities (averaged over the target set)
    rather than hard argmax for a more stable μ_t estimate on noisy data.

    Solves the linear system:  C · π_t = μ_t  →  π_t = C⁻¹ · μ_t

    Returns:
        target_priors: estimated target class distribution π_t, shape (num_classes,)
        importance_weights: per-class ratio w_c = π_t[c] / π_s[c], shape (num_classes,)
    """
    model.eval()

    mu_t = torch.zeros(num_classes, device=device)
    total_samples = 0

    with torch.no_grad():
        for batch_data in target_loader:
            images = batch_data[0] if isinstance(batch_data, (tuple, list)) else batch_data
            images = images.to(device)

            probs = F.softmax(model(images), dim=1)  # (B, num_classes)
            mu_t += probs.sum(dim=0)
            total_samples += images.size(0)

    mu_t = mu_t / total_samples

    C_inv = torch.linalg.pinv(C)
    target_priors = torch.matmul(C_inv, mu_t)
    target_priors = torch.clamp(target_priors, min=1e-8)
    target_priors = target_priors / target_priors.sum()

    importance_weights = target_priors / (source_priors + 1e-8)
    return target_priors, importance_weights


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Shannon entropy of the softmax distribution.
    Used for TENT-style test-time entropy minimisation.
    """
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def evaluate_with_prior_correction(
    model: nn.Module,
    loader: DataLoader,
    importance_weights: torch.Tensor,
    device: torch.device,
) -> float:
    """
    Evaluate accuracy with BBSE prior correction applied at inference.

    The corrected posterior is:
        p_corrected(y | x) ∝ w_y · p_model(y | x)

    Returns top-1 accuracy (%).
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_data in loader:
            images = batch_data[0] if isinstance(batch_data, (tuple, list)) else batch_data
            labels = batch_data[1] if isinstance(batch_data, (tuple, list)) else None

            images = images.to(device)
            probs = F.softmax(model(images), dim=1)
            corrected = probs * importance_weights.unsqueeze(0)
            preds = corrected.argmax(dim=1)

            if labels is not None:
                labels = labels.to(device)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

    return 100.0 * correct / total
