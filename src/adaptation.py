import copy
import logging
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# Patrini et al. 2017 (https://arxiv.org/pdf/1609.03683)
# T[i,j] = P(ŷ=j | y=i)
# Symmetric noise: diagonal = 1-ε, off-diagonal = ε/(K-1)
def build_transition_matrix(
    noise_rate: float = 0.30,
    num_classes: int = 10,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    off_diag = noise_rate / (num_classes - 1)
    T = torch.full((num_classes, num_classes), off_diag, device=device)
    T.fill_diagonal_(1.0 - noise_rate)
    return T


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
    noise_rate: float = 0.30,
) -> float:
    """
    Grid-search for the temperature T that minimises NLL on the validation set.
    Labels are forward-corrected via T before NLL is computed.
    Reference: Guo et al., ICML 2017. https://arxiv.org/abs/1706.04599
    """
    model.eval()
    num_classes = None
    all_logits, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            logits = model(images.to(device)).cpu()
            all_logits.append(logits)
            all_labels.append(labels)
            num_classes = logits.shape[1]

    all_logits = torch.cat(all_logits, dim=0)   # (N, K)
    all_labels = torch.cat(all_labels, dim=0)   # (N,)

    # Forward correction: build soft label targets p̃ = T[y] for each sample
    T = build_transition_matrix(noise_rate, num_classes, device=torch.device("cpu"))
    soft_labels = T[all_labels]                 # (N, K)  — row of T for each noisy label

    best_T, best_nll = 1.0, float("inf")
    for temp in torch.linspace(t_min, t_max, steps):
        temp = temp.item()
        log_probs = F.log_softmax(all_logits / temp, dim=1)   # (N, K)
        nll = -(soft_labels * log_probs).sum(dim=1).mean().item()
        if nll < best_nll:
            best_nll, best_T = nll, temp

    logging.info(f"    Optimal T={best_T:.3f}  (NLL={best_nll:.4f})")
    return best_T


def recalibrate_bn_robust(
    model: nn.Module,
    target_loader: DataLoader,
    device: torch.device,
    alpha: float = 0.1,
) -> nn.Module:
    """
    Alpha-blended batch-norm recalibration with median-of-batch-means.

    Instead of fully replacing source BN stats with target stats (which can
    be catastrophic when the corruption changes statistics dramatically),
    we blend:  final = (1-α) * source + α * target_median

    α=0 means no adaptation (keep source stats).
    α=1 means full replacement (original behaviour, risky).
    α=0.1 is a conservative default that partially adapts without destroying
    the learned representations.

    Reference: Schneider et al., NeurIPS 2020. https://arxiv.org/abs/2006.16971
    """
    model.train()
    model.requires_grad_(False)

    bn_modules = [
        m
        for m in model.modules()
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
    ]

    # Save original source BN stats for blending
    source_means = [m.running_mean.clone() for m in bn_modules]
    source_vars = [m.running_var.clone() for m in bn_modules]

    for module in bn_modules:
        module.momentum = 0

    all_means: list[list[torch.Tensor]] = [[] for _ in bn_modules]
    all_vars: list[list[torch.Tensor]] = [[] for _ in bn_modules]
    hooks = []

    def make_hook(idx: int):
        def hook(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
            x = input[0].detach()
            if x.dim() == 4:
                m = x.mean(dim=[0, 2, 3])
                v = x.var(dim=[0, 2, 3], unbiased=False)
            else:
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
            target_mean = torch.stack(all_means[i]).median(dim=0).values.to(device)
            target_var = torch.stack(all_vars[i]).median(dim=0).values.clamp(min=1e-8).to(device)

            # Alpha-blend: partial adaptation
            blended_mean = (1.0 - alpha) * source_means[i] + alpha * target_mean
            blended_var = (1.0 - alpha) * source_vars[i] + alpha * target_var

            module.running_mean.copy_(blended_mean)
            module.running_var.copy_(blended_var)

    logging.info("    BN recal alpha=%.2f (%.0f%% source, %.0f%% target)", alpha, (1-alpha)*100, alpha*100)
    model.eval()
    return model


def compute_soft_confusion_matrix(
    model: nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    device: torch.device,
    noise_rate: float = 0.30,
) -> torch.Tensor:
    """
    Forward correction (Patrini et al. 2017, Eq. 3):
        p̃(y | x) = T · p(y | x)

    where T is the noise transition matrix.  The confusion matrix C used by
    BBSE is then estimated from these noise-corrected probabilities.

    C[i, j] = E[ p̃_θ(i | x) | y = j ]
    """
    model.eval()

    T = build_transition_matrix(noise_rate, num_classes, device)

    class_prob_sums = torch.zeros(num_classes, num_classes, device=device)
    class_counts = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            probs = F.softmax(model(images), dim=1)          # (B, K)
            corrected = (T @ probs.T).T                      # (B, K)  forward correction

            for i in range(images.size(0)):
                true_class = labels[i]
                class_prob_sums[:, true_class] += corrected[i]
                class_counts[true_class] += 1

    for j in range(num_classes):
        if class_counts[j] > 0:
            class_prob_sums[:, j] /= class_counts[j]

    return class_prob_sums


def compute_confusion_matrix_from_noisy_source(
    model: nn.Module,
    source_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    noise_rate: float = 0.30,
) -> torch.Tensor:
    """
    Estimate the clean confusion matrix C for BBSE from a noisy-labelled
    source dataset (e.g. source_toxic.pt with 30% symmetric noise).

    Steps
    -----
    1. Compute the *noisy* confusion matrix:
         C_noisy[i, j] = E[ p_θ(i | x) | ỹ = j ]
       where ỹ are the noisy labels in source_toxic.
    2. Correct using the known noise transition matrix T:
         C_clean = C_noisy @ T⁻¹
       because  C_noisy = C_clean @ T  for symmetric noise on balanced classes.

    Reference: Lipton et al., ICML 2018 (BBSE);
               Patrini et al., CVPR 2017 (transition matrix).
    """
    model.eval()

    class_prob_sums = torch.zeros(num_classes, num_classes, device=device)
    class_counts = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for batch_data in source_loader:
            images = batch_data[0].to(device)
            labels = batch_data[1].to(device)  # noisy labels

            probs = F.softmax(model(images), dim=1)  # (B, K)

            for i in range(images.size(0)):
                noisy_class = labels[i]
                class_prob_sums[:, noisy_class] += probs[i]
                class_counts[noisy_class] += 1

    # Normalise to get C_noisy
    for j in range(num_classes):
        if class_counts[j] > 0:
            class_prob_sums[:, j] /= class_counts[j]

    C_noisy = class_prob_sums  # (K, K)

    # Denoise: C_clean = C_noisy @ T^{-1}
    T = build_transition_matrix(noise_rate, num_classes, device)
    T_reg = T + 1e-4 * torch.eye(num_classes, device=device)
    try:
        T_inv = torch.linalg.inv(T_reg)
    except torch.linalg.LinAlgError:
        T_inv = torch.linalg.pinv(T)

    C_clean = C_noisy @ T_inv

    # Clamp and re-normalise columns to be valid distributions
    C_clean = torch.clamp(C_clean, min=1e-8)
    C_clean = C_clean / C_clean.sum(dim=0, keepdim=True)

    logging.info("    C_noisy diagonal: %s", [f"{C_noisy[i,i]:.3f}" for i in range(num_classes)])
    logging.info("    C_clean diagonal: %s", [f"{C_clean[i,i]:.3f}" for i in range(num_classes)])

    return C_clean


def get_source_priors(dataset: Dataset, num_classes: int = 10) -> torch.Tensor:
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
    Estimate target class priors directly from model's average softmax
    predictions (μ_t) with Laplace smoothing.

    This is much more stable than EM, which tends to collapse to a degenerate
    2-3 class solution when the model is overconfident on corrupted data.

    Method:
      1. Compute μ_t = (1/N) Σ_x softmax(f(x))  on the target domain
      2. Apply Laplace smoothing: π_t ∝ μ_t + ε  (ε = 1/K)
      3. Clamp importance weights to [0.2, 5.0] to prevent extreme corrections

    The softmax average is a consistent estimator of the label distribution
    when the classifier is well-calibrated (which our temperature scaling
    ensures).

    Reference: Alexandari et al., ICML 2020 — direct label frequency estimation
    """
    model.eval()

    # Collect all softmax predictions on the target domain
    all_probs = []
    with torch.no_grad():
        for batch_data in target_loader:
            images = batch_data[0] if isinstance(batch_data, (tuple, list)) else batch_data
            images = images.to(device)
            probs = F.softmax(model(images), dim=1)
            all_probs.append(probs)
    all_probs = torch.cat(all_probs, dim=0)  # (N, K)

    # Direct μ_t estimation: average softmax predictions
    mu_t = all_probs.mean(dim=0)  # (K,)

    # Laplace smoothing to prevent any class from having zero prior
    eps = 1.0 / num_classes
    pi_t = mu_t + eps
    pi_t = pi_t / pi_t.sum()

    pi_s = source_priors.to(device)

    # Importance weights with conservative clamping
    importance_weights = pi_t / (pi_s + 1e-10)
    importance_weights = importance_weights.clamp(min=0.2, max=5.0)

    # Log entropy of estimated distribution (should be > 1.5 for 10 classes)
    entropy = -(pi_t * torch.log(pi_t + 1e-10)).sum().item()
    logging.info("    μ_t (raw avg softmax): %s", [f"{mu_t[i]:.4f}" for i in range(num_classes)])
    logging.info("    π_t (smoothed):        %s", [f"{pi_t[i]:.4f}" for i in range(num_classes)])
    logging.info("    π_t entropy: %.3f  (max=%.3f)", entropy, math.log(num_classes))
    logging.info("    Clamped weights:       %s", [f"{importance_weights[i]:.3f}" for i in range(num_classes)])

    return pi_t, importance_weights


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def evaluate_with_prior_correction(
    model: nn.Module,
    loader: DataLoader,
    importance_weights: torch.Tensor,
    device: torch.device,
) -> float:
    """
    Inference with BBSE prior correction:
        p_corrected(y | x) ∝ w_y · p_model(y | x)
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


# ======================================================================
# TENT / SAR  –  Test-time ENTropy minimisation
# ======================================================================
# Reference:  Niu et al., "Towards Stable Test-Time Adaptation in Dynamic
#             Wild World", ICLR 2023  (SAR).
#             Wang et al., "Tent: Fully Test-Time Adaptation by Entropy
#             Minimization", ICLR 2021.
# ======================================================================


def _collect_bn_params(model: nn.Module):
    """Return lists of BN (weight, bias) parameters and their names."""
    params, names = [], []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            for np_name, p in m.named_parameters():
                if np_name in ("weight", "bias"):
                    params.append(p)
                    names.append(f"{nm}.{np_name}")
    return params, names


def tent_adapt(
    model: nn.Module,
    target_loader: DataLoader,
    device: torch.device,
    lr: float = 5e-4,
    steps: int = 2,
    num_classes: int = 10,
    entropy_margin: float = 0.4,
) -> nn.Module:
    """
    SAR-style selective entropy minimisation on the target stream.

    Only BN affine parameters (γ, β) are updated.  Samples whose prediction
    entropy exceeds `entropy_margin * ln(K)` are excluded from the loss to
    prevent the model from collapsing on very uncertain inputs.

    Parameters
    ----------
    model           : source-trained model (BN stats already recalibrated)
    target_loader   : unlabelled target batches
    device          : cuda / cpu
    lr              : learning rate for BN affine params
    steps           : number of full passes over the target stream
    num_classes     : K (for entropy threshold)
    entropy_margin  : fraction of max entropy (ln K) used as reliability gate

    Returns
    -------
    model with adapted BN affine parameters
    """
    model = copy.deepcopy(model)       # don't mutate the caller's model
    model.train()

    # ── Freeze everything except BN affine params ────────────────────
    for p in model.parameters():
        p.requires_grad_(False)

    bn_params, bn_names = _collect_bn_params(model)
    for p in bn_params:
        p.requires_grad_(True)

    if not bn_params:
        logging.warning("    TENT: no BN params found – skipping adaptation")
        model.eval()
        return model

    logging.info("    TENT: adapting %d BN affine params over %d step(s)", len(bn_params), steps)

    optimizer = torch.optim.Adam(bn_params, lr=lr)
    e_threshold = entropy_margin * math.log(num_classes)

    for step in range(steps):
        total_loss, n_reliable, n_total = 0.0, 0, 0
        for batch_data in target_loader:
            images = batch_data[0] if isinstance(batch_data, (tuple, list)) else batch_data
            images = images.to(device)

            logits = model(images)
            ent = softmax_entropy(logits)

            # SAR gate: keep only reliable (low-entropy) predictions
            mask = ent < e_threshold
            n_total += len(ent)
            n_reliable += mask.sum().item()

            if mask.sum() == 0:
                continue

            loss = ent[mask].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        fraction = n_reliable / max(n_total, 1)
        logging.info(
            "    TENT step %d/%d  |  loss %.4f  |  reliable %.1f%%",
            step + 1, steps, total_loss / max(len(target_loader), 1), 100 * fraction,
        )

    # Restore full requires_grad and switch to eval
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()
    return model
