# Robust Vision Challenge [Hackenza 2026]

Team: Pranav M R, Jayant Chandwani, Ishaan Gupta, Abhav Garg

## Overview
This repository contains our submission for the Robust Vision challenge. The goal is to train on a noisy / label-poisoned source dataset and robustly adapt to an unlabeled target domain at test time.

**Key idea:** train a noise-robust classifier on `source_toxic.pt`, then apply lightweight, unsupervised test-time adaptation (BatchNorm stat alignment + entropy minimization) and label-shift correction (BBSE) on the target distribution.

### Repository layout
- `train.py` — end-to-end multi-phase pipeline (training → estimation → test-time adaptation → submission)
- `model_submission.py` — model definition (`RobustClassifier`) + `load_weights(path)` helper for graders
- `requirements.txt` — Python dependencies
- `data/` — datasets (place provided `.pt` files here)
- `checkpoints/` — intermediate checkpoints created by `train.py`

### Setup
Create a virtual environment and install dependencies:

**Windows (PowerShell)**
```powershell
conda create -n rv python=3.11
conda activate rv
pip install -r requirements.txt
```

**Linux/macOS (bash/zsh)**
```bash
conda create -n rv python=3.11
conda activate rv
pip install -r requirements.txt
```

### Data
Place the challenge-provided files under `data/`:
- `source_toxic.pt` — training set (expects keys: `images`, `labels`)
- `static.pt` — unlabeled target set (expects keys: `images`)
- `test_suite_public.pt` — scenario suite used to generate the final submission (expects scenario tensors)
- `val_sanity.pt` — small clean validation set (expects keys: `images`, `labels`)

### Run
Run the full pipeline:

```bash
# Full pipeline: train from scratch to generate submission.csv
python train.py
```

`train.py` is fully resumable. If interrupted, re-running it will pick up from the last saved checkpoint. Delete `checkpoints/` to force a clean retraining run.

Outputs:
- `weights.pth` — final weights for submission
- `submission.csv` — predictions in the expected format
- `checkpoints/` — phase checkpoints for resuming runs

### Phases at a glance
- **Phase 1 — Robust training:** train from random initialization on noisy labels using Symmetric Cross-Entropy (SCE).
- **Phase 2 — Confusion estimation:** estimate and correct a confusion matrix used for label-shift estimation.
- **Phase 3 — Test-time adaptation:** re-estimate BatchNorm running stats on target data and optionally apply TENT-style entropy minimization (BN affine params only).
- **Phase 4 — BBSE prior correction:** estimate target priors and apply a log-prior correction to predictions.

## Architecture & Compliance

**Model:** Custom ResNet-style classifier (`RobustClassifier`). It is a 3-stage residual network for 1×28×28 grayscale inputs: a stem (Conv→BN→ReLU), then three residual stages (64→64→128→256 channels; 28×28→14×14→7×7), followed by AdaptiveAvgPool, Dropout(0.25), and a Linear(256, 10) head. Each residual block contains two BatchNorm2d layers (plus one on the skip connection when channels or stride change), for **15 BatchNorm2d layers** total. This is intentional because BatchNorm statistics are our main handle for test-time adaptation.

**Initialization:** Random initialization. No pre-trained weights are used at any stage.

**Augmentation:** Standard geometry and photometric transforms only: `RandomCrop(28, padding=4, reflect)`, `RandomHorizontalFlip`, `RandomRotation(15°)`, `RandomErasing(p=0.25)`. We do not use corruption-simulating augmentations (no AugMix, PixMix, or corruption-mimicking transforms).

## Evaluation setup

To compare approaches during development, we built a private `eval_suite` from labeled data, containing **8 corruption scenarios** at fixed severity levels:

| Scenario | Type |
| --- | --- |
| `contrast_reduction` | Photometric |
| `defocus_blur` | Blur |
| `gaussian_noise` | Additive noise |
| `impulse_noise_medium` | Salt-and-pepper noise |
| `impulse_noise_heavy` | Salt-and-pepper noise (severe) |
| `pixelate_medium` | Spatial downsampling |
| `posterize` | Bit-depth reduction |
| `shot_noise` | Poisson noise |

Each scenario contains 5,000 labeled images with a **deliberately uneven class distribution** (e.g., Sandal = 35%, Trouser = 1.2%) to simulate the type of prior shift we expect in the hidden evaluation. The training set is near-uniform (≈6,000 per class), so the model must handle both covariate shift and label shift. We used **Macro-F1** as the primary comparison metric because it weights all 10 classes equally regardless of support, making it sensitive to failures on minority classes. All ablations reported here (GCE vs. SCE, TENT on/off, temperature scaling) were measured on this suite.

Compliance note: `eval_suite` was used only for evaluation and hyperparameter tuning. We did not use it to train the submitted model in any form (no gradient updates and no mixing into the training set).

## Phase 1: Robust training (Decontamination)

**Approach:** We train on `source_toxic.pt` (60,000 images, 30% symmetric label noise) using [**Symmetric Cross-Entropy (SCE) loss**](https://arxiv.org/abs/1908.06112). Optimization uses SGD (momentum=0.9, weight_decay=5e-4) with a 5-epoch linear warm-up followed by cosine annealing, for 100 epochs total. We keep the checkpoint with the best noisy-validation accuracy.

Training-data restriction note: all learned weights come only from `source_toxic.pt`, as instructed. We do not train on any additional labeled data, and we avoid banned augmentation techniques (for example, Mixup). Only the permitted augmentations listed in the first section are used.

**What we tried:** We initially trained with [**Generalized Cross-Entropy (GCE) loss**](https://arxiv.org/abs/1805.07836) ($q=0.7$), which interpolates between MAE and CE and is well-known to resist noisy gradients. After switching to SCE we saw a consistent improvement across all corruption scenarios on our eval set, so we went with SCE for the final submission. In practice the difference wasn't dramatic, but SCE was more stable to tune.

**Justification:** SCE combines a standard CE term with a Reverse CE (RCE) term:

$$\mathcal{L}_{SCE} = \alpha \cdot \underbrace{-\log p_y}_{\text{CE}} + \beta \cdot \underbrace{-\sum_k p_k \log \tilde{y}_k}_{\text{RCE}}$$

With symmetric label noise at rate $\eta$, the CE gradient can become dominated by memorized noise once the model becomes confident. The RCE term counteracts this by treating the model's softmax output as a soft label and penalizing deviation from the (smoothed) one-hot target. This makes the loss *symmetric* in $(p, y)$ and helps bound the influence of noisy samples. Coefficients $\alpha=1.0$ and $\beta=0.5$ balance learning speed against noise resistance.

## Phase 2: Distribution estimation (Reconnaissance)

**Approach:** After Phase 1, we compute $\hat{C_n}$ on the held-out 10% noisy validation split. Instead of hard argmax predictions, we accumulate **soft confusion counts from the model's softmax probabilities**. This gives a smoother, lower-variance estimate of $C_{noisy}[j,k] = P(\hat{y}=k \mid \tilde{y}=j)$, especially for minority classes. Using the known [**symmetric noise transition matrix T**](https://arxiv.org/abs/1609.03683) (with $\eta=0.3$), we recover the **clean confusion matrix** $C_{true}$, which is used by the BBSE estimator in Phase 3.

**What we tried:** We first tried a standard hard-prediction confusion matrix on the clean `val_sanity.pt` set, but that set has only 100 samples, so the estimates were noisy. Switching to the full 10% noisy validation split with soft counts was more stable. We also experimented with [**temperature scaling**](https://arxiv.org/abs/1706.04599) before BBSE inversion, aiming for a better $\mu_{target}$ estimate. In our tests it hurt performance because it amplified the BBSE-estimated class weights and over-corrected the prior, so we removed it.

**Justification:** The key insight, formalized by Patrini et al., is that under class-dependent label noise the *noisy* posterior and the *clean* posterior are related by the transition matrix $T$. For symmetric noise at rate $\eta$ across $K$ classes, $T$ takes a closed form:

$$T_{ij} = P(\tilde{y}=j \mid y=i) = \begin{cases} 1 - \eta & \text{if } i = j \\ \dfrac{\eta}{K-1} & \text{if } i \neq j \end{cases}$$

Every clean label stays correct with probability $1-\eta=0.7$, and flips to any one of the other $K-1=9$ classes with equal probability $\eta/(K-1) \approx 0.033$. This gives a diagonally dominant matrix that is easy to invert analytically.

Because the model is trained on noisy labels, the resulting confusion matrix $C_{noisy}[j,k] = P(\hat{y}=k \mid \tilde{y}=j)$ reflects noisy-label statistics. The correction step recovers the clean version:

$$C_{noisy} = T^\top C_{true} \quad \Longrightarrow \quad C_{true} = T^{-1} C_{noisy}$$

$C_{true}$ then acts as the BBSE channel matrix. It describes how predictions distribute per true class, which is exactly what BBSE needs to invert label shift at test time. This lets BBSE handle large prior shifts (e.g., Sandal = 35% of eval samples vs. Trouser = 1.2%) without requiring any target labels.

## Phase 3: Test-time adaptation (Alignment)

**Approach:** For each test scenario, three sequential adaptation steps are applied to a freshly reloaded copy of the trained model:

1. **[BN Statistics Reset & Re-estimation](https://arxiv.org/abs/2006.10963) (`adapt_bn`):** All BatchNorm running means and variances are zeroed and recomputed over the target batch in a single no-grad forward pass (`momentum=None` for a cumulative average). This replaces source-domain BN statistics with target-domain statistics and helps correct covariate shift induced by sensor noise.

2. **[TENT Entropy Minimization](https://arxiv.org/abs/2006.10726) (`tent_adapt`):** Only the BN affine parameters ($\gamma$, $\beta$) are unfrozen and updated for 10 Adam steps (lr=2e-3), minimizing Shannon entropy $H = -\sum_k p_k \log p_k$ over the target batch. This encourages the model to produce confident, low-entropy predictions under the new domain.

3. **[BBSE Prior Correction](https://arxiv.org/abs/1802.03916) (`predict_with_prior`):** The target class prior $\hat{p}_{target}$ is estimated by inverting the BBSE equation: $\hat{\mu}_{target} = C_{true}^\top \hat{p}_{target}$, where $\hat{\mu}_{target}$ is the empirical prediction-frequency vector. Final predictions use a log-ratio correction:

$$\hat{y} = \arg\max_k \left[\log f_k(x) + \log \frac{\hat{p}_{target}[k]}{1/K}\right]$$

**What we tried:** We ran the full pipeline with and without TENT. For mild corruptions like `contrast_reduction` and `defocus_blur`, the difference was small. For harder shifts like `pixelate_medium`, TENT produced a meaningful improvement because entropy minimization encourages more decisive predictions when BN statistics alone do not fully close the domain gap. TENT can become unstable if the learning rate or step count is too high (it may collapse to a single class), so we kept it conservative at 8-10 steps with lr=2e-3. We found 8 steps to perform slightly better.

**Generalization Justification:** BN re-estimation and TENT are both unsupervised and react to the test distribution as it arrives, without assuming a specific corruption type. By updating only BN parameters (a small fraction of the total weights), the feature representation learned during Phase 1 stays largely intact. Only the internal normalization shifts to match the new domain, which is why we expect the approach to transfer to unseen corruptions in `hidden_eval.pt`.

## References

- [Symmetric Cross-Entropy Loss for Robust Learning with Noisy Labels](https://arxiv.org/abs/1908.06112) (Wang et al., ECCV 2020) -> SCE
- [Generalized Cross-Entropy Loss for Training Deep Neural Networks with Noisy Labels](https://arxiv.org/abs/1805.07836) (Zhang et al., NeurIPS 2018) -> GCE
- [Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach](https://arxiv.org/abs/1609.03683) (Patrini et al., CVPR 2017) -> Noise Transition Matrix
- [Test-Time Training with Self-Supervision for Generalization under Distribution Shifts](https://arxiv.org/abs/2006.10963) (Sun et al., ICML 2020) -> BNStats
- [Black Box Shift Estimation](https://arxiv.org/abs/1802.03916) (Lipton et al., ICML 2018) -> BBSE
- [Tent: Fully Test-Time Adaptation by Entropy Minimization](https://arxiv.org/abs/2006.10726) (Wang et al., ICLR 2021) -> Tent
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) (Guo et al., ICML 2017) -> Temperature Scaling
