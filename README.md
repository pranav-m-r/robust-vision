**Project**: Robust Computer Vision — Hackenza 2026

**Team**: Pranav M R, Jayant Chandwani, Ishaan Gupta, Abhav Garg

**Overview**
- Goal: Build a model that trains on a noisy, label-poisoned source (`source_toxic.pt`) and adapts to an unlabeled, noisy target (`static.pt`) using Test-Time Adaptation (TTA) methods so the final model is robust across 24 hidden corruption scenarios.
- Repository contents (key files):
  - `train.py` — Primary trainer + multi-phase pipeline (Phase**Project**: Robust Computer Vision — Hackenza 2026

**Team**: Pranav M R, Jayant Chandwani, Ishaan Gupta, Abhav Garg

**Overview**
- Goal: Build a model that trains on a noisy, label-poisoned source (`source_toxic.pt`) and adapts to an unlabeled, noisy target (`static.pt`) using Test-Time Adaptation (TTA) methods so the final model is robust across 24 hidden corruption scenarios.
- Repository contents (key files):
  - `train.py` — Primary trainer + multi-phase pipeline (Phase 1: training, Phase 2: confusion estimation, Phase 3–4: BNStats/TTA and submission generation).
  - `model_submission.py` — Model architecture: `RobustClassifier` and `load_weights()` support.
  - `requirements.txt` — Minimal python dependencies.
  - `generate_eval_data.py` — Local helper to generate small eval sets (used for development only).
  - `checkpoints/` — Saved phase checkpoints (phase_1_training.pt, phase_2_confusion.pt, train_latest.pt).
  - `results/` — Example outputs and weights used during development.

**Quick setup (Linux)**
1. Create & activate a Python venv (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
# additional recommended packages used by evaluation & utilities
pip install scikit-learn hydra-core omegaconf python-dotenv rich
```

3. Place the required dataset files into `data/`:
- `source_toxic.pt` (training; keys: `images`, `labels`)
- `static.pt` (unlabeled target; keys: `images`)
- `val_sanity.pt` (100 clean samples; keys: `images`, `labels`)
- `test_suite_public.pt` (optional local test suite for submission generation)

Note: example small eval data can be generated for development with `generate_eval_data.py`.

**Run full pipeline (reproducible)**
- Train from scratch (this follows the Forensic Audit rules):

```bash
# run full train pipeline; it will save phase checkpoints under checkpoints/
python train.py --data-dir data
```

- Continue from existing checkpoints / compute Phase 2 only (if checkpoints exist):

```bash
# If a `checkpoints/phase_1_training.pt` exists the script will load it automatically.
python train.py --data-dir data
```

- Evaluate locally (development):

```bash
# generate a small eval dataset (dev only)
python generate_eval_data.py

# run evaluation pipeline (uses results/SCE + tent/weights.pth by default)
python evaluate_models.py
```

**Core logic (what we implemented and why)**
- Phase 1 — Robust training on noisy labels
  - Uses Symmetric Cross-Entropy (`SCELoss`) to mitigate label noise (CE + reverse-CE term).
  - Training is standard SGD with cosine LR schedule + linear warmup.
  - Checkpointing: phase-level checkpoints (`checkpoints/phase_1_training.pt`) and `train_latest.pt` for resume.

- Phase 2 — Confusion matrix estimation & correction
  - Compute empirical noisy confusion matrix on held-out validation (from source noisy labels), correct it using the known symmetric noise transition matrix `T` (the challenge provides noise rate), then invert (with Tikhonov regularisation) to obtain `C_true` used for BBSE.(`Source:- https://arxiv.org/pdf/1609.03683`)
  - Regularisation used to stabilise inversion: `BBSE_REG` (configurable, set in `train.py`).

- Phase 3 — Test-Time Adaptation (TTA)
  - `adapt_bn()` : Reset & re-estimate BatchNorm running statistics on target data (cumulative moving average) — aligns feature distributions.
  - `tent_adapt()` : Entropy-minimization fine-tuning of BN affine parameters (γ, β) only — fast, unsupervised adaptation that avoids changing convolution weights.

- Phase 4 — BBSE (Black-Box Shift Estimation)
  - Estimate target class priors from model predictions using the corrected confusion matrix (`C_true`), apply label-shift correction by shifting logits: `logits + log(p_target / p_source)`.

Why this approach?
- The pipeline enforces strict separation: Phase 1 trains from source-only noisy data (no external clean data). Phases 3–4 adapt to the target without labels (unsupervised), using methods that are fast and preserve generalization (BN re-alignment + entropy minimization + BBSE for prior correction).

**Forensic Audit & Reproducibility Notes (MUST READ)**
- The organizers will re-run `train.py` from scratch on `source_toxic.pt`. To comply:
  - `train.py` trains from random initialization by default; it does not load external pretrained weights.
  - Do NOT add any pretrained checkpoint loading in `train.py` if you intend to publish as the final submission.
  - Any use of external labeled data or forbidden augmentations will lead to disqualification.
- The `model_submission.py` contains `RobustClassifier` and a `load_weights(path)` helper to satisfy the grader's automated weight loading.
- Checkpoints created under `checkpoints/` are for convenience and reproducibility; the final submission should include the `weights.pth` you want graded alongside `train.py` and `model_submission.py`.

**How to reproduce the reported results**
1. Ensure `data/source_toxic.pt` and `data/val_sanity.pt` are available.
2. Run full training: `python train.py --data-dir data`. This will create `checkpoints/phase_1_training.pt` and `train_latest.pt`.
3. Optionally run `python generate_eval_data.py` (dev) then `python evaluate_models.py` to reproduce local evaluation metrics.

**Recommended hyperparameters (tuning suggestions)**
- `SCE_ALPHA`, `SCE_BETA` (loss weights) — trade-off between CE and robust RCE component.
- `TENT_LR` and `TENT_STEPS` — controls speed & strength of TTA (higher lr/steps helps severe corruptions but risks overfitting to target batch).
- `BBSE_REG` — Tikhonov regularisation to stabilise confusion-matrix inversion under extreme corruptions.
All top-level constants live in `train.py` so experiments are reproducible via simple code edits or CLI flags if you add them.

**Unit / smoke tests**
- Minimal smoke test to check environment & model import:

```bash
python -c "import torch; from model_submission import RobustClassifier; m=RobustClassifier(); print('OK', sum(p.numel() for p in m.parameters()))"
```

- Run quick eval pipeline (dev):

```bash
python generate_eval_data.py
python evaluate_models.py
```

**Accessibility & usability considerations**
- CLI scripts are small and produce plaintext logs; saved evaluation text files are timestamped.
- All data I/O uses PyTorch `torch.load`/`torch.save` for consistent, platform-independent behaviour.
- Keep batch sizes and device placement in `train.py` as top-level constants for quick adaptation on different hardware.

**Files to include with a final submission**
- `train.py` (trainer)
- `model_submission.py` (architecture + `load_weights` method)
- `weights.pth` (final model weights produced by your run of `train.py`)
- `submission.csv` (output predictions for public demo)
- `README.md` (this file)

**Notes / Constraints**
- The provided `requirements.txt` includes the minimal dependencies; we recommend installing `scikit-learn` and a few helper packages used during development (listed above).
- `generate_eval_data.py` is for development/testing only — do NOT use it during Phase 1 training for submissions (it generates corruptions that are forbidden during the training phase per the competition rules).

---
*This README was created to document the project structure, how to run and reproduce the pipeline, and to explain the core methods used for robust test-time adaptation.*
 1: training, Phase 2: confusion estimation, Phase 3–4: BNStats/TTA and submission generation).
  - `model_submission.py` — Model architecture: `RobustClassifier` and `load_weights()` support.
  - `requirements.txt` — Minimal python dependencies.
  - `generate_eval_data.py` — Local helper to generate small eval sets (used for development only).
  - `checkpoints/` — Saved phase checkpoints (phase_1_training.pt, phase_2_confusion.pt, train_latest.pt).
  - `results/` — Example outputs and weights used during development.

**Quick setup (Linux)**
1. Create & activate a Python venv (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
# additional recommended packages used by evaluation & utilities
pip install scikit-learn hydra-core omegaconf python-dotenv rich
```

3. Place the required dataset files into `data/`:
- `source_toxic.pt` (training; keys: `images`, `labels`)
- `static.pt` (unlabeled target; keys: `images`)
- `val_sanity.pt` (100 clean samples; keys: `images`, `labels`)
- `test_suite_public.pt` (optional local test suite for submission generation)

Note: example small eval data can be generated for development with `generate_eval_data.py`.

**Run full pipeline (reproducible)**
- Train from scratch (this follows the Forensic Audit rules):

```bash
# run full train pipeline; it will save phase checkpoints under checkpoints/
python train.py --data-dir data
```

- Continue from existing checkpoints / compute Phase 2 only (if checkpoints exist):

```bash
# If a `checkpoints/phase_1_training.pt` exists the script will load it automatically.
python train.py --data-dir data
```

- Evaluate locally (development):

```bash
# generate a small eval dataset (dev only)
python generate_eval_data.py

# run evaluation pipeline (uses results/SCE + tent/weights.pth by default)
python evaluate_models.py
```

**Core logic (what we implemented and why)**
- Phase 1 — Robust training on noisy labels
  - Uses Symmetric Cross-Entropy (`SCELoss`) to mitigate label noise (CE + reverse-CE term).
  - Training is standard SGD with cosine LR schedule + linear warmup.
  - Checkpointing: phase-level checkpoints (`checkpoints/phase_1_training.pt`) and `train_latest.pt` for resume.

- Phase 2 — Confusion matrix estimation & correction
  - Compute empirical noisy confusion matrix on held-out validation (from source noisy labels), correct it using the known symmetric noise transition matrix `T` (the challenge provides noise rate), then invert (with Tikhonov regularisation) to obtain `C_true` used for BBSE.(`Source:- https://arxiv.org/pdf/1609.03683`)
  - Regularisation used to stabilise inversion: `BBSE_REG` (configurable, set in `train.py`).

- Phase 3 — Test-Time Adaptation (TTA)
  - `adapt_bn()` : Reset & re-estimate BatchNorm running statistics on target data (cumulative moving average) — aligns feature distributions.
  - `tent_adapt()` : Entropy-minimization fine-tuning of BN affine parameters (γ, β) only — fast, unsupervised adaptation that avoids changing convolution weights.

- Phase 4 — BBSE (Black-Box Shift Estimation)
  - Estimate target class priors from model predictions using the corrected confusion matrix (`C_true`), apply label-shift correction by shifting logits: `logits + log(p_target / p_source)`.

Why this approach?
- The pipeline enforces strict separation: Phase 1 trains from source-only noisy data (no external clean data). Phases 3–4 adapt to the target without labels (unsupervised), using methods that are fast and preserve generalization (BN re-alignment + entropy minimization + BBSE for prior correction).

**Forensic Audit & Reproducibility Notes (MUST READ)**
- The organizers will re-run `train.py` from scratch on `source_toxic.pt`. To comply:
  - `train.py` trains from random initialization by default; it does not load external pretrained weights.
  - Do NOT add any pretrained checkpoint loading in `train.py` if you intend to publish as the final submission.
  - Any use of external labeled data or forbidden augmentations will lead to disqualification.
- The `model_submission.py` contains `RobustClassifier` and a `load_weights(path)` helper to satisfy the grader's automated weight loading.
- Checkpoints created under `checkpoints/` are for convenience and reproducibility; the final submission should include the `weights.pth` you want graded alongside `train.py` and `model_submission.py`.

**How to reproduce the reported results**
1. Ensure `data/source_toxic.pt` and `data/val_sanity.pt` are available.
2. Run full training: `python train.py --data-dir data`. This will create `checkpoints/phase_1_training.pt` and `train_latest.pt`.
3. Optionally run `python generate_eval_data.py` (dev) then `python evaluate_models.py` to reproduce local evaluation metrics.

**Recommended hyperparameters (tuning suggestions)**
- `SCE_ALPHA`, `SCE_BETA` (loss weights) — trade-off between CE and robust RCE component.
- `TENT_LR` and `TENT_STEPS` — controls speed & strength of TTA (higher lr/steps helps severe corruptions but risks overfitting to target batch).
- `BBSE_REG` — Tikhonov regularisation to stabilise confusion-matrix inversion under extreme corruptions.
All top-level constants live in `train.py` so experiments are reproducible via simple code edits or CLI flags if you add them.

**Unit / smoke tests**
- Minimal smoke test to check environment & model import:

```bash
python -c "import torch; from model_submission import RobustClassifier; m=RobustClassifier(); print('OK', sum(p.numel() for p in m.parameters()))"
```

- Run quick eval pipeline (dev):

```bash
python generate_eval_data.py
python evaluate_models.py
```

**Accessibility & usability considerations**
- CLI scripts are small and produce plaintext logs; saved evaluation text files are timestamped.
- All data I/O uses PyTorch `torch.load`/`torch.save` for consistent, platform-independent behaviour.
- Keep batch sizes and device placement in `train.py` as top-level constants for quick adaptation on different hardware.

**Files to include with a final submission**
- `train.py` (trainer)
- `model_submission.py` (architecture + `load_weights` method)
- `weights.pth` (final model weights produced by your run of `train.py`)
- `submission.csv` (output predictions for public demo)
- `README.md` (this file)

**Notes / Constraints**
- The provided `requirements.txt` includes the minimal dependencies; we recommend installing `scikit-learn` and a few helper packages used during development (listed above).
- `generate_eval_data.py` is for development/testing only — do NOT use it during Phase 1 training for submissions (it generates corruptions that are forbidden during the training phase per the competition rules).

---
*This README was created to document the project structure, how to run and reproduce the pipeline, and to explain the core methods used for robust test-time adaptation.*
