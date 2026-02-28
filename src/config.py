import math
from dataclasses import dataclass, field


@dataclass
class TrainDataConfig:
    """Paths for the training pipeline."""
    train_path: str = "data/source_toxic.pt"
    val_path: str = "data/val_sanity.pt"


@dataclass
class EvalDataConfig:
    """Paths for the evaluation pipeline."""
    model_path: str = "results/train/base_model.pt"
    val_path: str = "data/val_sanity.pt"      # clean labels → calibration & accuracy monitoring
    source_path: str = "data/source_toxic.pt"  # noisy labels → BBSE confusion matrix (denoised via T⁻¹)
    test_path: str = "data/static.pt"          # target domain → BN recal & BBSE priors


@dataclass
class TrainingConfig:
    """Hyper-parameters for the training loop."""
    seed: int = 42
    epochs: int = 25
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9
    num_classes: int = 10
    augmentation_factor: int = 1


@dataclass
class InferenceConfig:
    """Minimal config for EvalPipeline (no training-specific fields)."""
    seed: int = 42
    batch_size: int = 64
    num_classes: int = 10
    noise_rate: float = 0.3   # source_toxic has 30% symmetric label noise


@dataclass
class LossConfig:
    """Hyper-parameters for TruncatedLoss."""
    q: float = 0.7
    k: float = 0.5
    warmup_epochs: int = 10   # CE warmup before truncation kicks in


@dataclass
class TemperatureConfig:
    """Hyper-parameters for post-hoc temperature scaling."""
    t_min: float = 0.5
    t_max: float = 5.0
    steps: int = 200


@dataclass
class AdaptationConfig:
    """Hyper-parameters for test-time adaptation."""
    # ── On/Off switches for each adaptation step ─────────────────────
    enable_temp_scaling: bool = True      # [1] Temperature scaling
    enable_bn_recal: bool = True          # [2] Alpha-blended BN recalibration
    enable_tent: bool = True              # [3] TENT / SAR entropy minimisation
    enable_prior_estimation: bool = True  # [4] Label shift prior estimation
    enable_prior_correction: bool = True  # [5] Prior-corrected inference
    # ── Hyperparameters ──────────────────────────────────────────────
    tent_lr: float = 5e-4
    tent_steps: int = 2
    entropy_margin: float = 0.5   # fraction of ln(K) used as SAR threshold (raised from 0.4)
    bn_alpha: float = 0.1         # BN recal blend: 0=keep source, 1=full target replacement


@dataclass
class TrainPipelineConfig:
    """Config for TrainPipeline – trains the model and saves base_model.pt."""
    _target_: str = "src.train_pipeline.TrainPipeline"
    data: TrainDataConfig = field(default_factory=TrainDataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)


@dataclass
class EvalPipelineConfig:
    """Config for EvalPipeline – adapts the model and generates predictions."""
    _target_: str = "src.eval_pipeline.EvalPipeline"
    data: EvalDataConfig = field(default_factory=EvalDataConfig)
    training: InferenceConfig = field(default_factory=InferenceConfig)
    temperature: TemperatureConfig = field(default_factory=TemperatureConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)


@dataclass
class LoggingConfig:
    """Logging / output directory settings."""
    log_dir: str = "logs"
    results_dir: str = "results"
    console: bool = True


@dataclass
class Config:
    """Top-level Hydra config. Registered in the config store as 'base_config'."""
    train_pipeline: TrainPipelineConfig = field(default_factory=TrainPipelineConfig)
    eval_pipeline: EvalPipelineConfig = field(default_factory=EvalPipelineConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
