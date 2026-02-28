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
    val_path: str = "data/val_sanity.pt"
    test_path: str = "data/static.pt"


@dataclass
class TrainingConfig:
    """Hyper-parameters for the training loop."""
    seed: int = 42
    epochs: int = 25
    batch_size: int = 64
    lr: float = 1e-5
    num_classes: int = 10
    augmentation_factor: int = 5


@dataclass
class InferenceConfig:
    """Minimal config for EvalPipeline (no training-specific fields)."""
    seed: int = 42
    batch_size: int = 64
    num_classes: int = 10
    noise_rate: float = 0.3


@dataclass
class LossConfig:
    """Hyper-parameters for TruncatedLoss."""
    q: float = 0.7
    k: float = 0.5


@dataclass
class TemperatureConfig:
    """Hyper-parameters for post-hoc temperature scaling."""
    t_min: float = 0.1
    t_max: float = 10.0
    steps: int = 200


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
