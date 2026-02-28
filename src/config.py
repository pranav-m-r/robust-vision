from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Paths to the three .pt dataset files."""
    source_path: str = "source_toxic.pt"
    val_path: str = "val_sanity.pt"
    target_path: str = "target_static.pt"


@dataclass
class TrainingConfig:
    """Hyper-parameters for the training loop."""
    seed: int = 42
    epochs: int = 25
    batch_size: int = 64
    lr: float = 1e-5
    num_classes: int = 10


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
class PipelineConfig:
    """Config consumed by src.pipeline.Pipeline via hydra.utils.instantiate."""
    _target_: str = "src.pipeline.Pipeline"
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
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
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
