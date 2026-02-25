from dataclasses import dataclass
from imports import *

@dataclass
class PathsConfig:
    """File path configurations. """
    og_images_root: Path
    inference_dir: Path
    cropped_img_root: Path
    checkpoint_dir: Path

@dataclass
class DatasetConfig:
    """Dataset processing configurations. """
    patch_size: int
    train_split: float
    valid_split: float
    num_inference_imgs: int

@dataclass
class ModelConfig:
    """ Model architecture configuration. """
    name: str
    in_channels: int
    out_channels: int
    init_features: int

@dataclass
class LossConfig:
    """ Loss function configuration. """
    name: str
    alpha: float = 0.8      # For CombinedLoss

@dataclass
class PreprocessingConfig:
    """ Preprocessing augmentation settings. """
    bright_threshold: float
    bright_copies: int
    random_augment: int

@dataclass
class AugmentationConfig:
    """ Runtime augmentation settings (ColorJitter). """
    enabled: bool
    brightness: float
    contrast: float
    saturation: float
    hue: float

@dataclass
class TrainConfig:
    """Training parameters and checkpoint settings."""
    learning_rate: float
    epochs: int
    batch_size: int
    num_workers: int
    save_every: int
    keep_last_k: int
    optimiser: str
    resume_from: str | None
    patience: int

@dataclass
class SchedulerConfig:
    """ Learning rate scheduler settings. """
    type: str
    mode: str
    factor: float
    patience: int
    min_lr: float

@dataclass
class ValidationConfig:
    """ Validation settings. """
    val_every: int
    save_comparisons: bool

@dataclass
class TestConfig:
    """ Testing settings. """
    batch_size: int
    save_outputs: bool

@dataclass
class InferenceConfig:
    """ Inference settings for tiled processing. """
    tiled: bool
    tile_size: int
    overlap: int
    batch_size: int

@dataclass
class Config:
    """Main configuration object."""
    paths: PathsConfig
    dataset: DatasetConfig
    model: ModelConfig
    loss: LossConfig
    preprocessing: PreprocessingConfig
    augmentation: AugmentationConfig
    train: TrainConfig
    scheduler: SchedulerConfig
    validation: ValidationConfig
    test: TestConfig
    inference: InferenceConfig

def load_config(path: str | Path) -> Config:
    """
    Load configuration from YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        Config object with all settings.

    Raises:
        FileNotFoundError: If configuration file does not exist
        yaml.YAMLError: If YAML is malformed
    """
    config_path = Path(path).resolve()
    base = config_path.parent
    d = yaml.safe_load(config_path.read_text())

    # paths
    p = d["paths"]
    paths = PathsConfig(
        og_images_root=(base / p["og_images_root"]).resolve(),
        inference_dir=(base / p["inference_dir"]).resolve(),
        cropped_img_root=(base / p["cropped_img_root"]).resolve(),
        checkpoint_dir=(base / p["ckpt_dir"]).resolve()
    )

    # dataset
    ds = d.get("dataset", {})
    dataset = DatasetConfig(
        patch_size=int(ds.get("patch_size")),
        train_split=float(ds.get("train_split")),
        valid_split=float(ds.get("valid_split")),
        num_inference_imgs=int(ds.get("num_inference_imgs"))
    )

    # model
    m = d.get("model", {})
    model = ModelConfig(
        name=str(m.get("name", "UNet")),
        in_channels=int(m.get("in_channels", 3)),
        out_channels=int(m.get("out_channels", 3)),
        init_features=int(m.get("init_features", 32))
    )

    # loss
    l = d.get("loss", {})
    loss = LossConfig(
        name=str(l.get("name", "MSELoss")),
        alpha=float(l.get("alpha", 0.8))
    )

    # preprocessing
    prep = d.get("preprocessing", {})
    preprocessing = PreprocessingConfig(
        bright_threshold=float(prep.get("bright_threshold", 200.0)),
        bright_copies=int(prep.get("bright_copies", 5)),
        random_augment=int(prep.get("random_augment", 50))
    )

    # augmentation
    aug = d.get("augmentation", {})
    augmentation = AugmentationConfig(
        enabled=bool(aug.get("enabled", False)),
        brightness=float(aug.get("brightness", 0.2)),
        contrast=float(aug.get("contrast", 0.2)),
        saturation=float(aug.get("saturation", 0.2)),
        hue=float(aug.get("hue", 0.05))
    )

    # train
    t = d["train"]
    train = TrainConfig(
        learning_rate=float(t.get("learning_rate", 1e-4)),
        epochs=int(t["epochs"]),
        batch_size=int(t["batch_size"]),
        num_workers=int(t["num_workers"]),
        save_every=int(t["save_every"]),
        keep_last_k=int(t["keep_last_k"]),
        optimiser=str(t["optimiser"]),
        resume_from=t.get("resume_from", None),
        patience=int(t["patience"]),
    )

    # scheduler
    sch = d.get("scheduler", {})
    scheduler = SchedulerConfig(
        type=str(sch.get("type", "ReduceLROnPlateau")),
        mode=str(sch.get("mode", "min")),
        factor=float(sch.get("factor", 0.5)),
        patience=int(sch.get("patience", 5)),
        min_lr=float(sch.get("min_lr", 1e-7))
    )

    # validation
    val = d.get("validation", {})
    validation = ValidationConfig(
        val_every=int(val.get("val_every", 1)),
        save_comparisons=bool(val.get("save_comparisons", True))
    )

    # test
    test_cfg = d.get("test", {})
    test = TestConfig(
        batch_size=int(test_cfg.get("batch_size", 16)),
        save_outputs=bool(test_cfg.get("save_outputs", True))
    )

    # inference
    inf = d.get("inference", {})
    inference = InferenceConfig(
        tiled=bool(inf.get("tiled", True)),
        tile_size=int(inf.get("tile_size", 512)),
        overlap=int(inf.get("overlap", 64)),
        batch_size=int(inf.get("batch_size", 4))
    )
    return Config(
        paths=paths,
        dataset=dataset,
        model=model,
        loss=loss,
        preprocessing=preprocessing,
        augmentation=augmentation,
        train=train,
        scheduler=scheduler,
        validation=validation,
        test=test,
        inference=inference
    )

