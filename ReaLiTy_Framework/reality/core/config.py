"""Load, validate and snapshot run configuration.

A run is fully described by YAML (README -> *Configuration*); nothing about
datasets, sensor geometry, task type or hyperparameters is hard-coded in the
pipeline. YAML is parsed into dataclasses, validated strictly (unknown keys are
errors, not silent typos) and can be snapshotted back to YAML so any run is
reproducible from its checkpoint.
"""

import dataclasses
import typing
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

__all__ = [
    "ConfigError",
    "LabelSpec",
    "DataSpec",
    "TaskSpec",
    "SensorSpec",
    "DegradationSpec",
    "ModelSpec",
    "NormalizationSpec",
    "TrainingSpec",
    "OutputSpec",
    "Config",
    "load_config",
]

TASK_TYPES = ("sensor", "weather")
WEATHER_TYPES = ("rain", "snow")
DEGRADATION_TYPES = ("physics", "learned")
#: Datasets whose layout must be declared in config rather than assumed.
SELF_DESCRIBING_DATASETS = ("generic",)
#: Where normalization constants may come from.
NORMALIZATION_SOURCES = ("computed", "picgan_default")
#: Split folders in the released dataset layout.
DATASET_SPLITS = ("train", "test")
#: the weather model's lidar return modes.
LIDAR_RETURN_MODES = ("strongest", "last")
#: the weather model implementations vendored under reality/structure/weather_model.
WEATHER_MODEL_MODULES = ("atmos_models", "atmos_models_cpu")
#: Output activations the target generator supports.
OUTPUT_ACTIVATIONS = ("tanh", "sigmoid")


class ConfigError(ValueError):
    """Raised when a config is malformed, incomplete or internally inconsistent."""


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


@dataclass
class LabelSpec:
    """Optional semantic labels; their presence enables the reflectance channel."""

    path: str
    format: str = "semantickitti"


@dataclass
class DataSpec:
    """A dataset and the role it plays in this run.

    Identity lives on disk (``<data_root>/<Dataset>/``); role is decided here. The
    same ``Data/KITTI`` therefore serves as the source of one run and, with a
    different config, the target of another, with no data movement.
    """

    dataset: str
    #: Explicit location. Usually omitted: the adapter resolves
    #: ``<data_root>/<Dataset>`` from the dataset name.
    path: Optional[str] = None
    format: Optional[str] = None
    #: Declared column layout, e.g. ``[x, y, z, intensity, ring]``. Never guessed.
    columns: Tuple[str, ...] = ()
    intensity_scale: float = 1.0
    labels: Optional[LabelSpec] = None
    #: Restrict a sequenced dataset to these sequences (KITTI 00-10 are labelled,
    #: 11-21 are not; mixing them would mix 3- and 2-channel sources in one run).
    #: Only applies to the raw-layout fallback; the released layout uses folders.
    sequences: Tuple[str, ...] = ()
    #: Which split folder to read: "train" or "test". Datasets organised in the
    #: released layout keep both under the same root.
    split: str = "train"

    @property
    def has_labels(self) -> bool:
        return self.labels is not None


@dataclass
class TaskSpec:
    """``sensor`` (projection -> PICGAN) or ``weather`` (projection -> the weather model -> PICGAN)."""

    type: str

    @property
    def is_weather(self) -> bool:
        return self.type == "weather"


@dataclass
class SensorSpec:
    """Spherical-projection geometry of the target sensor."""

    proj_H: int
    proj_W: int
    fov_up: float
    fov_down: float

    @property
    def shape(self) -> Tuple[int, int]:
        return self.proj_H, self.proj_W


@dataclass
class DegradationSpec:
    """Geometric degradation stage. Enabled for weather transfer, skipped otherwise.

    The the weather model parameters carry that model's own defaults; ``weather_model_path`` points at
    the directory holding ``atmos_models.py`` and ``mie_q.npz``, since the weather model is
    GPL-3.0 and is not vendored into this repository.
    """

    enabled: bool = False
    type: Optional[str] = None
    weather: Optional[str] = None
    precipitation_rate: Optional[float] = None
    #: the weather model lidar return mode: "strongest" or "last".
    mode: str = "strongest"
    #: Maximum lidar range in metres.
    rmax: float = 200.0
    #: Minimum (bistatic) lidar range in metres.
    rmin: float = 1.5
    #: Beam divergence in radians.
    bdiv: float = 3.0e-3
    #: Directory containing the weather model's atmos_models.py and mie_q.npz. Defaults to the
    #: copy vendored at reality/structure/weather_model.
    weather_model_path: Optional[str] = None
    #: Which the weather model implementation to import: the vectorised torch one, or the
    #: original per-point CPU one shipped alongside it.
    weather_model_module: str = "atmos_models"


@dataclass
class NormalizationSpec:
    """Where the per-channel normalization constants come from.

    ``computed`` measures them from the configured source and target datasets;
    ``picgan_default`` uses PICGAN's original VoxelScape-fitted constants, kept so
    published runs reproduce exactly.
    """

    source: str = "computed"
    #: Frames sampled per dataset when measuring.
    frames: int = 50
    #: Seed for frame selection; None spreads the sample evenly instead.
    seed: Optional[int] = 0


@dataclass
class ModelSpec:
    """Intensity-transfer model selected from the model registry."""

    type: str = "picgan"
    #: Output activation of the target-domain generator. "tanh" is the original
    #: behaviour and confines output to +-1 sigma of the target intensity;
    #: "sigmoid" gives [0, 1] data units, lifting that ceiling.
    output_activation: str = "tanh"
    #: Weight of the distributional (Wasserstein) generator term. 0 disables it.
    lambda_wasserstein: float = 0.0
    #: Add a binary retroreflector channel to the source stack when labels allow.
    retro_channel: bool = False


@dataclass
class TrainingSpec:
    """Training hyperparameters. Defaults match PICGAN's documented settings."""

    batch_size: int = 8
    epochs: int = 200
    learning_rate: float = 1.0e-5
    lambda_cycle: float = 10.0
    lambda_physics: float = 10.0
    num_workers: int = 4
    seed: int = 42
    #: Discriminator target on real samples. 1.0 is the original behaviour; below
    #: 1.0 is one-sided label smoothing, used to weaken a dominant discriminator.
    label_smoothing_real: float = 1.0
    #: Discriminator learning rate. None follows `learning_rate`.
    disc_learning_rate: Optional[float] = None


@dataclass
class OutputSpec:
    """Where a run writes its weights and metadata sidecar."""

    checkpoint_dir: str


@dataclass
class Config:
    """A fully resolved run configuration."""

    source: DataSpec
    target: DataSpec
    task: TaskSpec
    #: Root holding one folder per dataset, e.g. ``Data/KITTI``, ``Data/CADC``.
    data_root: Optional[str] = None
    model: ModelSpec = field(default_factory=ModelSpec)
    training: TrainingSpec = field(default_factory=TrainingSpec)
    geometric_degradation: DegradationSpec = field(default_factory=DegradationSpec)
    normalization: NormalizationSpec = field(default_factory=NormalizationSpec)
    sensor: Optional[SensorSpec] = None
    output: Optional[OutputSpec] = None

    # -- construction ------------------------------------------------------- #

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Build and validate a config from a plain dict."""
        if not isinstance(data, dict):
            raise ConfigError(f"config must be a mapping, got {type(data).__name__}")
        cfg: "Config" = _build(cls, data, path="")
        cfg._resolve()
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Config":
        """Load and validate a config from a YAML file."""
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        try:
            raw = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: invalid YAML: {exc}") from None
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: config must be a mapping, got {type(raw).__name__}")
        return cls.from_dict(raw)

    # -- resolution & validation -------------------------------------------- #

    @property
    def run_name(self) -> str:
        """``<source>_to_<target>``, the conventional run/checkpoint name."""
        return f"{self.source.dataset}_to_{self.target.dataset}"

    def _resolve(self) -> None:
        """Fill in values derivable from the rest of the config."""
        if self.output is None:
            self.output = OutputSpec(checkpoint_dir=f"checkpoints/{self.run_name}")
        # A dataset with no explicit path is looked up by name under data_root.
        if self.data_root:
            for spec in (self.source, self.target):
                if not spec.path:
                    spec.path = str(Path(self.data_root) / dataset_folder(spec.dataset))

    def validate(self) -> "Config":
        """Check cross-section consistency. Raises :class:`ConfigError`."""
        if self.task.type not in TASK_TYPES:
            raise ConfigError(
                f"task.type: expected one of {list(TASK_TYPES)}, got {self.task.type!r}"
            )

        for role in ("source", "target"):
            _validate_data(getattr(self, role), role)

        deg = self.geometric_degradation
        # Sensor transfer bypasses the physics/the weather model stage; weather transfer needs
        # it, because that is what produces the physics intensity PICGAN consumes.
        if self.task.is_weather and not deg.enabled:
            raise ConfigError(
                "geometric_degradation.enabled: must be true for task.type='weather' "
                "(the degradation stage produces the physics intensity PICGAN requires)"
            )
        if not self.task.is_weather and deg.enabled:
            raise ConfigError(
                "geometric_degradation.enabled: must be false for task.type='sensor' "
                "(sensor transfer takes physics intensity from the source simulator)"
            )

        if deg.enabled:
            if deg.type is None:
                raise ConfigError(
                    "geometric_degradation.type: required when enabled, "
                    f"one of {list(DEGRADATION_TYPES)}"
                )
            if deg.type not in DEGRADATION_TYPES:
                raise ConfigError(
                    f"geometric_degradation.type: expected one of "
                    f"{list(DEGRADATION_TYPES)}, got {deg.type!r}"
                )
            if deg.weather not in WEATHER_TYPES:
                raise ConfigError(
                    f"geometric_degradation.weather: expected one of "
                    f"{list(WEATHER_TYPES)}, got {deg.weather!r}"
                )
            if deg.precipitation_rate is None:
                raise ConfigError(
                    "geometric_degradation.precipitation_rate: required when enabled"
                )
            if deg.precipitation_rate <= 0:
                raise ConfigError(
                    "geometric_degradation.precipitation_rate: must be > 0, "
                    f"got {deg.precipitation_rate}"
                )
            if deg.mode not in LIDAR_RETURN_MODES:
                raise ConfigError(
                    f"geometric_degradation.mode: expected one of "
                    f"{list(LIDAR_RETURN_MODES)}, got {deg.mode!r}"
                )
            if deg.rmax <= deg.rmin:
                raise ConfigError(
                    f"geometric_degradation: rmax ({deg.rmax}) must exceed "
                    f"rmin ({deg.rmin})"
                )
            if deg.bdiv <= 0:
                raise ConfigError(
                    f"geometric_degradation.bdiv: must be > 0, got {deg.bdiv}"
                )
            if deg.weather_model_module not in WEATHER_MODEL_MODULES:
                raise ConfigError(
                    f"geometric_degradation.weather_model_module: expected one of "
                    f"{list(WEATHER_MODEL_MODULES)}, got {deg.weather_model_module!r}"
                )

        if not self.model.type:
            raise ConfigError("model.type: must be a non-empty name")
        if self.model.output_activation not in OUTPUT_ACTIVATIONS:
            raise ConfigError(
                f"model.output_activation: expected one of {list(OUTPUT_ACTIVATIONS)}, "
                f"got {self.model.output_activation!r}"
            )
        if self.model.lambda_wasserstein < 0:
            raise ConfigError(
                f"model.lambda_wasserstein: must be >= 0, got {self.model.lambda_wasserstein}"
            )

        if self.normalization.source not in NORMALIZATION_SOURCES:
            raise ConfigError(
                f"normalization.source: expected one of {list(NORMALIZATION_SOURCES)}, "
                f"got {self.normalization.source!r}"
            )
        if self.normalization.frames <= 0:
            raise ConfigError(
                f"normalization.frames: must be > 0, got {self.normalization.frames}"
            )

        t = self.training
        for name in ("batch_size", "epochs", "num_workers", "learning_rate"):
            value = getattr(t, name)
            if name == "num_workers":
                if value < 0:
                    raise ConfigError(f"training.{name}: must be >= 0, got {value}")
            elif value <= 0:
                raise ConfigError(f"training.{name}: must be > 0, got {value}")
        for name in ("lambda_cycle", "lambda_physics"):
            if getattr(t, name) < 0:
                raise ConfigError(f"training.{name}: must be >= 0, got {getattr(t, name)}")
        if not 0.0 < t.label_smoothing_real <= 1.0:
            raise ConfigError(
                f"training.label_smoothing_real: must be in (0, 1], got "
                f"{t.label_smoothing_real}"
            )
        if t.disc_learning_rate is not None and t.disc_learning_rate <= 0:
            raise ConfigError(
                f"training.disc_learning_rate: must be > 0, got {t.disc_learning_rate}"
            )

        if self.sensor is not None:
            s = self.sensor
            if s.proj_H <= 0 or s.proj_W <= 0:
                raise ConfigError(
                    f"sensor: proj_H/proj_W must be > 0, got {s.proj_H}x{s.proj_W}"
                )
            if s.fov_up <= s.fov_down:
                raise ConfigError(
                    f"sensor: fov_up ({s.fov_up}) must be greater than fov_down ({s.fov_down})"
                )
        return self

    # -- snapshot ----------------------------------------------------------- #

    def to_dict(self) -> Dict[str, Any]:
        """Plain-data view of the resolved config (YAML-safe, round-trippable)."""
        return _to_plain(self)

    def snapshot(self, path: Union[str, Path]) -> Path:
        """Write the resolved config to ``path`` as YAML and return that path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        return path


def load_config(path: Union[str, Path]) -> Config:
    """Load and validate a run config from YAML."""
    return Config.load(path)


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


#: On-disk folder name per registered dataset, so identity reads naturally.
DATASET_FOLDERS = {
    "kitti": "KITTI", "cadc": "CADC", "nuscenes": "nuScenes",
    "boreas": "Boreas", "voxelscape": "VoxelScape",
}


def dataset_folder(name: str) -> str:
    """Folder name for a dataset under ``data_root``."""
    return DATASET_FOLDERS.get(name.lower(), name)


def _validate_data(spec: DataSpec, role: str) -> None:
    if not spec.dataset:
        raise ConfigError(f"{role}.dataset: must be a non-empty adapter name")
    if spec.split not in DATASET_SPLITS:
        raise ConfigError(
            f"{role}.split: expected one of {list(DATASET_SPLITS)}, got {spec.split!r}"
        )
    if spec.intensity_scale <= 0:
        raise ConfigError(f"{role}.intensity_scale: must be > 0, got {spec.intensity_scale}")
    if spec.dataset in SELF_DESCRIBING_DATASETS:
        for key in ("path", "format"):
            if not getattr(spec, key):
                raise ConfigError(
                    f"{role}.{key}: required for the '{spec.dataset}' adapter "
                    f"(bring-your-own data declares its own layout)"
                )
        if not spec.columns:
            raise ConfigError(
                f"{role}.columns: required for the '{spec.dataset}' adapter; "
                f"column layout is declared, never guessed"
            )


def _build(cls: type, data: Any, path: str) -> Any:
    """Recursively build a dataclass from a mapping, strictly."""
    where = path or "config"
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")

    hints = typing.get_type_hints(cls)
    known = {f.name: f for f in fields(cls)}

    unknown = [k for k in data if k not in known]
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {sorted(unknown)}; valid keys are {sorted(known)}"
        )

    kwargs: Dict[str, Any] = {}
    for name, f in known.items():
        child = f"{path}.{name}" if path else name
        if name in data:
            kwargs[name] = _convert(hints[name], data[name], child)
        elif f.default is MISSING and f.default_factory is MISSING:  # type: ignore[misc]
            raise ConfigError(f"{child}: required")
    return cls(**kwargs)


def _convert(annotation: Any, value: Any, path: str) -> Any:
    """Coerce a YAML value to its annotated type, or raise ConfigError."""
    origin = typing.get_origin(annotation)

    if origin is Union:  # Optional[X]
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if value is None:
            return None
        if len(args) == 1:
            return _convert(args[0], value, path)
        raise ConfigError(f"{path}: unsupported union type {annotation}")

    if origin in (tuple, list):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        (item_type,) = [a for a in typing.get_args(annotation) if a is not Ellipsis] or [str]
        items = [_convert(item_type, v, f"{path}[{i}]") for i, v in enumerate(value)]
        return tuple(items) if origin is tuple else items

    if dataclasses.is_dataclass(annotation):
        return _build(annotation, value, path)

    if annotation is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected a boolean, got {value!r}")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    if annotation is Any:
        return value
    raise ConfigError(f"{path}: unsupported type {annotation}")


def _to_plain(value: Any) -> Any:
    """Convert dataclasses/tuples to YAML-safe plain data, dropping ``None``."""
    if dataclasses.is_dataclass(value):
        out: Dict[str, Any] = {}
        for f in fields(value):
            v = getattr(value, f.name)
            if v is None:
                continue
            out[f.name] = _to_plain(v)
        return out
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value
