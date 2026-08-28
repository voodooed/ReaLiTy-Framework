"""DatasetAdapter — the common interface every dataset is loaded through.

An adapter declares its column layout, intensity scale, sensor geometry and
whether semantic labels are available, then turns a frame into a
:class:`~reality.core.context.Sample`. Adding a dataset means adding an adapter
and a config entry, never editing pipeline code.

Reflectance decision (README -> *Reflectance handling*), resolved here once:
labels present -> reflectance is mapped from them and appended as a point column,
``meta.has_reflectance = True``, PICGAN gets a 3-channel source stack. Labels
absent -> no reflectance column, ``has_reflectance = False``, 2-channel source.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from reality.core.config import DataSpec, SensorSpec
from reality.core.context import Sample, SampleMeta
from reality.physics.reflectance import ReflectanceLUT, default_lut

#: Columns every adapter must expose, in this order, so downstream stages
#: (projection, the weather model) can rely on them regardless of the dataset.
REQUIRED_COLUMNS: Tuple[str, ...] = ("x", "y", "z", "intensity")

#: Point-cloud formats the released layout accepts.
CLOUD_SUFFIXES: Tuple[str, ...] = (".npy", ".bin")
#: Split folders in the released layout.
SPLITS: Tuple[str, ...] = ("train", "test")


class DatasetError(ValueError):
    """Raised for a missing path, an undeclared layout or a malformed frame."""


@dataclass(frozen=True)
class FrameRef:
    """A locatable frame: where its cloud is, and its labels if any."""

    id: str
    path: Path
    label_path: Optional[Path] = None
    extra: Dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def has_labels(self) -> bool:
        return self.label_path is not None and Path(self.label_path).is_file()


class DatasetAdapter(ABC):
    """Base class for dataset adapters."""

    #: Declared native column layout. Never inferred from file size.
    columns: Tuple[str, ...] = REQUIRED_COLUMNS
    #: Divisor that brings the native intensity onto a 0-1 scale.
    intensity_scale: float = 1.0
    #: Sensor geometry used when the config does not override it.
    default_sensor: Optional[SensorSpec] = None
    #: Which class map in the reflectance LUT this dataset's labels use.
    label_scheme: str = "semantickitti"
    #: Registry name, set by subclasses.
    name: str = "base"

    def __init__(self, spec: DataSpec, sensor: Optional[SensorSpec] = None,
                 task: str = "sensor", lut: Optional[ReflectanceLUT] = None,
                 split: Optional[str] = None) -> None:
        self.spec = spec
        self.task = task
        self.split = split or getattr(spec, "split", "train")
        self.sensor = sensor or self.default_sensor
        self._lut = lut
        if spec.columns:
            self.columns = tuple(spec.columns)
        if spec.intensity_scale is not None:
            self.intensity_scale = float(spec.intensity_scale)
        missing = [c for c in REQUIRED_COLUMNS if c not in self.columns]
        if missing:
            raise DatasetError(
                f"{type(self).__name__}: declared columns {list(self.columns)} are missing "
                f"required column(s) {missing}"
            )
        self.root = Path(spec.path) if spec.path else None

    # -- the released layout ---------------------------------------------------- #

    @property
    def split_dir(self) -> Optional[Path]:
        """``<root>/<split>`` when the dataset uses the released layout."""
        if self.root is None:
            return None
        candidate = self.root / self.split
        return candidate if candidate.is_dir() else None

    @property
    def uses_released_layout(self) -> bool:
        """True when this dataset is organised as ``<root>/train`` and ``<root>/test``."""
        return self.split_dir is not None

    @property
    def label_root(self) -> Optional[Path]:
        """Where labels live: declared in config, or ``<root>/labels`` by convention."""
        if self.spec.labels is not None:
            return Path(self.spec.labels.path)
        if self.root is not None and (self.root / "labels").is_dir():
            return self.root / "labels"
        return None

    def list_split_frames(self) -> List[FrameRef]:
        """Enumerate clouds in the split folder, by convention rather than by dataset.

        Any ``.npy`` or ``.bin`` under ``<root>/<split>`` is a frame. Labels, when
        the config declares them, are matched by stem under ``labels.path`` (either
        directly or in a matching split sub-folder).
        """
        directory = self.split_dir
        files = sorted(p for p in directory.rglob("*")
                       if p.suffix.lower() in CLOUD_SUFFIXES and p.is_file())
        label_root = self.label_root
        frames = []
        for path in files:
            label_path = None
            if label_root is not None:
                for candidate in (label_root / self.split / f"{path.stem}.label",
                                  label_root / f"{path.stem}.label"):
                    if candidate.is_file():
                        label_path = candidate
                        break
            frames.append(FrameRef(id=path.stem, path=path, label_path=label_path,
                                   extra={"split": self.split}))
        return frames

    def load_split_points(self, frame: FrameRef) -> np.ndarray:
        """Read a released-layout cloud, honouring the declared column layout."""
        if frame.path.suffix.lower() == ".npy":
            return read_npy(frame.path, len(self.columns))
        return read_bin(frame.path, len(self.columns))

    # -- to implement -------------------------------------------------------- #

    @abstractmethod
    def list_frames(self) -> List[FrameRef]:
        """Enumerate the frames this adapter can load, in a stable order."""

    @abstractmethod
    def load_points(self, frame: FrameRef) -> np.ndarray:
        """Read one frame's native point array, shape ``(N, len(self.columns))``."""

    # -- provided ------------------------------------------------------------ #

    @property
    def lut(self) -> ReflectanceLUT:
        """Reflectance LUT, loaded lazily so unlabelled datasets never touch it."""
        if self._lut is None:
            self._lut = default_lut()
        return self._lut

    def load_labels(self, frame: FrameRef) -> Optional[np.ndarray]:
        """Read per-point semantic class ids, or None when unavailable."""
        return None

    def reflectance_from_labels(self, labels: np.ndarray) -> np.ndarray:
        """Map semantic class ids to reflectance in [0, 1]."""
        return self.lut.lookup(labels, dataset=self.label_scheme)

    def load_sample(self, frame: FrameRef) -> Sample:
        """Load a frame into a :class:`Sample`, resolving the reflectance path."""
        points = np.asarray(self.load_points(frame), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != len(self.columns):
            raise DatasetError(
                f"{self.name}: {frame.path} produced shape {points.shape}, expected "
                f"(N, {len(self.columns)}) for declared columns {list(self.columns)}"
            )
        columns = list(self.columns)

        # Normalise intensity onto 0-1 without disturbing the other columns.
        if self.intensity_scale != 1.0:
            points = points.copy()
            points[:, columns.index("intensity")] /= self.intensity_scale

        labels = self.load_labels(frame)
        if labels is not None:
            labels = np.asarray(labels).reshape(-1)
            if labels.shape[0] != points.shape[0]:
                raise DatasetError(
                    f"{self.name}: {frame.id} has {points.shape[0]} points but "
                    f"{labels.shape[0]} labels"
                )
            reflectance = self.reflectance_from_labels(labels)
            points = np.column_stack([points, labels.astype(np.float32),
                                      reflectance.astype(np.float32)])
            columns += ["label", "reflectance"]

        meta = SampleMeta(
            dataset=self.name, task=self.task, sensor=self.name, fov=self.sensor,
            intensity_scale=self.intensity_scale, has_reflectance=labels is not None,
            columns=tuple(columns), extra={"frame_id": frame.id, "path": str(frame.path)},
        )
        return Sample(points=points.astype(np.float32), meta=meta).validate()

    def __len__(self) -> int:
        return len(self.list_frames())

    def __getitem__(self, index: int) -> Sample:
        return self.load_sample(self.list_frames()[index])

    def __repr__(self) -> str:
        return f"{type(self).__name__}(root={self.root}, columns={list(self.columns)})"


# --------------------------------------------------------------------------- #
# Shared readers
# --------------------------------------------------------------------------- #


def read_bin(path: Path, n_columns: int, dtype=np.float32) -> np.ndarray:
    """Read a headerless binary cloud with a *declared* column count.

    The column count is never inferred from file size; a file that is not an exact
    multiple of the declared width is an error, not a cue to guess a layout.
    """
    raw = np.fromfile(str(path), dtype=dtype)
    if raw.size == 0:
        raise DatasetError(f"{path}: file is empty")
    if raw.size % n_columns:
        raise DatasetError(
            f"{path}: {raw.size} {np.dtype(dtype).name} values is not a multiple of the "
            f"declared {n_columns} columns"
        )
    return raw.reshape(-1, n_columns)


def read_npy(path: Path, n_columns: int) -> np.ndarray:
    """Read an ``.npy`` cloud and check it against the declared column count."""
    arr = np.load(str(path))
    if arr.ndim != 2 or arr.shape[1] != n_columns:
        raise DatasetError(
            f"{path}: shape {arr.shape} does not match the declared {n_columns} columns"
        )
    return arr


def read_semantickitti_labels(path: Path) -> np.ndarray:
    """Read a SemanticKITTI ``.label`` file: low 16 bits semantic, high 16 instance."""
    raw = np.fromfile(str(path), dtype=np.uint32)
    return (raw & 0xFFFF).astype(np.int32)
