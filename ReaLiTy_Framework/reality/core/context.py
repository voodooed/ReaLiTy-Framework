"""The :class:`Sample` object that flows between pipeline stages.

Stages depend only on this contract, so any stage can be replaced
independently. Shapes follow README -> *The `Sample` contract*::

    points      : (N, C)            native columns, dataset-tagged
    range_image : (K, H, W)         range, incidence, [reflectance], intensity, mask
    phy         : (1, H, W) | None  physics-based intensity, set upstream
    mapping     : projection index map, for exact back-projection
    meta        : dataset, sensor, task, fov, intensity_scale, has_reflectance
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Tuple

import numpy as np

from reality.core.config import SensorSpec


@dataclass
class SampleMeta:
    """Provenance and layout of a sample, carried end to end."""

    dataset: str
    task: str
    sensor: Optional[str] = None
    fov: Optional[SensorSpec] = None
    intensity_scale: float = 1.0
    #: False when semantic labels are unavailable: reflectance is dropped and the
    #: PICGAN source stack is built with 2 channels instead of 3.
    has_reflectance: bool = True
    columns: Tuple[str, ...] = ()
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def source_channels(self) -> int:
        """Channel count of the PICGAN source stack implied by this sample."""
        return 3 if self.has_reflectance else 2


@dataclass
class Sample:
    """One point cloud as it moves through the pipeline."""

    points: np.ndarray
    meta: SampleMeta
    range_image: Optional[np.ndarray] = None
    #: Physics-based intensity. Produced upstream (source simulator for sensor
    #: transfer, the weather model for weather transfer) and never computed by the model.
    phy: Optional[np.ndarray] = None
    mapping: Optional[np.ndarray] = None
    #: Names of the ``range_image`` channels, in order.
    channels: Tuple[str, ...] = ()

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def image_shape(self) -> Optional[Tuple[int, int]]:
        """``(H, W)`` of the range image, or None before projection."""
        if self.range_image is None:
            return None
        return int(self.range_image.shape[-2]), int(self.range_image.shape[-1])

    @property
    def has_phy(self) -> bool:
        return self.phy is not None

    def channel(self, name: str) -> np.ndarray:
        """Return the ``(H, W)`` plane of ``range_image`` named ``name``."""
        if self.range_image is None:
            raise ValueError("sample has no range_image; run projection first")
        if name not in self.channels:
            raise KeyError(f"no channel '{name}'; available: {list(self.channels)}")
        return self.range_image[self.channels.index(name)]

    def point_column(self, name: str) -> np.ndarray:
        """Return the ``(N,)`` column of ``points`` named ``name``."""
        if name not in self.meta.columns:
            raise KeyError(f"no point column '{name}'; available: {list(self.meta.columns)}")
        return self.points[:, self.meta.columns.index(name)]

    def raw_cloud(self) -> np.ndarray:
        """The native ``(N, 4)`` x, y, z, intensity cloud, for stages like the weather model."""
        return np.column_stack([self.point_column(c) for c in ("x", "y", "z", "intensity")])

    def replace(self, **changes: Any) -> "Sample":
        """Return a copy with the given fields replaced."""
        return replace(self, **changes)

    def validate(self) -> "Sample":
        """Check the shape contract; raises ValueError on violation."""
        if self.points.ndim != 2:
            raise ValueError(f"points must be (N, C), got shape {self.points.shape}")
        if self.range_image is not None:
            if self.range_image.ndim != 3:
                raise ValueError(
                    f"range_image must be (K, H, W), got shape {self.range_image.shape}"
                )
            if self.channels and len(self.channels) != self.range_image.shape[0]:
                raise ValueError(
                    f"channels {list(self.channels)} do not match range_image "
                    f"with {self.range_image.shape[0]} planes"
                )
        if self.phy is not None:
            if self.phy.ndim != 3 or self.phy.shape[0] != 1:
                raise ValueError(f"phy must be (1, H, W), got shape {self.phy.shape}")
            if self.range_image is not None and self.phy.shape[1:] != self.range_image.shape[1:]:
                raise ValueError(
                    f"phy spatial shape {self.phy.shape[1:]} != range_image "
                    f"{self.range_image.shape[1:]}"
                )
        return self
