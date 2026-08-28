"""GeometricDegradation — the pluggable weather stage.

A degradation takes a clear-weather point cloud and returns a degraded one whose
physics-based intensity is populated, so that everything downstream (projection,
PICGAN, back-projection) is unchanged whichever implementation is selected
(README -> *Extensibility*).

Ordering note: degradation runs on the **3D cloud, before projection**. the weather model
models scattering, point drop and range attenuation along each beam and its API
consumes an ``(N, 4)`` cloud, so it cannot operate on a range image. The physics
intensity it produces is carried as a point column and becomes ``Sample.phy``
when the degraded cloud is projected.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import numpy as np

from reality.core.config import Config
from reality.core.context import Sample
from reality.core.pipeline import Stage

#: Column carrying the physics-based intensity a degradation produces.
PHYSICS_COLUMN = "physics_intensity"
#: Column carrying the degradation's per-point outcome label.
LABEL_COLUMN = "degradation_label"


class DegradationError(RuntimeError):
    """Raised when a degradation cannot be applied to a sample."""


class GeometricDegradation(Stage):
    """Base class for weather degradation plugins."""

    name = "degradation"

    def __init__(self, config: Optional[Config] = None) -> None:
        super().__init__(config)
        self.spec = config.geometric_degradation if config is not None else None

    @abstractmethod
    def apply(self, sample: Sample) -> Sample:
        """Return a degraded Sample carrying a physics-intensity column."""

    @property
    def available(self) -> bool:
        """Whether this degradation can actually run (its model is present)."""
        return True

    # -- shared helpers -------------------------------------------------------- #

    @staticmethod
    def attach(sample: Sample, xyz: np.ndarray, physics_intensity: np.ndarray,
               labels: Optional[np.ndarray] = None) -> Sample:
        """Build the degraded Sample, preserving the source's other columns.

        Geometry and intensity are replaced with the degraded values; columns the
        source adapter added (semantic label, LUT reflectance) are carried through
        unchanged, because a material's reflectance does not change with weather.
        The physics intensity is appended so projection turns it into ``phy``.
        """
        columns = list(sample.meta.columns)
        points = np.array(sample.points, dtype=np.float32, copy=True)
        if xyz.shape != (points.shape[0], 3):
            raise DegradationError(
                f"degraded geometry must be ({points.shape[0]}, 3), got {xyz.shape}"
            )
        points[:, :3] = xyz
        points[:, columns.index("intensity")] = physics_intensity

        extra, names = [], []
        if PHYSICS_COLUMN in columns:
            points[:, columns.index(PHYSICS_COLUMN)] = physics_intensity
        else:
            extra.append(physics_intensity)
            names.append(PHYSICS_COLUMN)
        if labels is not None:
            if LABEL_COLUMN in columns:
                points[:, columns.index(LABEL_COLUMN)] = labels
            else:
                extra.append(labels)
                names.append(LABEL_COLUMN)
        if extra:
            points = np.column_stack([points] + extra).astype(np.float32)
            columns += names

        meta = sample.meta
        meta.columns = tuple(columns)
        meta.extra = dict(meta.extra)
        return sample.replace(points=points.astype(np.float32), meta=meta)
