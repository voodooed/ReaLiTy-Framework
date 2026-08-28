"""Range-image intensity back to 3D, by index. Geometry is never recomputed.

The generated intensity image is written onto the points the projection recorded
in ``Sample.mapping``, so x, y and z come back exactly as they went in -- the
output stays a drop-in replacement for the native cloud.

**Alignment is checked, not assumed.** A mapping built from one cloud and applied
to another silently scatters intensities onto the wrong points; that is the
failure this module is written to make impossible. ``project()`` records the
length of the cloud it indexed in ``meta.extra['projection']['n_points']`` and
:func:`backproject` refuses any cloud of a different length. For the weather path
this means the cloud passed here must be the same the weather model-degraded cloud
that was projected, not the clear-weather original.

**Dropped points.** A point receives no intensity when it was outside the vertical
field of view, had zero range, or lost its pixel to a nearer return. Those points
are never silently written and never collapsed to the origin: the ``fill`` policy
says what happens to them and a boolean ``written`` mask says which they were.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from reality.core.config import Config
from reality.core.context import Sample
from reality.core.pipeline import Stage
from reality.core.registry import TRANSFORMS


class BackProjectionError(ValueError):
    """Raised when an intensity image cannot be mapped back onto a cloud."""


@dataclass
class BackProjection:
    """Result of writing an intensity image back onto a cloud."""

    #: The cloud with its intensity column replaced. Geometry is untouched.
    points: np.ndarray
    #: True for points that received a generated intensity.
    written: np.ndarray

    @property
    def n_written(self) -> int:
        return int(self.written.sum())

    @property
    def n_dropped(self) -> int:
        return int((~self.written).sum())


def backproject(sample: Sample, intensity_image: np.ndarray,
                cloud: Optional[np.ndarray] = None,
                fill: Union[str, float] = "keep",
                intensity_column: str = "intensity") -> BackProjection:
    """Write ``intensity_image`` onto the points ``sample.mapping`` indexes.

    ``fill`` decides what happens to points that own no pixel: ``"keep"`` leaves
    the original intensity in place, ``"nan"`` marks them as having no generated
    value, and a float writes that constant.
    """
    if sample.mapping is None:
        raise BackProjectionError(
            f"{sample.meta.dataset}: sample has no mapping; project it first"
        )
    points = np.array(sample.points if cloud is None else cloud, dtype=np.float32, copy=True)
    if points.ndim != 2:
        raise BackProjectionError(f"cloud must be (N, C), got shape {points.shape}")

    recorded = sample.meta.extra.get("projection", {})
    expected = recorded.get("n_points")
    if expected is not None and points.shape[0] != expected:
        raise BackProjectionError(
            f"mapping was built from a cloud of {expected} points but this cloud has "
            f"{points.shape[0]}. The cloud back-projected must be the same one that was "
            f"projected, or the indices point at different points."
        )
    if intensity_column not in sample.meta.columns:
        raise BackProjectionError(
            f"no '{intensity_column}' column to write into; columns are "
            f"{list(sample.meta.columns)}"
        )

    image = np.asarray(intensity_image, dtype=np.float32)
    if image.ndim == 3:
        if image.shape[0] != 1:
            raise BackProjectionError(
                f"intensity image must be (1, H, W) or (H, W), got {image.shape}"
            )
        image = image[0]
    if image.shape != sample.mapping.shape:
        raise BackProjectionError(
            f"intensity image {image.shape} does not match the mapping "
            f"{sample.mapping.shape}"
        )

    mapping = sample.mapping
    occupied = mapping >= 0
    owners = mapping[occupied]
    if owners.size and int(owners.max()) >= points.shape[0]:
        raise BackProjectionError(
            f"mapping indexes point {int(owners.max())} but the cloud has "
            f"{points.shape[0]} points"
        )

    column = list(sample.meta.columns).index(intensity_column)
    written = np.zeros(points.shape[0], dtype=bool)
    points[owners, column] = image[occupied]
    written[owners] = True

    if fill != "keep":
        value = np.nan if fill == "nan" else float(fill)
        points[~written, column] = value
    return BackProjection(points=points, written=written)


@TRANSFORMS.register("backprojection")
class BackProjectionStage(Stage):
    """Pipeline stage: writes the generated intensity back onto the cloud."""

    name = "backprojection"

    def __init__(self, config: Optional[Config] = None,
                 fill: Union[str, float] = "keep") -> None:
        super().__init__(config)
        self.fill = fill

    def apply(self, sample: Sample) -> Sample:
        """Back-project the sample's generated intensity image.

        Expects ``meta.extra['generated_intensity']`` to hold the model output.
        """
        generated = sample.meta.extra.get("generated_intensity")
        if generated is None:
            raise BackProjectionError(
                f"{sample.meta.dataset}: no generated intensity to back-project; "
                f"the model stage must run first"
            )
        result = backproject(sample, generated, fill=self.fill)
        meta = sample.meta
        meta.extra = dict(meta.extra)
        meta.extra["backprojection"] = {"n_written": result.n_written,
                                        "n_dropped": result.n_dropped}
        return sample.replace(points=result.points, meta=meta)
