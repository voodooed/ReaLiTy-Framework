"""Spherical projection: an (N, C) point cloud to a range-image stack.

Channel order follows PICGAN's confirmed layout (README -> *The Sample contract*)::

    range, incidence, [reflectance], intensity, mask

Reflectance is present only when the source has semantic labels
(``meta.has_reflectance``); the PICGAN adapter reads the channels it needs by
name, so the extra intensity and mask planes are carried without disturbing it.

Sensor geometry comes from ``meta.fov`` or the config, never from a constant here.
The values used for the real datasets were measured in : KITTI HDL-64E
(64 x 1024, +3.0/-25.0 deg) and CADC VLP-32C (32 x 1024, +15.0/-25.0 deg).

**Index mapping.** ``Sample.mapping`` is an ``(H, W)`` int64 image holding, for
each pixel, the index of the point that owns it in the cloud that was projected,
or -1 where no point landed. Those indices are always into the *full* input
cloud: points rejected for zero range or for falling outside the vertical field
of view are excluded by masking, never by compacting the array, so an index never
shifts. ``meta.extra['projection']`` records the length of the cloud the mapping
was built against, and back-projection refuses a cloud of a different length.

**Incidence.** Taken from the data when the dataset provides it (a simulator
export such as VoxelScape carries it natively). Otherwise estimated from local
surface geometry using the convention already established in the author's
physradar project (``geometry/candidates.py``): the incidence cosine is
``|dot(surface_normal, unit ray from sensor to point)|`` clipped to [0, 1], with
pixels whose neighbourhood cannot define a plane filled by the frame median. On
an organised range image the local neighbourhood is the pixel's four neighbours,
so the normal is the cross product of the horizontal and vertical 3D differences
rather than a PCA over a kNN ball.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from reality.core.config import Config, SensorSpec
from reality.core.context import Sample
from reality.core.registry import TRANSFORMS
from reality.preprocessing.base import Transform

#: Minimum range for a point to be projectable at all.
EPS_RANGE = 1e-6
#: Value used for an incidence cosine that cannot be estimated anywhere in a frame.
NEUTRAL_INCIDENCE = 0.5


class ProjectionError(ValueError):
    """Raised when a Sample cannot be projected with the geometry it declares."""


def spherical_pixel_coordinates(xyz: np.ndarray, sensor: SensorSpec
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map points to fractional pixel coordinates.

    Returns ``(depth, row, col, in_fov)``. ``row``/``col`` are integer pixel
    indices; ``in_fov`` is False for points outside the vertical field of view,
    which are dropped rather than clamped onto the edge rows -- clamping would
    pile out-of-range returns onto the first and last rows and hand them another
    point's intensity on the way back.
    """
    depth = np.linalg.norm(xyz, axis=1)
    valid = depth > EPS_RANGE

    safe_depth = np.where(valid, depth, 1.0)
    yaw = -np.arctan2(xyz[:, 1], xyz[:, 0])
    pitch = np.arcsin(np.clip(xyz[:, 2] / safe_depth, -1.0, 1.0))

    fov_up = np.radians(sensor.fov_up)
    fov_down = np.radians(sensor.fov_down)
    fov = fov_up - fov_down
    if fov <= 0:
        raise ProjectionError(
            f"sensor fov_up ({sensor.fov_up}) must exceed fov_down ({sensor.fov_down})"
        )

    u = 0.5 * (yaw / np.pi + 1.0)          # [0, 1] across azimuth
    v = 1.0 - (pitch - fov_down) / fov     # [0, 1] top to bottom

    in_fov = valid & (v >= 0.0) & (v < 1.0)
    col = np.clip(np.floor(u * sensor.proj_W), 0, sensor.proj_W - 1).astype(np.int64)
    row = np.clip(np.floor(v * sensor.proj_H), 0, sensor.proj_H - 1).astype(np.int64)
    return depth, row, col, in_fov


def build_index_map(depth: np.ndarray, row: np.ndarray, col: np.ndarray,
                    in_fov: np.ndarray, sensor: SensorSpec) -> np.ndarray:
    """Resolve which point owns each pixel; nearest wins.

    Indices are positions in the input cloud, so the map stays aligned with the
    array it was built from. Empty pixels hold -1.
    """
    mapping = np.full((sensor.proj_H, sensor.proj_W), -1, dtype=np.int64)
    candidates = np.nonzero(in_fov)[0]
    if candidates.size == 0:
        return mapping
    # Write farthest first so the nearest point ends up owning a contested pixel.
    order = candidates[np.argsort(-depth[candidates], kind="stable")]
    mapping[row[order], col[order]] = order
    return mapping


def estimate_incidence(xyz_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Incidence cosine per pixel from the organised neighbourhood.

    ``|dot(n, r)|`` where ``n`` is the local surface normal and ``r`` the unit ray
    from the sensor to the point. The absolute value follows the physradar
    convention: a surface and its flipped normal describe the same geometry, and
    the normal's sign is arbitrary.
    """
    filled = np.where(mask[None, :, :], xyz_image, 0.0)

    du = np.roll(filled, -1, axis=2) - np.roll(filled, 1, axis=2)   # along azimuth
    dv = np.roll(filled, -1, axis=1) - np.roll(filled, 1, axis=1)   # along elevation
    neighbours = (mask & np.roll(mask, -1, axis=1) & np.roll(mask, 1, axis=1)
                  & np.roll(mask, -1, axis=0) & np.roll(mask, 1, axis=0))
    # Elevation does not wrap: the top and bottom rows have no vertical neighbour.
    neighbours[0, :] = False
    neighbours[-1, :] = False

    normal = np.cross(du, dv, axis=0)
    norm = np.linalg.norm(normal, axis=0)
    ray_norm = np.linalg.norm(filled, axis=0)
    usable = neighbours & (norm > EPS_RANGE) & (ray_norm > EPS_RANGE)

    safe_norm = np.where(usable, norm, 1.0)
    safe_ray = np.where(usable, ray_norm, 1.0)
    cosine = np.abs(np.sum((normal / safe_norm) * (filled / safe_ray), axis=0))
    cosine = np.clip(cosine, 0.0, 1.0)

    # Pixels whose neighbourhood cannot define a plane take the frame median,
    # which invents neither a grazing nor a face-on incidence.
    estimated = usable & mask
    fill = float(np.median(cosine[estimated])) if estimated.any() else NEUTRAL_INCIDENCE
    return np.where(estimated, cosine, fill).astype(np.float32) * mask


def project(sample: Sample, sensor: Optional[SensorSpec] = None) -> Sample:
    """Project a Sample into a range-image stack, preserving the index mapping."""
    sensor = sensor or sample.meta.fov
    if sensor is None:
        raise ProjectionError(
            f"{sample.meta.dataset}: no sensor geometry. Declare it in config "
            f"(sensor: {{proj_H, proj_W, fov_up, fov_down}}) or in the adapter."
        )
    xyz = np.column_stack([sample.point_column(c) for c in ("x", "y", "z")]).astype(np.float64)
    depth, row, col, in_fov = spherical_pixel_coordinates(xyz, sensor)
    mapping = build_index_map(depth, row, col, in_fov, sensor)

    occupied = mapping >= 0
    owner = np.where(occupied, mapping, 0)
    mask = occupied.astype(np.float32)

    def plane(values: np.ndarray) -> np.ndarray:
        return (np.asarray(values)[owner] * mask).astype(np.float32)

    xyz_image = np.stack([plane(xyz[:, i]) for i in range(3)])

    channels = ["range", "incidence"]
    planes = [plane(depth), None]  # incidence filled in below

    if "incidence" in sample.meta.columns:
        # A simulator that exports incidence is authoritative; do not re-estimate.
        planes[1] = plane(sample.point_column("incidence"))
    else:
        planes[1] = estimate_incidence(xyz_image, occupied)

    if sample.meta.has_reflectance:
        if "reflectance" not in sample.meta.columns:
            raise ProjectionError(
                f"{sample.meta.dataset}: meta.has_reflectance is True but the cloud has "
                f"no 'reflectance' column; columns are {list(sample.meta.columns)}"
            )
        channels.append("reflectance")
        planes.append(plane(sample.point_column("reflectance")))

    channels.append("intensity")
    planes.append(plane(sample.point_column("intensity")))
    channels.append("mask")
    planes.append(mask)

    phy = None
    if "physics_intensity" in sample.meta.columns:
        # Sensor transfer: the physics intensity comes from the source simulator.
        phy = plane(sample.point_column("physics_intensity"))[None, :, :]

    meta = sample.meta
    meta.fov = sensor
    meta.extra = dict(meta.extra)
    meta.extra["projection"] = {
        "n_points": int(sample.points.shape[0]),
        "shape": (sensor.proj_H, sensor.proj_W),
        "row": row, "col": col, "in_fov": in_fov,
        "n_projected": int(occupied.sum()),
        "n_dropped": int(sample.points.shape[0] - occupied.sum()),
    }
    return Sample(
        points=sample.points, meta=meta,
        range_image=np.stack(planes).astype(np.float32),
        phy=phy if phy is not None else sample.phy,
        mapping=mapping, channels=tuple(channels),
    ).validate()


@TRANSFORMS.register("projection")
class SphericalProjection(Transform):
    """Pipeline stage wrapping :func:`project`."""

    name = "projection"

    def __init__(self, config: Optional[Config] = None,
                 sensor: Optional[SensorSpec] = None) -> None:
        super().__init__(config)
        self.sensor = sensor or (config.sensor if config is not None else None)

    def apply(self, sample: Sample) -> Sample:
        return project(sample, self.sensor)
