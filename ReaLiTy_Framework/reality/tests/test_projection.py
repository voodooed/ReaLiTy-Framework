"""Spherical projection: geometry, channel order, and index-map integrity."""

import numpy as np
import pytest

from reality.core.config import SensorSpec
from reality.core.context import Sample, SampleMeta
from reality.preprocessing.projection import (
    ProjectionError,
    SphericalProjection,
    estimate_incidence,
    project,
    spherical_pixel_coordinates,
)

KITTI_SENSOR = SensorSpec(proj_H=64, proj_W=1024, fov_up=3.0, fov_down=-25.0)
CADC_SENSOR = SensorSpec(proj_H=32, proj_W=1024, fov_up=15.0, fov_down=-25.0)


def make_cloud(n=2000, seed=0, columns=("x", "y", "z", "intensity"),
               has_reflectance=False, sensor=KITTI_SENSOR, xyz=None):
    """A synthetic cloud inside the sensor's field of view."""
    rng = np.random.default_rng(seed)
    if xyz is None:
        # Spread in azimuth and range, with elevations that mostly land in FOV.
        azimuth = rng.uniform(-np.pi, np.pi, n)
        radius = rng.uniform(3.0, 50.0, n)
        elevation = np.radians(rng.uniform(sensor.fov_down + 1, sensor.fov_up - 0.5, n))
        xyz = np.column_stack([
            radius * np.cos(elevation) * np.cos(azimuth),
            radius * np.cos(elevation) * np.sin(azimuth),
            radius * np.sin(elevation),
        ])
    n = len(xyz)
    extras = [rng.random(n) for _ in range(len(columns) - 3)]
    points = np.column_stack([xyz] + extras).astype(np.float32)
    meta = SampleMeta(dataset="synthetic", task="sensor", fov=sensor,
                      columns=tuple(columns), has_reflectance=has_reflectance)
    return Sample(points=points, meta=meta)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def test_pixel_coordinates_span_the_image():
    sample = make_cloud(5000)
    xyz = sample.points[:, :3].astype(np.float64)
    depth, row, col, in_fov = spherical_pixel_coordinates(xyz, KITTI_SENSOR)
    assert depth.shape == row.shape == col.shape == in_fov.shape == (5000,)
    assert row[in_fov].min() >= 0 and row[in_fov].max() < KITTI_SENSOR.proj_H
    assert col[in_fov].min() >= 0 and col[in_fov].max() < KITTI_SENSOR.proj_W
    assert np.allclose(depth, np.linalg.norm(xyz, axis=1))


def test_elevation_maps_top_row_to_fov_up():
    """A point at the top of the FOV lands on row 0, one at the bottom on row H-1."""
    top = np.array([[10.0, 0.0, 10.0 * np.tan(np.radians(2.9))]])
    bottom = np.array([[10.0, 0.0, 10.0 * np.tan(np.radians(-24.9))]])
    _, row_top, _, ok_top = spherical_pixel_coordinates(top, KITTI_SENSOR)
    _, row_bottom, _, ok_bottom = spherical_pixel_coordinates(bottom, KITTI_SENSOR)
    assert ok_top[0] and ok_bottom[0]
    assert row_top[0] == 0
    assert row_bottom[0] == KITTI_SENSOR.proj_H - 1


def test_points_outside_the_vertical_fov_are_dropped_not_clamped():
    """Clamping would pile out-of-range returns onto the edge rows."""
    inside = np.array([[10.0, 0.0, 0.0]])
    above = np.array([[10.0, 0.0, 10.0 * np.tan(np.radians(30.0))]])
    below = np.array([[10.0, 0.0, -10.0 * np.tan(np.radians(60.0))]])
    xyz = np.vstack([inside, above, below])
    _, _, _, in_fov = spherical_pixel_coordinates(xyz, KITTI_SENSOR)
    assert in_fov.tolist() == [True, False, False]

    sample = make_cloud(xyz=xyz)
    projected = project(sample)
    assert (projected.mapping == 0).sum() == 1, "only the in-FOV point owns a pixel"
    assert 1 not in projected.mapping and 2 not in projected.mapping


def test_zero_range_points_are_dropped():
    xyz = np.vstack([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    projected = project(make_cloud(xyz=xyz))
    assert 0 not in projected.mapping
    assert (projected.mapping == 1).sum() == 1


def test_nearest_point_wins_a_contested_pixel():
    """Two points on the same ray: the closer one owns the pixel."""
    far = np.array([[40.0, 0.0, 0.0]])
    near = np.array([[10.0, 0.0, 0.0]])
    projected = project(make_cloud(xyz=np.vstack([far, near])))
    assert 1 in projected.mapping and 0 not in projected.mapping
    row, col = np.argwhere(projected.mapping == 1)[0]
    assert projected.channel("range")[row, col] == pytest.approx(10.0)


def test_sensor_geometry_is_declared_not_assumed():
    sample = make_cloud(100)
    sample.meta.fov = None
    with pytest.raises(ProjectionError, match="no sensor geometry"):
        project(sample)


def test_invalid_fov_is_rejected():
    bad = SensorSpec(proj_H=64, proj_W=1024, fov_up=-25.0, fov_down=3.0)
    with pytest.raises(ProjectionError, match="fov_up"):
        spherical_pixel_coordinates(np.array([[1.0, 0.0, 0.0]]), bad)


def test_cadc_geometry_produces_a_32_row_image():
    projected = project(make_cloud(3000, sensor=CADC_SENSOR), CADC_SENSOR)
    assert projected.range_image.shape[1:] == (32, 1024)
    assert projected.mapping.shape == (32, 1024)


# --------------------------------------------------------------------------- #
# The index map -- the guardrail
# --------------------------------------------------------------------------- #


def test_mapping_indexes_the_full_cloud_not_a_filtered_one():
    """Indices must address the input array, even with dropped points ahead of them.

    A mapping built from a compacted cloud would be shifted by the number of
    points removed; every index here is checked against the original array.
    """
    # 50 unprojectable points first, so any compaction shifts indices by 50.
    dead = np.zeros((50, 3))
    live = make_cloud(500, seed=4).points[:, :3].astype(np.float64)
    sample = make_cloud(xyz=np.vstack([dead, live]))
    projected = project(sample)

    owners = projected.mapping[projected.mapping >= 0]
    assert owners.min() >= 50, "no dead point may own a pixel"
    assert owners.max() < sample.points.shape[0]

    # Each pixel's range must equal the range of the point the mapping names.
    rows, cols = np.nonzero(projected.mapping >= 0)
    for row, col in list(zip(rows, cols))[:200]:
        owner = projected.mapping[row, col]
        expected = np.linalg.norm(sample.points[owner, :3])
        assert projected.channel("range")[row, col] == pytest.approx(expected, rel=1e-5)


def test_every_channel_reads_from_the_owning_point():
    """All planes are gathered with the same index, so channels stay in register."""
    sample = make_cloud(1500, seed=6, columns=("x", "y", "z", "intensity", "reflectance"),
                        has_reflectance=True)
    projected = project(sample)
    rows, cols = np.nonzero(projected.mapping >= 0)
    owners = projected.mapping[rows, cols]
    assert np.allclose(projected.channel("intensity")[rows, cols],
                       sample.point_column("intensity")[owners], atol=1e-6)
    assert np.allclose(projected.channel("reflectance")[rows, cols],
                       sample.point_column("reflectance")[owners], atol=1e-6)


def test_projection_records_what_it_dropped():
    sample = make_cloud(2000, seed=7)
    stats = project(sample).meta.extra["projection"]
    assert stats["n_points"] == 2000
    assert stats["n_projected"] + stats["n_dropped"] == 2000
    assert stats["shape"] == (64, 1024)


def test_mask_marks_exactly_the_occupied_pixels():
    projected = project(make_cloud(2000, seed=8))
    assert np.array_equal(projected.channel("mask") > 0, projected.mapping >= 0)


def test_empty_pixels_are_zero_not_stale():
    projected = project(make_cloud(300, seed=9))
    empty = projected.mapping < 0
    for channel in projected.channels:
        if channel == "incidence":
            continue  # filled with the frame median by design
        assert np.all(projected.channel(channel)[empty] == 0.0)


# --------------------------------------------------------------------------- #
# Channels
# --------------------------------------------------------------------------- #


def test_channel_order_with_reflectance():
    sample = make_cloud(500, columns=("x", "y", "z", "intensity", "reflectance"),
                        has_reflectance=True)
    projected = project(sample)
    assert projected.channels == ("range", "incidence", "reflectance", "intensity", "mask")
    assert projected.range_image.shape[0] == 5


def test_channel_order_without_reflectance():
    projected = project(make_cloud(500, has_reflectance=False))
    assert projected.channels == ("range", "incidence", "intensity", "mask")
    assert projected.range_image.shape[0] == 4


def test_reflectance_claimed_but_absent_is_an_error():
    sample = make_cloud(100, has_reflectance=True)  # no reflectance column
    with pytest.raises(ProjectionError, match="no 'reflectance' column"):
        project(sample)


def test_simulator_physics_intensity_becomes_phy():
    """Sensor transfer: phy comes from the source simulator, not from the model."""
    sample = make_cloud(800, columns=("x", "y", "z", "intensity", "physics_intensity"))
    projected = project(sample)
    assert projected.phy is not None
    assert projected.phy.shape == (1, 64, 1024)
    rows, cols = np.nonzero(projected.mapping >= 0)
    owners = projected.mapping[rows, cols]
    assert np.allclose(projected.phy[0][rows, cols],
                       sample.point_column("physics_intensity")[owners], atol=1e-6)


def test_no_phy_when_the_source_does_not_provide_it():
    """KITTI has no physics intensity; the weather path fills it in """
    assert project(make_cloud(200)).phy is None


# --------------------------------------------------------------------------- #
# Incidence
# --------------------------------------------------------------------------- #


def test_incidence_matches_the_analytic_value_on_a_ground_plane():
    """For a horizontal plane the incidence cosine is |z| / range."""
    grid = np.linspace(-20, 20, 260)
    xs, ys = np.meshgrid(grid, grid)
    xyz = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, -2.0)])
    projected = project(make_cloud(xyz=xyz))

    occupied = projected.mapping >= 0
    interior = occupied.copy()
    interior[0, :] = interior[-1, :] = False  # edge rows have no vertical neighbour
    estimated = projected.channel("incidence")[interior]
    analytic = 2.0 / projected.channel("range")[interior]
    assert np.abs(estimated - analytic).mean() < 0.1


def test_incidence_is_bounded_and_finite():
    projected = project(make_cloud(4000, seed=11))
    incidence = projected.channel("incidence")
    assert np.isfinite(incidence).all()
    assert incidence.min() >= 0.0 and incidence.max() <= 1.0


def test_incidence_falls_back_to_the_frame_median_where_undefined():
    """A cloud too sparse to define any plane must not invent a grazing angle."""
    xyz = np.array([[10.0, 0.0, 0.0], [0.0, 12.0, 1.0], [-8.0, 3.0, -1.0]])
    projected = project(make_cloud(xyz=xyz))
    occupied = projected.mapping >= 0
    assert np.allclose(projected.channel("incidence")[occupied], 0.5)


def test_native_incidence_column_is_used_instead_of_estimating():
    """A simulator that exports incidence is authoritative."""
    sample = make_cloud(600, columns=("x", "y", "z", "intensity", "incidence"))
    sample.points[:, 4] = 0.123
    projected = project(sample)
    occupied = projected.mapping >= 0
    assert np.allclose(projected.channel("incidence")[occupied], 0.123)


def test_estimate_incidence_on_an_empty_frame():
    empty = np.zeros((3, 8, 8))
    mask = np.zeros((8, 8), dtype=bool)
    assert np.all(estimate_incidence(empty, mask) == 0.0)


# --------------------------------------------------------------------------- #
# Determinism and the stage wrapper
# --------------------------------------------------------------------------- #


def test_projection_is_deterministic():
    sample = make_cloud(3000, seed=12)
    first, second = project(sample), project(sample)
    assert np.array_equal(first.range_image, second.range_image)
    assert np.array_equal(first.mapping, second.mapping)


def test_contested_pixels_resolve_deterministically():
    """Ties must not depend on sort instability."""
    xyz = np.repeat(np.array([[10.0, 0.0, 0.0]]), 5, axis=0)
    runs = {project(make_cloud(xyz=xyz)).mapping.max() for _ in range(5)}
    assert len(runs) == 1


def test_stage_wrapper_uses_config_geometry(sensor_cfg_dict):
    from reality.core.config import Config

    sensor_cfg_dict["sensor"] = {"proj_H": 32, "proj_W": 256, "fov_up": 15.0, "fov_down": -25.0}
    stage = SphericalProjection(Config.from_dict(sensor_cfg_dict))
    projected = stage.apply(make_cloud(500, sensor=CADC_SENSOR))
    assert projected.range_image.shape[1:] == (32, 256)


# --------------------------------------------------------------------------- #
# Incidence semantics lock
#
# The incidence channel is cos(theta): theta = 0 (beam along the surface normal)
# gives 1.0, and grazing (theta -> 90 deg) tends to 0. These tests are the
# tripwire against a radians-vs-cosine regression, which would silently invert
# the channel's meaning and feed the physics loss the wrong quantity.
# --------------------------------------------------------------------------- #


def test_incidence_is_one_at_normal_incidence():
    """A sphere centred on the sensor is hit normally everywhere: cos(theta) = 1."""
    # Oversample the image grid so neighbouring pixels are occupied and a local
    # normal can actually be estimated.
    radius = 20.0
    elevation = np.radians(np.linspace(-24.5, 2.5, 2 * KITTI_SENSOR.proj_H))
    azimuth = np.linspace(-np.pi, np.pi, 2 * KITTI_SENSOR.proj_W, endpoint=False)
    grid_elevation, grid_azimuth = np.meshgrid(elevation, azimuth, indexing="ij")
    grid_elevation, grid_azimuth = grid_elevation.ravel(), grid_azimuth.ravel()
    xyz = np.column_stack([
        radius * np.cos(grid_elevation) * np.cos(grid_azimuth),
        radius * np.cos(grid_elevation) * np.sin(grid_azimuth),
        radius * np.sin(grid_elevation),
    ])
    projected = project(make_cloud(xyz=xyz))

    interior = projected.mapping >= 0
    interior[0, :] = interior[-1, :] = False
    incidence = projected.channel("incidence")[interior]
    assert incidence.mean() > 0.98, (
        f"normal incidence must give cos(theta) ~ 1.0, got mean {incidence.mean():.3f}"
    )
    assert incidence.min() > 0.9


def test_incidence_tends_to_zero_at_grazing():
    """A distant horizontal plane is grazed: cos(theta) = |z| / range -> 0."""
    grid = np.linspace(-60, 60, 400)
    xs, ys = np.meshgrid(grid, grid)
    xyz = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, -1.8)])
    projected = project(make_cloud(xyz=xyz))

    occupied = projected.mapping >= 0
    occupied[0, :] = occupied[-1, :] = False
    ranges = projected.channel("range")[occupied]
    incidence = projected.channel("incidence")[occupied]

    far = ranges > 40.0
    assert far.any()
    assert incidence[far].mean() < 0.1, "far ground returns must be near-grazing"
    near = ranges < 8.0
    if near.any():
        assert incidence[near].mean() > incidence[far].mean(), (
            "incidence must fall with range on a flat ground plane"
        )


def test_incidence_is_a_cosine_not_an_angle():
    """cos(theta) in [0, 1], never an angle in radians (which would reach ~1.57)."""
    grid = np.linspace(-25, 25, 300)
    xs, ys = np.meshgrid(grid, grid)
    xyz = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, -2.0)])
    projected = project(make_cloud(xyz=xyz))

    occupied = projected.mapping >= 0
    occupied[0, :] = occupied[-1, :] = False
    incidence = projected.channel("incidence")[occupied]
    ranges = projected.channel("range")[occupied]

    assert incidence.max() <= 1.0, "a cosine cannot exceed 1"
    # cos(theta) = |z| / r for a horizontal plane; the angle would be arccos of that.
    analytic_cosine = 2.0 / ranges
    analytic_radians = np.arccos(np.clip(analytic_cosine, -1, 1))
    assert np.abs(incidence - analytic_cosine).mean() < 0.1
    assert np.abs(incidence - analytic_radians).mean() > 0.5, (
        "the channel must be the cosine, not the angle"
    )
