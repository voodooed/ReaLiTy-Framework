"""Back-projection: index fidelity, untouched geometry, explicit dropped points."""

import numpy as np
import pytest

from reality.core.config import SensorSpec
from reality.core.context import Sample, SampleMeta
from reality.postprocessing.backprojection import (
    BackProjectionError,
    BackProjectionStage,
    backproject,
)
from reality.preprocessing.projection import project
from reality.tests.test_projection import KITTI_SENSOR, make_cloud


def projected_cloud(n=2000, seed=0, **kwargs):
    sample = make_cloud(n, seed=seed, **kwargs)
    return sample, project(sample)


# --------------------------------------------------------------------------- #
# The round trip
# --------------------------------------------------------------------------- #


def test_geometry_is_byte_identical_after_a_round_trip():
    original, projected = projected_cloud(2500, seed=1)
    result = backproject(projected, np.full(projected.mapping.shape, 0.5, dtype=np.float32))
    assert np.array_equal(result.points[:, :3], original.points[:, :3])
    assert result.points.dtype == original.points.dtype
    assert result.points.shape == original.points.shape


def test_intensity_lands_on_the_points_the_map_names():
    """Feed the range image back as intensity: each point must receive its own range.

    This is the alignment guardrail. If the mapping were built against a filtered
    cloud, or applied to a different one, points would receive a neighbour's range
    and this comparison would fail.
    """
    original, projected = projected_cloud(3000, seed=2)
    result = backproject(projected, projected.channel("range"))

    written_ranges = result.points[result.written, 3]
    true_ranges = np.linalg.norm(original.points[result.written, :3], axis=1)
    assert np.allclose(written_ranges, true_ranges, rtol=1e-5, atol=1e-4)


def test_round_trip_survives_leading_unprojectable_points():
    """The off-by-N failure mode: dropped points ahead of live ones."""
    dead = np.zeros((75, 3))
    live = make_cloud(600, seed=3).points[:, :3].astype(np.float64)
    sample = make_cloud(xyz=np.vstack([dead, live]))
    projected = project(sample)

    result = backproject(projected, projected.channel("range"))
    assert not result.written[:75].any(), "zero-range points cannot receive intensity"
    assert np.allclose(result.points[result.written, 3],
                       np.linalg.norm(sample.points[result.written, :3], axis=1),
                       rtol=1e-5, atol=1e-4)


def test_only_mapped_points_are_written():
    _, projected = projected_cloud(2000, seed=4)
    result = backproject(projected, np.ones(projected.mapping.shape, dtype=np.float32))
    assert result.n_written == int((projected.mapping >= 0).sum())
    assert result.n_written + result.n_dropped == projected.points.shape[0]
    assert set(np.nonzero(result.written)[0]) == set(
        projected.mapping[projected.mapping >= 0].tolist()
    )


def test_accepts_both_image_shapes():
    _, projected = projected_cloud(500, seed=5)
    flat = np.full(projected.mapping.shape, 0.3, dtype=np.float32)
    a = backproject(projected, flat)
    b = backproject(projected, flat[None, :, :])
    assert np.array_equal(a.points, b.points)


# --------------------------------------------------------------------------- #
# Dropped points are explicit
# --------------------------------------------------------------------------- #


def test_dropped_points_keep_their_original_intensity_by_default():
    original, projected = projected_cloud(2000, seed=6)
    result = backproject(projected, np.full(projected.mapping.shape, 0.9, dtype=np.float32))
    dropped = ~result.written
    assert dropped.any()
    assert np.allclose(result.points[dropped, 3], original.points[dropped, 3])
    assert np.allclose(result.points[result.written, 3], 0.9)


def test_dropped_points_can_be_marked_nan():
    _, projected = projected_cloud(2000, seed=7)
    result = backproject(projected, np.zeros(projected.mapping.shape, dtype=np.float32),
                         fill="nan")
    assert np.isnan(result.points[~result.written, 3]).all()
    assert np.isfinite(result.points[result.written, 3]).all()


def test_dropped_points_can_take_a_constant():
    _, projected = projected_cloud(1000, seed=8)
    result = backproject(projected, np.zeros(projected.mapping.shape, dtype=np.float32),
                         fill=-1.0)
    assert np.all(result.points[~result.written, 3] == -1.0)


def test_dropped_points_are_never_moved_to_the_origin():
    original, projected = projected_cloud(2000, seed=9)
    result = backproject(projected, np.zeros(projected.mapping.shape, dtype=np.float32),
                         fill="nan")
    dropped = ~result.written
    assert np.array_equal(result.points[dropped, :3], original.points[dropped, :3])
    moved_to_origin = np.all(result.points[dropped, :3] == 0.0, axis=1)
    assert moved_to_origin.sum() == np.all(original.points[dropped, :3] == 0.0, axis=1).sum()


# --------------------------------------------------------------------------- #
# Alignment is enforced
# --------------------------------------------------------------------------- #


def test_a_cloud_of_the_wrong_length_is_refused():
    """A mapping applied to a different cloud scatters intensity onto wrong points."""
    _, projected = projected_cloud(1000, seed=10)
    other = np.zeros((999, 4), dtype=np.float32)
    with pytest.raises(BackProjectionError, match="same one that was projected"):
        backproject(projected, np.zeros(projected.mapping.shape, dtype=np.float32), cloud=other)


def test_a_matching_replacement_cloud_is_accepted():
    """ back-projects onto the degraded cloud, which has the same length."""
    original, projected = projected_cloud(800, seed=11)
    degraded = original.points.copy()
    degraded[:, 3] *= 0.5
    result = backproject(projected, projected.channel("range"), cloud=degraded)
    assert np.array_equal(result.points[:, :3], degraded[:, :3])


def test_unprojected_sample_is_refused():
    sample = make_cloud(100)
    with pytest.raises(BackProjectionError, match="project it first"):
        backproject(sample, np.zeros((64, 1024), dtype=np.float32))


def test_image_shape_must_match_the_mapping():
    _, projected = projected_cloud(200, seed=12)
    with pytest.raises(BackProjectionError, match="does not match the mapping"):
        backproject(projected, np.zeros((16, 16), dtype=np.float32))


def test_multichannel_image_is_refused():
    _, projected = projected_cloud(200, seed=13)
    with pytest.raises(BackProjectionError, match=r"\(1, H, W\)"):
        backproject(projected, np.zeros((3,) + projected.mapping.shape, dtype=np.float32))


def test_missing_intensity_column_is_reported():
    _, projected = projected_cloud(200, seed=14)
    with pytest.raises(BackProjectionError, match="no 'reflectivity' column"):
        backproject(projected, np.zeros(projected.mapping.shape, dtype=np.float32),
                    intensity_column="reflectivity")


# --------------------------------------------------------------------------- #
# Stage wrapper
# --------------------------------------------------------------------------- #


def test_stage_reads_the_generated_intensity():
    _, projected = projected_cloud(600, seed=15)
    projected.meta.extra["generated_intensity"] = np.full(
        projected.mapping.shape, 0.42, dtype=np.float32
    )
    out = BackProjectionStage().apply(projected)
    owners = projected.mapping[projected.mapping >= 0]
    assert np.allclose(out.points[owners, 3], 0.42)
    assert out.meta.extra["backprojection"]["n_written"] == len(set(owners.tolist()))


def test_stage_without_a_generated_intensity_is_an_error():
    _, projected = projected_cloud(100, seed=16)
    with pytest.raises(BackProjectionError, match="model stage must run first"):
        BackProjectionStage().apply(projected)


def test_backprojection_is_deterministic():
    _, projected = projected_cloud(1500, seed=17)
    image = projected.channel("range")
    first = backproject(projected, image)
    second = backproject(projected, image)
    assert np.array_equal(first.points, second.points)
    assert np.array_equal(first.written, second.written)
