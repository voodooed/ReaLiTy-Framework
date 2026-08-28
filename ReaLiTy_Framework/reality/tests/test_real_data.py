"""Smoke tests against the real datasets, skipped when the data is absent.

These assert exactly what was observed while inspecting the data ,
so CI stays synthetic while this machine verifies the adapters against reality.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from reality.core.config import DataSpec
from reality.datasets import CadcAdapter, KittiAdapter

DATA_ROOT = Path(os.environ.get("REALITY_DATA_ROOT", "data"))
KITTI_ROOT = DATA_ROOT / "KITTI"
CADC_ROOT = DATA_ROOT / "CADC"

kitti_only = pytest.mark.skipif(not KITTI_ROOT.is_dir(), reason="KITTI data not present")
cadc_only = pytest.mark.skipif(not CADC_ROOT.is_dir(), reason="CADC data not present")


@pytest.fixture(scope="module")
def kitti_labelled():
    """Sequence 00 — labelled, so the 3-channel path."""
    return KittiAdapter(DataSpec(dataset="kitti", path=str(KITTI_ROOT)), sequences=["00"])


@pytest.fixture(scope="module")
def kitti_unlabelled():
    """Sequence 11 — the unlabelled test split, so the 2-channel path."""
    return KittiAdapter(DataSpec(dataset="kitti", path=str(KITTI_ROOT)), sequences=["11"])


@pytest.fixture(scope="module")
def cadc():
    return CadcAdapter(DataSpec(dataset="cadc", path=str(CADC_ROOT)),
                       dates=["2018_03_06"], drives=["0001"])


@kitti_only
def test_kitti_frame_matches_what_was_observed(kitti_labelled):
    frames = kitti_labelled.list_frames()
    assert len(frames) == 4541, "sequence 00 should hold 4541 frames"
    assert frames[0].id == "00/000000"

    sample = kitti_labelled.load_sample(frames[0])
    # 124,668 points: 1,994,688 bytes / (4 float32 * 4 bytes).
    assert sample.num_points == 124668
    assert sample.points.dtype == np.float32
    assert sample.meta.has_reflectance is True
    assert sample.meta.source_channels == 3
    assert sample.meta.columns == ("x", "y", "z", "intensity", "label", "reflectance")

    intensity = sample.point_column("intensity")
    assert 0.0 <= intensity.min() and intensity.max() <= 1.0
    assert intensity.max() == pytest.approx(0.99, abs=0.01)

    xyz = sample.points[:, :3]
    assert np.abs(xyz).max() < 200, "KITTI ranges stay well under 200 m"


@kitti_only
def test_kitti_labels_align_and_map_to_reflectance(kitti_labelled):
    sample = kitti_labelled[0]
    labels = sample.point_column("label").astype(int)
    reflectance = sample.point_column("reflectance")
    assert labels.shape[0] == sample.num_points
    assert labels.max() < 0x10000, "instance bits must be stripped"
    assert set(np.unique(labels)) <= set(kitti_labelled.lut.class_maps["semantickitti"])
    assert np.all((reflectance > 0) & (reflectance <= 1))
    # Road is the dominant surface and must carry the asphalt value.
    if (labels == 40).any():
        assert np.allclose(reflectance[labels == 40], 0.10)


@kitti_only
def test_kitti_unlabelled_split_uses_the_2_channel_path(kitti_unlabelled):
    frames = kitti_unlabelled.list_frames()
    assert len(frames) == 921, "sequence 11 should hold 921 frames"
    assert not any(f.has_labels for f in frames), "sequences 11-21 ship no labels"
    sample = kitti_unlabelled.load_sample(frames[0])
    assert sample.meta.has_reflectance is False
    assert sample.meta.source_channels == 2
    assert sample.points.shape[1] == 4


@cadc_only
def test_cadc_frame_matches_what_was_observed(cadc):
    frames = cadc.list_frames()
    assert len(frames) == 100, "2018_03_06/0001 holds 100 frames"
    assert frames[0].id == "2018_03_06/0001/0000000000"

    sample = cadc.load_sample(frames[0])
    # 44,940 points: 719,040 bytes / (4 float32 * 4 bytes).
    assert sample.num_points == 44940
    assert sample.meta.has_reflectance is False
    assert sample.meta.columns == ("x", "y", "z", "intensity")

    intensity = sample.point_column("intensity")
    assert 0.0 <= intensity.min() and intensity.max() <= 1.0
    # 8-bit sensor values already normalised: every value sits on the k/255 grid.
    scaled = intensity * 255.0
    assert np.abs(scaled - np.round(scaled)).max() < 1e-4

    z = sample.point_column("z")
    assert -10 < z.min() and z.max() < 30, "the 4-column reading gives a sane z"


@cadc_only
def test_cadc_beam_geometry_matches_the_declared_sensor(cadc):
    sample = cadc[0]
    xyz = sample.points[:, :3]
    r = np.linalg.norm(xyz, axis=1)
    elevation = np.degrees(np.arcsin(xyz[r > 0, 2] / r[r > 0]))
    assert elevation.min() >= cadc.sensor.fov_down - 0.5
    assert elevation.max() <= cadc.sensor.fov_up + 0.5


@kitti_only
@cadc_only
def test_both_domains_load_for_the_kitti_to_cadc_experiment(kitti_labelled, cadc):
    """The real pairing: KITTI source (3-channel) -> CADC target (intensity only)."""
    source, target = kitti_labelled[0], cadc[0]
    assert source.meta.has_reflectance and not target.meta.has_reflectance
    assert source.meta.source_channels == 3
    assert source.raw_cloud().shape[1] == 4, "the weather model gets x, y, z, intensity"
    assert target.point_column("intensity").max() <= 1.0


# --------------------------------------------------------------------------- #
# : projection and back-projection on real frames
# --------------------------------------------------------------------------- #


@kitti_only
def test_real_kitti_frame_projects_and_round_trips(kitti_labelled):
    """The alignment guardrail, on 124,668 real points with real labels."""
    from reality.postprocessing.backprojection import backproject
    from reality.preprocessing.projection import project

    sample = kitti_labelled[0]
    projected = project(sample)

    assert projected.channels == ("range", "incidence", "reflectance", "intensity", "mask")
    assert projected.range_image.shape == (5, 64, 1024)
    stats = projected.meta.extra["projection"]
    assert stats["n_points"] == 124668
    assert stats["n_projected"] + stats["n_dropped"] == 124668
    assert stats["n_projected"] > 40000, "a real frame should fill much of the image"

    # Range plane must equal the true range of the point each pixel names.
    rows, cols = np.nonzero(projected.mapping >= 0)
    owners = projected.mapping[rows, cols]
    true_range = np.linalg.norm(sample.points[owners, :3], axis=1)
    assert np.allclose(projected.channel("range")[rows, cols], true_range, rtol=1e-4, atol=1e-3)

    # Round trip: feed the range image back and every written point gets its own range.
    result = backproject(projected, projected.channel("range"))
    assert np.array_equal(result.points[:, :3], sample.points[:, :3]), "geometry untouched"
    assert np.allclose(result.points[result.written, 3],
                       np.linalg.norm(sample.points[result.written, :3], axis=1),
                       rtol=1e-4, atol=1e-3)


@kitti_only
def test_real_kitti_incidence_and_reflectance_are_physical(kitti_labelled):
    from reality.preprocessing.projection import project

    projected = project(kitti_labelled[0])
    occupied = projected.mapping >= 0
    incidence = projected.channel("incidence")[occupied]
    reflectance = projected.channel("reflectance")[occupied]
    assert np.isfinite(incidence).all()
    assert 0.0 <= incidence.min() and incidence.max() <= 1.0
    assert 0.0 < reflectance.min() and reflectance.max() <= 1.0
    # Road dominates a KITTI frame, so grazing incidence should be common.
    assert np.median(incidence) < 0.6


@cadc_only
def test_real_cadc_frame_projects_on_the_2_channel_path(cadc):
    """CADC has no labels, so projection must produce the reduced stack."""
    from reality.preprocessing.projection import project

    sample = cadc[0]
    projected = project(sample)
    assert projected.meta.has_reflectance is False
    assert projected.channels == ("range", "incidence", "intensity", "mask")
    assert projected.range_image.shape == (4, 32, 1024)
    assert projected.phy is None, "CADC supplies no physics intensity; the weather model does, in "

    occupied = projected.mapping >= 0
    intensity = projected.channel("intensity")[occupied]
    assert 0.0 <= intensity.min() and intensity.max() <= 1.0


@cadc_only
def test_real_cadc_round_trip_preserves_geometry(cadc):
    from reality.postprocessing.backprojection import backproject
    from reality.preprocessing.projection import project

    sample = cadc[0]
    projected = project(sample)
    result = backproject(projected, projected.channel("range"), fill="nan")
    assert np.array_equal(result.points[:, :3], sample.points[:, :3])
    assert np.allclose(result.points[result.written, 3],
                       np.linalg.norm(sample.points[result.written, :3], axis=1),
                       rtol=1e-4, atol=1e-3)
    assert np.isnan(result.points[~result.written, 3]).all()


@kitti_only
def test_real_kitti_output_is_a_drop_in_replacement(kitti_labelled, tmp_path):
    """A written frame must be byte-compatible with the format we read."""
    from reality.io import OutputWriter
    from reality.postprocessing.backprojection import backproject
    from reality.preprocessing.projection import project

    sample = kitti_labelled[0]
    projected = project(sample)
    result = backproject(projected, projected.channel("range"))

    writer = OutputWriter(tmp_path, fmt="bin", columns=("x", "y", "z", "intensity"))
    path = writer.write(projected, result.points, frame_id="000000")
    assert path.stat().st_size == sample.num_points * 4 * 4
    reread = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    assert reread.shape == (124668, 4)
    assert np.allclose(reread[:, :3], sample.points[:, :3])


@kitti_only
def test_real_kitti_incidence_is_grazing_dominated_not_voxelscape_like(kitti_labelled):
    """Semantics lock on real data .

    A road scene is dominated by grazing ground returns, so the occupied-pixel
    mean of cos(theta) sits near 0.3-0.4. PICGAN's shipped incidence constant
    assumes 0.7156, which is what the computed normalization path corrects. If
    this mean ever drifts up towards 0.72, the channel's definition changed.
    """
    from reality.preprocessing.projection import project

    projected = project(kitti_labelled[0])
    occupied = projected.mapping >= 0
    mean = float(projected.channel("incidence")[occupied].mean())
    assert 0.25 < mean < 0.45, f"expected a grazing-dominated cos(theta) mean, got {mean:.3f}"
    assert abs(mean - 0.7156) > 0.25, "this must not look like the VoxelScape constant"


@kitti_only
def test_computed_statistics_measure_the_cosine_channel(tmp_path):
    """The computed normalization must be fitted to these cos(theta) values."""
    from reality.core.config import Config
    from reality.preprocessing.statistics import PICGAN_DEFAULT_STATS, compute_from_config

    config = Config.from_dict({
        "source": {"dataset": "kitti", "path": str(KITTI_ROOT)},
        "target": {"dataset": "cadc", "path": str(CADC_ROOT)},
        "task": {"type": "sensor"},
        "sensor": {"proj_H": 64, "proj_W": 1024, "fov_up": 3.0, "fov_down": -25.0},
        "normalization": {"source": "computed", "frames": 2},
        "output": {"checkpoint_dir": str(tmp_path / "ckpt")},
    })
    stats = compute_from_config(config)
    mean, std = stats.pair("incidence")
    assert 0.25 < mean < 0.45
    assert (mean, std) != PICGAN_DEFAULT_STATS["incidence"]
    assert 0.0 <= stats.channels["incidence"].minimum
    assert stats.channels["incidence"].maximum <= 1.0
