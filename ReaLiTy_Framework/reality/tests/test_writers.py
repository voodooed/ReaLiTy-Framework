"""OutputWriter: native byte layout, matching what we confirmed on real data."""

import numpy as np
import pytest

from reality.core.context import Sample, SampleMeta
from reality.io import OutputWriter, WriterError


def sample_with(columns=("x", "y", "z", "intensity"), n=32, frame_id="00/000000"):
    rng = np.random.default_rng(0)
    points = rng.random((n, len(columns))).astype(np.float32)
    meta = SampleMeta(dataset="kitti", task="sensor", columns=tuple(columns),
                      extra={"frame_id": frame_id})
    return Sample(points=points, meta=meta)


def test_writes_headerless_float32_bin(tmp_path):
    """KITTI and CADC clouds are headerless float32 -- a written frame must match."""
    sample = sample_with()
    path = OutputWriter(tmp_path, fmt="bin").write(sample)
    assert path.suffix == ".bin"
    assert path.stat().st_size == sample.points.size * 4
    reread = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    assert np.array_equal(reread, sample.points)


def test_round_trips_through_the_kitti_adapter(tmp_path):
    """What we write must load back through the adapter that reads the real data."""
    from reality.core.config import DataSpec
    from reality.datasets import KittiAdapter

    sample = sample_with(n=64)
    velodyne = tmp_path / "data_odometry_velodyne/dataset/sequences/00/velodyne"
    velodyne.mkdir(parents=True)
    OutputWriter(velodyne, fmt="bin").write(sample, frame_id="000000")

    adapter = KittiAdapter(DataSpec(dataset="kitti", path=str(tmp_path)))
    loaded = adapter[0]
    assert loaded.points.shape == (64, 4)
    assert np.allclose(loaded.points, sample.points)


def test_writes_npy(tmp_path):
    sample = sample_with()
    path = OutputWriter(tmp_path, fmt="npy").write(sample)
    assert np.array_equal(np.load(path), sample.points)


def test_frame_id_subdirectories_are_created(tmp_path):
    path = OutputWriter(tmp_path).write(sample_with(frame_id="2018_03_06/0001/0000000000"))
    assert path == tmp_path / "2018_03_06/0001/0000000000.bin"
    assert path.is_file()


def test_selects_declared_columns_in_order(tmp_path):
    """A 6-column working cloud is written back in the target's native layout."""
    sample = sample_with(("x", "y", "z", "intensity", "label", "reflectance"))
    writer = OutputWriter(tmp_path, columns=("x", "y", "z", "intensity"))
    reread = np.fromfile(writer.write(sample), dtype=np.float32).reshape(-1, 4)
    assert reread.shape == (32, 4)
    assert np.array_equal(reread, sample.points[:, :4])


def test_column_reordering_follows_the_declaration(tmp_path):
    sample = sample_with(("x", "y", "z", "intensity", "ring"))
    writer = OutputWriter(tmp_path, columns=("intensity", "x", "y", "z"))
    reread = np.fromfile(writer.write(sample), dtype=np.float32).reshape(-1, 4)
    assert np.array_equal(reread[:, 0], sample.points[:, 3])
    assert np.array_equal(reread[:, 1:], sample.points[:, :3])


def test_missing_column_is_reported(tmp_path):
    writer = OutputWriter(tmp_path, columns=("x", "y", "z", "ring"))
    with pytest.raises(WriterError, match=r"\['ring'\] not in the cloud"):
        writer.write(sample_with())


def test_unsupported_format(tmp_path):
    with pytest.raises(WriterError, match="unsupported output format"):
        OutputWriter(tmp_path, fmt="pcd")


def test_writes_an_explicit_points_array(tmp_path):
    sample = sample_with()
    modified = sample.points.copy()
    modified[:, 3] = 0.25
    reread = np.fromfile(OutputWriter(tmp_path).write(sample, modified),
                         dtype=np.float32).reshape(-1, 4)
    assert np.allclose(reread[:, 3], 0.25)
