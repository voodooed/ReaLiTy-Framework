"""Dataset adapters, on synthetic clouds. No real data required."""

import numpy as np
import pytest

from reality.core.config import DataSpec, LabelSpec, SensorSpec
from reality.core.registry import DATASETS
from reality.datasets import (
    BoreasAdapter,
    CadcAdapter,
    DatasetError,
    GenericAdapter,
    KittiAdapter,
    NuScenesAdapter,
    VoxelScapeAdapter,
)

rng = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# Synthetic data builders
# --------------------------------------------------------------------------- #


def cloud(n=64, columns=4, intensity_max=1.0):
    """A synthetic cloud: plausible geometry, intensity in [0, intensity_max]."""
    pts = np.zeros((n, columns), dtype=np.float32)
    pts[:, 0] = rng.uniform(-40, 40, n)
    pts[:, 1] = rng.uniform(-40, 40, n)
    pts[:, 2] = rng.uniform(-3, 2, n)
    pts[:, 3] = rng.uniform(0, intensity_max, n)
    for extra in range(4, columns):
        pts[:, extra] = rng.integers(0, 32, n)
    return pts


def write_kitti(tmp_path, sequences=("00",), frames=2, labels=True, n=64):
    """Write the real KITTI odometry layout under tmp_path."""
    velo_root = tmp_path / "data_odometry_velodyne/dataset/sequences"
    label_root = tmp_path / "data_odometry_labels/dataset/sequences"
    ids = [0, 10, 40, 44, 48, 50, 70, 71, 72, 80, 81, 252]
    for seq in sequences:
        (velo_root / seq / "velodyne").mkdir(parents=True)
        if labels:
            (label_root / seq / "labels").mkdir(parents=True)
        for i in range(frames):
            stem = f"{i:06d}"
            cloud(n).tofile(velo_root / seq / "velodyne" / f"{stem}.bin")
            if labels:
                semantic = rng.choice(ids, n).astype(np.uint32)
                instance = rng.integers(0, 5, n).astype(np.uint32) << 16
                (semantic | instance).tofile(label_root / seq / "labels" / f"{stem}.label")
    return tmp_path


def write_cadc(tmp_path, drives=(("2018_03_06", "0001"),), frames=2, n=48, empty_drive=False):
    for date, drive in drives:
        data = tmp_path / "cadcd" / date / drive / "labeled/lidar_points/data"
        data.mkdir(parents=True)
        (tmp_path / "cadcd" / date / "calib").mkdir(parents=True, exist_ok=True)
        for i in range(frames):
            # CADC intensity is 8-bit already normalised onto a k/255 grid.
            pts = cloud(n)
            pts[:, 3] = rng.integers(0, 256, n) / 255.0
            pts.tofile(data / f"{i:010d}.bin")
    if empty_drive:
        (tmp_path / "cadcd" / "2019_02_27" / "0061").mkdir(parents=True)
    return tmp_path


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_every_adapter_is_registered():
    assert set(DATASETS.names()) >= {"kitti", "cadc", "generic", "voxelscape",
                                     "nuscenes", "boreas"}


@pytest.mark.parametrize("name, cls", [
    ("kitti", KittiAdapter), ("cadc", CadcAdapter), ("generic", GenericAdapter),
    ("voxelscape", VoxelScapeAdapter), ("nuscenes", NuScenesAdapter), ("boreas", BoreasAdapter),
])
def test_registry_resolves_to_the_adapter(name, cls):
    assert DATASETS.get(name) is cls


# --------------------------------------------------------------------------- #
# KITTI — 4 columns, labels present and absent
# --------------------------------------------------------------------------- #


def test_kitti_lists_frames(tmp_path):
    write_kitti(tmp_path, sequences=("00", "01"), frames=3)
    adapter = KittiAdapter(DataSpec(dataset="kitti", path=str(tmp_path)))
    frames = adapter.list_frames()
    assert len(frames) == 6 == len(adapter)
    assert frames[0].id == "00/000000"
    assert all(f.has_labels for f in frames)


def test_kitti_with_labels_takes_the_3_channel_path(tmp_path):
    write_kitti(tmp_path)
    sample = KittiAdapter(DataSpec(dataset="kitti", path=str(tmp_path)))[0]
    assert sample.meta.has_reflectance is True
    assert sample.meta.source_channels == 3
    assert sample.meta.columns == ("x", "y", "z", "intensity", "label", "reflectance")
    assert sample.points.shape == (64, 6)


def test_kitti_without_labels_takes_the_2_channel_path(tmp_path):
    write_kitti(tmp_path, labels=False)
    sample = KittiAdapter(DataSpec(dataset="kitti", path=str(tmp_path)))[0]
    assert sample.meta.has_reflectance is False
    assert sample.meta.source_channels == 2
    assert sample.meta.columns == ("x", "y", "z", "intensity")
    assert sample.points.shape == (64, 4)


def test_kitti_reflectance_comes_from_the_lut(tmp_path):
    write_kitti(tmp_path)
    adapter = KittiAdapter(DataSpec(dataset="kitti", path=str(tmp_path)))
    sample = adapter[0]
    labels = sample.point_column("label").astype(int)
    reflectance = sample.point_column("reflectance")
    assert np.all((reflectance >= 0) & (reflectance <= 1))
    for class_id in np.unique(labels):
        expected = adapter.lut.reflectance_for(int(class_id))
        assert np.allclose(reflectance[labels == class_id], expected)


def test_kitti_exposes_the_raw_cloud_for_weather_model(tmp_path):
    """the weather model needs untouched x, y, z, intensity even when reflectance is appended."""
    write_kitti(tmp_path)
    adapter = KittiAdapter(DataSpec(dataset="kitti", path=str(tmp_path)))
    frame = adapter.list_frames()[0]
    on_disk = np.fromfile(frame.path, dtype=np.float32).reshape(-1, 4)
    raw = adapter.load_sample(frame).raw_cloud()
    assert raw.shape == (64, 4)
    assert np.allclose(raw, on_disk)


def test_kitti_instance_bits_are_stripped(tmp_path):
    """Labels are uint32 with instance id in the high 16 bits."""
    write_kitti(tmp_path)
    adapter = KittiAdapter(DataSpec(dataset="kitti", path=str(tmp_path)))
    labels = adapter.load_labels(adapter.list_frames()[0])
    assert labels.max() < 0x10000


def test_kitti_sequence_filter(tmp_path):
    write_kitti(tmp_path, sequences=("00", "01", "02"))
    adapter = KittiAdapter(DataSpec(dataset="kitti", path=str(tmp_path)), sequences=["01"])
    assert {f.extra["sequence"] for f in adapter.list_frames()} == {"01"}


def test_kitti_sensor_geometry_matches_hdl64(tmp_path):
    write_kitti(tmp_path)
    adapter = KittiAdapter(DataSpec(dataset="kitti", path=str(tmp_path)))
    assert adapter.sensor == SensorSpec(proj_H=64, proj_W=1024, fov_up=3.0, fov_down=-25.0)
    assert adapter[0].meta.fov == adapter.sensor


def test_kitti_config_sensor_overrides_the_default(tmp_path):
    write_kitti(tmp_path)
    override = SensorSpec(proj_H=64, proj_W=2048, fov_up=2.0, fov_down=-24.9)
    adapter = KittiAdapter(DataSpec(dataset="kitti", path=str(tmp_path)), sensor=override)
    assert adapter[0].meta.fov == override


def test_kitti_explicit_label_path(tmp_path):
    write_kitti(tmp_path)
    spec = DataSpec(dataset="kitti", path=str(tmp_path),
                    labels=LabelSpec(path=str(tmp_path / "data_odometry_labels/dataset/sequences"),
                                     format="semantickitti"))
    assert KittiAdapter(spec)[0].meta.has_reflectance is True


def test_kitti_label_count_mismatch_is_an_error(tmp_path):
    write_kitti(tmp_path, frames=1)
    label = tmp_path / "data_odometry_labels/dataset/sequences/00/labels/000000.label"
    np.zeros(7, dtype=np.uint32).tofile(label)
    with pytest.raises(DatasetError, match="64 points but 7 labels"):
        KittiAdapter(DataSpec(dataset="kitti", path=str(tmp_path)))[0]


def test_kitti_missing_root(tmp_path):
    with pytest.raises(DatasetError, match="neither a 'train' folder"):
        KittiAdapter(DataSpec(dataset="kitti", path=str(tmp_path / "nope")))


# --------------------------------------------------------------------------- #
# CADC — target domain, no per-point labels
# --------------------------------------------------------------------------- #


def test_cadc_lists_frames(tmp_path):
    write_cadc(tmp_path, drives=(("2018_03_06", "0001"), ("2018_03_06", "0002")), frames=2)
    adapter = CadcAdapter(DataSpec(dataset="cadc", path=str(tmp_path)))
    frames = adapter.list_frames()
    assert len(frames) == 4
    assert frames[0].id == "2018_03_06/0001/0000000000"


def test_cadc_always_takes_the_no_reflectance_path(tmp_path):
    """CADC ships cuboids, not per-point labels, so the target stays 1-channel."""
    write_cadc(tmp_path)
    sample = CadcAdapter(DataSpec(dataset="cadc", path=str(tmp_path)))[0]
    assert sample.meta.has_reflectance is False
    assert sample.meta.source_channels == 2
    assert sample.meta.columns == ("x", "y", "z", "intensity")
    assert 0.0 <= sample.point_column("intensity").min()
    assert sample.point_column("intensity").max() <= 1.0


def test_cadc_skips_drives_without_lidar(tmp_path):
    """2019_02_27/0061 exists in the real dataset but ships no lidar_points/data."""
    write_cadc(tmp_path, frames=1, empty_drive=True)
    assert len(CadcAdapter(DataSpec(dataset="cadc", path=str(tmp_path)))) == 1


def test_cadc_skips_the_calib_directory(tmp_path):
    write_cadc(tmp_path, frames=1)
    frames = CadcAdapter(DataSpec(dataset="cadc", path=str(tmp_path))).list_frames()
    assert all("calib" not in f.id for f in frames)


def test_cadc_date_and_drive_filters(tmp_path):
    write_cadc(tmp_path, drives=(("2018_03_06", "0001"), ("2018_03_07", "0002")), frames=1)
    adapter = CadcAdapter(DataSpec(dataset="cadc", path=str(tmp_path)), dates=["2018_03_07"])
    assert [f.extra["date"] for f in adapter.list_frames()] == ["2018_03_07"]


def test_cadc_sensor_geometry_matches_vlp32c(tmp_path):
    write_cadc(tmp_path, frames=1)
    adapter = CadcAdapter(DataSpec(dataset="cadc", path=str(tmp_path)))
    assert adapter.sensor == SensorSpec(proj_H=32, proj_W=1024, fov_up=15.0, fov_down=-25.0)


# --------------------------------------------------------------------------- #
# Generic — everything from config
# --------------------------------------------------------------------------- #


def test_generic_4_column_bin(tmp_path):
    (tmp_path / "a.bin").write_bytes(cloud(32, 4).tobytes())
    spec = DataSpec(dataset="generic", path=str(tmp_path), format="bin",
                    columns=("x", "y", "z", "intensity"))
    sample = GenericAdapter(spec)[0]
    assert sample.points.shape == (32, 4)
    assert sample.meta.has_reflectance is False


def test_generic_5_column_bin_with_ring(tmp_path):
    (tmp_path / "a.bin").write_bytes(cloud(32, 5).tobytes())
    spec = DataSpec(dataset="generic", path=str(tmp_path), format="bin",
                    columns=("x", "y", "z", "intensity", "ring"))
    sample = GenericAdapter(spec)[0]
    assert sample.points.shape == (32, 5)
    assert sample.meta.columns == ("x", "y", "z", "intensity", "ring")


def test_generic_declared_columns_are_not_guessed(tmp_path):
    """A 5-column file declared as 4 columns must fail, not be silently reshaped."""
    (tmp_path / "a.bin").write_bytes(cloud(30, 5).tobytes())
    spec = DataSpec(dataset="generic", path=str(tmp_path), format="bin",
                    columns=("x", "y", "z", "intensity"))
    # 30*5 = 150 values, not a multiple of 4.
    with pytest.raises(DatasetError, match="not a multiple of the declared 4 columns"):
        GenericAdapter(spec)[0]


def test_generic_intensity_scale_is_applied(tmp_path):
    pts = cloud(16, 4)
    pts[:, 3] = np.linspace(0, 255, 16)
    (tmp_path / "a.bin").write_bytes(pts.tobytes())
    spec = DataSpec(dataset="generic", path=str(tmp_path), format="bin",
                    columns=("x", "y", "z", "intensity"), intensity_scale=255.0)
    sample = GenericAdapter(spec)[0]
    assert sample.point_column("intensity").max() == pytest.approx(1.0)
    assert sample.meta.intensity_scale == 255.0
    # Geometry must be untouched by intensity scaling.
    assert np.allclose(sample.points[:, :3], pts[:, :3])


def test_generic_with_labels_gets_reflectance(tmp_path):
    data, labels = tmp_path / "data", tmp_path / "labels"
    data.mkdir(); labels.mkdir()
    (data / "a.bin").write_bytes(cloud(20, 4).tobytes())
    np.full(20, 70, dtype=np.uint32).tofile(labels / "a.label")
    spec = DataSpec(dataset="generic", path=str(data), format="bin",
                    columns=("x", "y", "z", "intensity"),
                    labels=LabelSpec(path=str(labels), format="semantickitti"))
    sample = GenericAdapter(spec)[0]
    assert sample.meta.has_reflectance is True
    assert np.allclose(sample.point_column("reflectance"), 0.49)  # vegetation


def test_generic_npy_format(tmp_path):
    np.save(tmp_path / "a.npy", cloud(12, 4))
    spec = DataSpec(dataset="generic", path=str(tmp_path), format="npy",
                    columns=("x", "y", "z", "intensity"))
    assert GenericAdapter(spec)[0].points.shape == (12, 4)


def test_generic_rejects_unknown_format(tmp_path):
    spec = DataSpec(dataset="generic", path=str(tmp_path), format="pcd",
                    columns=("x", "y", "z", "intensity"))
    with pytest.raises(DatasetError, match="unsupported format"):
        GenericAdapter(spec)


def test_generic_requires_the_intensity_column(tmp_path):
    spec = DataSpec(dataset="generic", path=str(tmp_path), format="bin",
                    columns=("x", "y", "z"))
    with pytest.raises(DatasetError, match="missing required column"):
        GenericAdapter(spec)


def test_generic_empty_file(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"")
    spec = DataSpec(dataset="generic", path=str(tmp_path), format="bin",
                    columns=("x", "y", "z", "intensity"))
    with pytest.raises(DatasetError, match="empty"):
        GenericAdapter(spec)[0]


# --------------------------------------------------------------------------- #
# Adapters awaiting data
# --------------------------------------------------------------------------- #


@pytest.mark.skip(reason="data pending")
def test_voxelscape_source(tmp_path):
    spec = DataSpec(dataset="voxelscape", path=str(tmp_path), format="bin")
    adapter = VoxelScapeAdapter(spec)
    assert adapter.provides_physics_intensity
    assert adapter[0].meta.columns[-1] == "physics_intensity"


@pytest.mark.skip(reason="data pending")
def test_nuscenes_target(tmp_path):
    spec = DataSpec(dataset="nuscenes", path=str(tmp_path))
    sample = NuScenesAdapter(spec)[0]
    assert sample.meta.columns == ("x", "y", "z", "intensity", "ring")
    assert sample.point_column("intensity").max() <= 1.0  # scaled from 0-255


@pytest.mark.skip(reason="data pending")
def test_boreas_target(tmp_path):
    spec = DataSpec(dataset="boreas", path=str(tmp_path))
    assert BoreasAdapter(spec)[0].meta.has_reflectance is False
