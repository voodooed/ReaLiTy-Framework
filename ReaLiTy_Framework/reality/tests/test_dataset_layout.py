"""The released dataset layout: <root>/train and <root>/test, by convention."""

import numpy as np
import pytest

from reality.core.config import Config, ConfigError, DataSpec, LabelSpec
from reality.datasets import CadcAdapter, GenericAdapter, KittiAdapter

COLUMNS = ("x", "y", "z", "intensity")


def cloud(n=64, columns=4, seed=0):
    rng = np.random.default_rng(seed)
    points = np.zeros((n, columns), dtype=np.float32)
    points[:, 0] = rng.uniform(-40, 40, n)
    points[:, 1] = rng.uniform(-40, 40, n)
    points[:, 2] = rng.uniform(-3, 2, n)
    points[:, 3] = rng.uniform(0, 1, n)
    for extra in range(4, columns):
        points[:, extra] = rng.random(n)
    return points


def build_dataset(root, train=3, test=2, fmt="bin", columns=4, labels=False):
    """Write a dataset in the released layout."""
    for split, count in (("train", train), ("test", test)):
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            points = cloud(columns=columns, seed=i)
            if fmt == "bin":
                points.tofile(directory / f"{i:06d}.bin")
            else:
                np.save(directory / f"{i:06d}.npy", points)
        if labels:
            label_dir = root / "labels" / split
            label_dir.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                np.full(64, 70, dtype=np.uint32).tofile(label_dir / f"{i:06d}.label")
    return root


# --------------------------------------------------------------------------- #
# Convention
# --------------------------------------------------------------------------- #


def test_train_and_test_are_read_by_folder(tmp_path):
    build_dataset(tmp_path, train=5, test=2)
    train = KittiAdapter(DataSpec(dataset="kitti", path=str(tmp_path), split="train"))
    test = KittiAdapter(DataSpec(dataset="kitti", path=str(tmp_path), split="test"))
    assert train.uses_released_layout and test.uses_released_layout
    assert len(train) == 5 and len(test) == 2
    assert {f.extra["split"] for f in train.list_frames()} == {"train"}


def test_default_split_is_train(tmp_path):
    build_dataset(tmp_path, train=4, test=1)
    assert len(KittiAdapter(DataSpec(dataset="kitti", path=str(tmp_path)))) == 4


@pytest.mark.parametrize("fmt", ["bin", "npy"])
def test_both_formats_are_accepted(tmp_path, fmt):
    build_dataset(tmp_path, train=2, test=1, fmt=fmt)
    adapter = KittiAdapter(DataSpec(dataset="kitti", path=str(tmp_path)))
    sample = adapter[0]
    assert sample.points.shape == (64, 4)
    assert sample.meta.columns == COLUMNS


def test_mixed_formats_in_one_split(tmp_path):
    build_dataset(tmp_path, train=2, test=1, fmt="bin")
    np.save(tmp_path / "train" / "extra.npy", cloud(seed=9))
    assert len(KittiAdapter(DataSpec(dataset="kitti", path=str(tmp_path)))) == 3


def test_declared_columns_still_govern_bin(tmp_path):
    """an earlier rule holds: a .bin needs declared columns, never a guess."""
    (tmp_path / "train").mkdir(parents=True)
    cloud(n=30, columns=5).tofile(tmp_path / "train" / "a.bin")
    spec = DataSpec(dataset="generic", path=str(tmp_path), format="bin", columns=COLUMNS)
    from reality.datasets.base import DatasetError

    with pytest.raises(DatasetError, match="not a multiple of the declared 4"):
        GenericAdapter(spec)[0]


def test_npy_shape_must_match_declared_columns(tmp_path):
    (tmp_path / "train").mkdir(parents=True)
    np.save(tmp_path / "train" / "a.npy", cloud(columns=5))
    from reality.datasets.base import DatasetError

    with pytest.raises(DatasetError, match="does not match the declared 4"):
        KittiAdapter(DataSpec(dataset="kitti", path=str(tmp_path)))[0]


def test_labels_are_matched_by_stem(tmp_path):
    build_dataset(tmp_path, train=2, test=1, labels=True)
    spec = DataSpec(dataset="kitti", path=str(tmp_path),
                    labels=LabelSpec(path=str(tmp_path / "labels"),
                                     format="semantickitti"))
    sample = KittiAdapter(spec)[0]
    assert sample.meta.has_reflectance is True
    assert sample.meta.source_channels == 3
    assert np.allclose(sample.point_column("reflectance"), 0.49)  # vegetation


def test_without_labels_the_source_is_two_channel(tmp_path):
    build_dataset(tmp_path, train=2, test=1)
    sample = KittiAdapter(DataSpec(dataset="kitti", path=str(tmp_path)))[0]
    assert sample.meta.has_reflectance is False
    assert sample.meta.source_channels == 2


def test_target_domain_uses_the_same_convention(tmp_path):
    build_dataset(tmp_path, train=3, test=4)
    train = CadcAdapter(DataSpec(dataset="cadc", path=str(tmp_path), split="train"))
    test = CadcAdapter(DataSpec(dataset="cadc", path=str(tmp_path), split="test"))
    assert len(train) == 3 and len(test) == 4
    assert train[0].meta.has_reflectance is False


def test_any_adapter_follows_the_convention(tmp_path):
    """The layout is dataset-agnostic, which is the point of it."""
    from reality.datasets import BoreasAdapter, NuScenesAdapter, VoxelScapeAdapter

    for cls, columns in [(KittiAdapter, 4), (CadcAdapter, 4), (BoreasAdapter, 5),
                         (NuScenesAdapter, 5), (VoxelScapeAdapter, 5)]:
        root = tmp_path / cls.name
        build_dataset(root, train=2, test=1, columns=columns)
        adapter = cls(DataSpec(dataset=cls.name, path=str(root)))
        assert adapter.uses_released_layout
        assert len(adapter) == 2, f"{cls.name} should read its train folder"


def test_nested_subfolders_are_walked(tmp_path):
    """A converted dataset may keep its own sub-structure inside train/."""
    nested = tmp_path / "train" / "seq00"
    nested.mkdir(parents=True)
    cloud().tofile(nested / "000000.bin")
    (tmp_path / "test").mkdir()
    assert len(KittiAdapter(DataSpec(dataset="kitti", path=str(tmp_path)))) == 1


# --------------------------------------------------------------------------- #
# Fallback to raw layouts
# --------------------------------------------------------------------------- #


def test_raw_kitti_layout_still_works(tmp_path):
    """Sequence handling becomes the fallback, not the primary path."""
    from reality.tests.test_datasets import write_kitti

    write_kitti(tmp_path, sequences=("00",), frames=3)
    adapter = KittiAdapter(DataSpec(dataset="kitti", path=str(tmp_path)))
    assert not adapter.uses_released_layout
    assert len(adapter) == 3


def test_released_layout_takes_precedence_over_raw(tmp_path):
    from reality.tests.test_datasets import write_kitti

    write_kitti(tmp_path, sequences=("00",), frames=3)
    build_dataset(tmp_path, train=1, test=1)
    adapter = KittiAdapter(DataSpec(dataset="kitti", path=str(tmp_path)))
    assert adapter.uses_released_layout
    assert len(adapter) == 1


def test_neither_layout_is_reported(tmp_path):
    from reality.datasets.base import DatasetError

    with pytest.raises(DatasetError, match="neither a 'train' folder"):
        KittiAdapter(DataSpec(dataset="kitti", path=str(tmp_path / "empty")))


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_split_is_config_driven():
    config = Config.from_dict({
        "source": {"dataset": "kitti", "path": "/data/src", "split": "test"},
        "target": {"dataset": "cadc", "path": "/data/tgt"},
        "task": {"type": "sensor"},
    })
    assert config.source.split == "test"
    assert config.target.split == "train", "train is the default"


def test_invalid_split_is_rejected():
    with pytest.raises(ConfigError, match="source.split"):
        Config.from_dict({
            "source": {"dataset": "kitti", "path": "/d", "split": "validation"},
            "target": {"dataset": "cadc", "path": "/d"},
            "task": {"type": "sensor"},
        })


def test_no_training_scale_is_baked_in(tmp_path):
    """The framework trains on whatever is in train/, not a presumed dataset size."""
    build_dataset(tmp_path, train=17, test=3)
    assert len(KittiAdapter(DataSpec(dataset="kitti", path=str(tmp_path)))) == 17
    build_dataset(tmp_path / "big", train=120, test=5)
    assert len(KittiAdapter(DataSpec(dataset="kitti", path=str(tmp_path / "big")))) == 120


# --------------------------------------------------------------------------- #
# Dataset-centric resolution: identity on disk, role in config
# --------------------------------------------------------------------------- #


def test_dataset_name_resolves_to_a_folder_under_data_root():
    config = Config.from_dict({
        "data_root": "Data",
        "source": {"dataset": "kitti"},
        "target": {"dataset": "cadc"},
        "task": {"type": "sensor"},
    })
    assert config.source.path == "Data/KITTI"
    assert config.target.path == "Data/CADC"


@pytest.mark.parametrize("dataset, folder", [
    ("kitti", "KITTI"), ("cadc", "CADC"), ("nuscenes", "nuScenes"),
    ("boreas", "Boreas"), ("voxelscape", "VoxelScape"),
])
def test_every_dataset_has_a_readable_folder_name(dataset, folder):
    from reality.core.config import dataset_folder

    assert dataset_folder(dataset) == folder


def test_role_is_config_not_filesystem(tmp_path):
    """The same dataset folder serves as source in one run and target in another."""
    build_dataset(tmp_path / "KITTI", train=3, test=1)
    build_dataset(tmp_path / "CADC", train=2, test=1)

    weather = Config.from_dict({
        "data_root": str(tmp_path),
        "source": {"dataset": "kitti"}, "target": {"dataset": "cadc"},
        "task": {"type": "weather"},
        "geometric_degradation": {"enabled": True, "type": "physics",
                                  "weather": "snow", "precipitation_rate": 30.0},
    })
    sensor = Config.from_dict({
        "data_root": str(tmp_path),
        "source": {"dataset": "cadc"}, "target": {"dataset": "kitti"},
        "task": {"type": "sensor"},
    })
    # Identical paths, opposite roles, no data moved.
    assert weather.source.path == sensor.target.path == str(tmp_path / "KITTI")
    assert weather.target.path == sensor.source.path == str(tmp_path / "CADC")


def test_weather_is_a_config_attribute_not_a_folder_level(tmp_path):
    """CADC is the snow dataset; the weather type never appears on disk."""
    build_dataset(tmp_path / "CADC", train=2, test=1)
    config = Config.from_dict({
        "data_root": str(tmp_path),
        "source": {"dataset": "kitti", "path": str(tmp_path / "CADC")},
        "target": {"dataset": "cadc"},
        "task": {"type": "weather"},
        "geometric_degradation": {"enabled": True, "type": "physics",
                                  "weather": "snow", "precipitation_rate": 30.0},
    })
    assert config.target.path == str(tmp_path / "CADC")
    assert "snow" not in config.target.path.lower()
    assert config.geometric_degradation.weather == "snow"


def test_explicit_path_overrides_the_data_root(tmp_path):
    build_dataset(tmp_path / "elsewhere", train=1, test=1)
    config = Config.from_dict({
        "data_root": "Data",
        "source": {"dataset": "kitti", "path": str(tmp_path / "elsewhere")},
        "target": {"dataset": "cadc"},
        "task": {"type": "sensor"},
    })
    assert config.source.path == str(tmp_path / "elsewhere")
    assert config.target.path == "Data/CADC"


def test_a_dataset_with_no_location_is_reported_when_it_is_loaded():
    """A config may name a dataset without a location; loading one requires it.

    Validation stays permissive so a config can be inspected, planned and
    described without data present.
    """
    from reality.datasets.base import DatasetError

    config = Config.from_dict({"source": {"dataset": "kitti"},
                               "target": {"dataset": "cadc"},
                               "task": {"type": "sensor"}})
    assert config.source.path is None
    with pytest.raises(DatasetError, match="no location"):
        KittiAdapter(config.source)


def test_labels_are_discovered_under_the_dataset_folder(tmp_path):
    """Data/<name>/labels needs no declaration in config."""
    root = tmp_path / "KITTI"
    build_dataset(root, train=2, test=1, labels=True)
    config = Config.from_dict({
        "data_root": str(tmp_path),
        "source": {"dataset": "kitti"}, "target": {"dataset": "cadc",
                                                   "path": str(root)},
        "task": {"type": "sensor"},
    })
    adapter = KittiAdapter(config.source)
    assert adapter.label_root == root / "labels"
    sample = adapter[0]
    assert sample.meta.has_reflectance is True
    assert sample.meta.source_channels == 3


def test_a_dataset_without_labels_needs_no_special_casing(tmp_path):
    build_dataset(tmp_path / "CADC", train=2, test=1, labels=False)
    config = Config.from_dict({
        "data_root": str(tmp_path),
        "source": {"dataset": "cadc"}, "target": {"dataset": "cadc"},
        "task": {"type": "sensor"},
    })
    adapter = CadcAdapter(config.source)
    assert adapter.label_root is None
    assert adapter[0].meta.has_reflectance is False


def test_adding_a_dataset_is_a_folder_plus_a_config(tmp_path):
    """No code change: drop Data/<name>/ and name it in config."""
    build_dataset(tmp_path / "Boreas", train=4, test=2)
    config = Config.from_dict({
        "data_root": str(tmp_path),
        "source": {"dataset": "kitti", "path": str(tmp_path / "Boreas")},
        "target": {"dataset": "boreas"},
        "task": {"type": "sensor"},
    })
    from reality.datasets import BoreasAdapter

    assert config.target.path == str(tmp_path / "Boreas")
    # Boreas declares 5 columns; the folder here holds 4, so declare them.
    config.target.columns = ("x", "y", "z", "intensity")
    assert len(BoreasAdapter(config.target)) == 4
