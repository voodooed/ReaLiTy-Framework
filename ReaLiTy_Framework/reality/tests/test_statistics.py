"""Data-derived normalization statistics and the computed/default switch."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from reality.core.config import Config
from reality.core.context import Sample, SampleMeta
from reality.preprocessing.projection import project
from reality.preprocessing.statistics import (
    PICGAN_DEFAULT_STATS,
    ChannelStats,
    NormalizationStats,
    StatisticsError,
    accumulate,
    compute_from_config,
    resolve,
    stats_path,
)

DATA_ROOT = Path(os.environ.get("REALITY_DATA_ROOT", "data"))
kitti_only = pytest.mark.skipif(not (DATA_ROOT / "KITTI").is_dir(),
                                reason="KITTI data not present")
cadc_only = pytest.mark.skipif(not (DATA_ROOT / "CADC").is_dir(),
                               reason="CADC data not present")


def projected(n=1500, seed=0, has_reflectance=True, with_phy=False, value=None):
    from reality.tests.test_projection import KITTI_SENSOR, make_cloud

    columns = ["x", "y", "z", "intensity"]
    if has_reflectance:
        columns.append("reflectance")
    if with_phy:
        columns.append("physics_intensity")
    sample = make_cloud(n, seed=seed, columns=tuple(columns),
                        has_reflectance=has_reflectance, sensor=KITTI_SENSOR)
    if value is not None:
        sample.points[:, 3] = value
    return project(sample)


# --------------------------------------------------------------------------- #
# Accumulation
# --------------------------------------------------------------------------- #


def test_accumulates_the_channels_present():
    stats = accumulate([projected()], ("range", "incidence", "reflectance", "intensity"))
    assert set(stats) == {"range", "incidence", "reflectance", "intensity"}
    for name, channel in stats.items():
        assert channel.count > 0
        assert channel.std >= 0
        assert channel.minimum <= channel.mean <= channel.maximum


def test_statistics_are_measured_over_occupied_pixels_only():
    """Empty pixels are absent returns; including them would encode proj_W."""
    sample = projected(has_reflectance=False, value=0.8)
    stats = accumulate([sample], ("intensity",))["intensity"]
    assert stats.mean == pytest.approx(0.8, abs=1e-5), "occupied pixels all hold 0.8"
    # The all-pixel mean is diluted by empties, and is kept only for reference.
    assert stats.all_pixel_mean < stats.mean
    assert stats.count == int((sample.mapping >= 0).sum())


def test_mean_and_std_match_numpy_over_the_mask():
    sample = projected(2000, seed=3, has_reflectance=False)
    occupied = sample.mapping >= 0
    stats = accumulate([sample], ("range",))["range"]
    values = sample.channel("range")[occupied].astype(np.float64)
    assert stats.mean == pytest.approx(values.mean(), rel=1e-9)
    assert stats.std == pytest.approx(values.std(), rel=1e-6)


def test_accumulates_across_frames():
    frames = [projected(800, seed=i, has_reflectance=False) for i in range(3)]
    combined = accumulate(frames, ("range",))["range"]
    total = sum(int((f.mapping >= 0).sum()) for f in frames)
    assert combined.count == total


def test_phy_is_measured_when_present():
    stats = accumulate([projected(with_phy=True)], ("phy",))
    assert "phy" in stats and stats["phy"].count > 0


def test_missing_channels_are_simply_absent():
    """KITTI has no phy until the weather model supplies it; that is not an error."""
    stats = accumulate([projected(has_reflectance=False)], ("range", "reflectance", "phy"))
    assert set(stats) == {"range"}


def test_unprojected_samples_are_refused():
    from reality.tests.test_projection import make_cloud

    with pytest.raises(StatisticsError, match="project first"):
        accumulate([make_cloud(10)], ("range",))


# --------------------------------------------------------------------------- #
# The switch
# --------------------------------------------------------------------------- #


def test_picgan_default_reproduces_the_published_constants_exactly():
    """`picgan_default` must keep prior results reproducible, to the digit."""
    stats = NormalizationStats.picgan_default()
    assert stats.mode == "picgan_default"
    assert stats.pair("range") == (0.0965, 0.1068)
    assert stats.pair("incidence") == (0.7156, 0.6352)
    assert stats.pair("reflectance") == (0.2979, 0.2743)
    assert stats.pair("phy") == (0.1745, 0.1515)
    assert stats.pair("intensity") == (0.0158, 0.0462)


def test_defaults_match_picgans_own_module():
    """The constants live in PICGAN; ReaLiTy must not hold a second copy."""
    import sys

    sys.path.insert(0, str(Path("reality/models/PICGAN").resolve()))
    try:
        import transform_utils
    finally:
        sys.path.pop(0)
    assert PICGAN_DEFAULT_STATS == transform_utils.PICGAN_DEFAULT_STATS


def test_unmeasured_channels_fall_back_to_the_default():
    stats = NormalizationStats(channels={"range": ChannelStats(mean=5.0, std=2.0)})
    assert stats.pair("range") == (5.0, 2.0)
    assert stats.pair("incidence") == PICGAN_DEFAULT_STATS["incidence"]


def test_zero_variance_channel_cannot_produce_a_zero_divisor():
    stats = NormalizationStats(channels={"range": ChannelStats(mean=1.0, std=0.0)})
    assert stats.pair("range")[1] > 0


def test_round_trips_through_json(tmp_path):
    stats = NormalizationStats(
        channels={"range": ChannelStats(mean=1.5, std=0.5, count=10)},
        mode="computed", source_dataset="kitti", target_dataset="cadc",
        n_source_frames=8, seed=0,
    )
    reloaded = NormalizationStats.load(stats.save(tmp_path / "stats.json"))
    assert reloaded.to_dict() == stats.to_dict()
    assert reloaded.pair("range") == (1.5, 0.5)


def test_saved_file_records_its_provenance(tmp_path):
    stats = NormalizationStats(channels={"range": ChannelStats(mean=1.0, std=1.0)},
                               source_dataset="kitti", target_dataset="cadc",
                               n_source_frames=4, n_target_frames=4, seed=7)
    data = json.loads(stats.save(tmp_path / "s.json").read_text())
    assert data["mode"] == "computed"
    assert data["source_dataset"] == "kitti" and data["target_dataset"] == "cadc"
    assert data["statistics_over"] == "occupied pixels only"
    assert data["seed"] == 7


def config_for(tmp_path, normalization="picgan_default", **kw):
    return Config.from_dict({
        "source": {"dataset": "kitti", "path": str(DATA_ROOT / "KITTI")},
        "target": {"dataset": "cadc", "path": str(DATA_ROOT / "CADC")},
        "task": {"type": "sensor"},
        "sensor": {"proj_H": 64, "proj_W": 1024, "fov_up": 3.0, "fov_down": -25.0},
        "normalization": {"source": normalization, **kw},
        "output": {"checkpoint_dir": str(tmp_path / "ckpt")},
    })


def test_resolve_honours_the_default_switch(tmp_path):
    stats = resolve(config_for(tmp_path, "picgan_default"))
    assert stats.mode == "picgan_default"
    assert not stats_path(config_for(tmp_path)).exists(), "defaults need no cache file"


# --------------------------------------------------------------------------- #
# Real data
# --------------------------------------------------------------------------- #


@kitti_only
@cadc_only
def test_computed_statistics_from_the_real_datasets(tmp_path):
    """Statistics must be measured from the degraded stacks, end to end."""
    config = config_for(tmp_path, "computed", frames=3)
    stats = compute_from_config(config)

    assert stats.mode == "computed"
    assert stats.source_dataset == "kitti" and stats.target_dataset == "cadc"
    assert stats.n_source_frames == 3 and stats.n_target_frames == 3

    for channel in ("range", "incidence", "reflectance"):
        assert channel in stats.channels, f"{channel} must be measured on the source"
    assert "intensity" in stats.channels, "intensity must be measured on the target"
    assert "phy" in stats.fallbacks, "KITTI has no phy until the weather model supplies it"

    # The finding that motivated the fix: KITTI incidence is nowhere near 0.7156.
    measured_incidence = stats.pair("incidence")[0]
    assert 0.2 < measured_incidence < 0.5
    assert abs(measured_incidence - PICGAN_DEFAULT_STATS["incidence"][0]) > 0.3

    for name, channel in stats.channels.items():
        assert np.isfinite(channel.mean) and np.isfinite(channel.std)
        assert channel.std > 0, f"{name} has no spread"


@kitti_only
@cadc_only
def test_target_intensity_is_measured_on_the_target_not_the_source(tmp_path):
    """Denormaweather_modeltion must use CADC's intensity, not KITTI's."""
    config = config_for(tmp_path, "computed", frames=2)
    stats = compute_from_config(config)

    from reality.core.config import DataSpec
    from reality.datasets import KittiAdapter
    from reality.preprocessing.projection import project as project_sample

    kitti = KittiAdapter(DataSpec(dataset="kitti", path=str(DATA_ROOT / "KITTI")),
                         sequences=["00"])
    kitti_intensity = accumulate([project_sample(kitti[0], config.sensor)], ("intensity",))
    assert stats.pair("intensity")[0] != pytest.approx(
        kitti_intensity["intensity"].mean, abs=1e-4
    ), "target statistics must not be the source's"


@kitti_only
@cadc_only
def test_computed_statistics_are_cached_and_reused(tmp_path):
    config = config_for(tmp_path, "computed", frames=2)
    first = resolve(config)
    path = stats_path(config)
    assert path.is_file(), "statistics must be cached beside the config snapshot"

    # A second resolve reuses the file rather than re-measuring.
    path.write_text(json.dumps({**first.to_dict(), "n_source_frames": 999}))
    assert resolve(config).n_source_frames == 999
    assert resolve(config, recompute=True).n_source_frames == 2


@kitti_only
@cadc_only
def test_frame_selection_is_reproducible(tmp_path):
    config = config_for(tmp_path, "computed", frames=3, seed=11)
    a = compute_from_config(config)
    b = compute_from_config(config)
    assert a.pair("range") == b.pair("range")
    assert a.pair("incidence") == b.pair("incidence")


# --------------------------------------------------------------------------- #
# : phy statistics come from the degraded cloud
# --------------------------------------------------------------------------- #


@kitti_only
@cadc_only
def test_phy_is_measured_after_degradation(tmp_path):
    """phy exists only once the weather model has run, so it must be measured post-degradation."""
    from reality.degradation import PhysicsWeatherDegradation
    from reality.tests.test_degradation import FakeWeatherModel

    config = Config.from_dict({
        "source": {"dataset": "kitti", "path": str(DATA_ROOT / "KITTI"),
                   "sequences": ["00"]},
        "target": {"dataset": "cadc", "path": str(DATA_ROOT / "CADC")},
        "task": {"type": "weather"},
        "geometric_degradation": {"enabled": True, "type": "physics",
                                  "weather": "snow", "precipitation_rate": 30.0},
        "sensor": {"proj_H": 32, "proj_W": 256, "fov_up": 3.0, "fov_down": -25.0},
        "normalization": {"source": "computed", "frames": 2},
        "output": {"checkpoint_dir": str(tmp_path / "ckpt")},
    })
    stats = compute_from_config(
        config, degradation=PhysicsWeatherDegradation(config, weather_model=FakeWeatherModel()))

    assert "phy" in stats.channels, "the weather path must measure phy"
    assert stats.fallbacks == [], f"nothing should fall back, got {stats.fallbacks}"
    assert stats.pair("phy") != PICGAN_DEFAULT_STATS["phy"]
    assert 0.0 < stats.pair("phy")[0] < 1.0


@kitti_only
@cadc_only
def test_missing_degradation_warns_that_phy_is_unmeasured(tmp_path, monkeypatch):
    """Silently normalising phy with VoxelScape constants would be the worst outcome.

    the weather model's absence is simulated so this holds whether or not it is installed here.
    """
    import warnings as warnings_module

    monkeypatch.setattr("reality.degradation.physics_weather.find_weather_model",
                        lambda *args, **kwargs: None)

    config = Config.from_dict({
        "source": {"dataset": "kitti", "path": str(DATA_ROOT / "KITTI"),
                   "sequences": ["00"]},
        "target": {"dataset": "cadc", "path": str(DATA_ROOT / "CADC")},
        "task": {"type": "weather"},
        "geometric_degradation": {"enabled": True, "type": "physics",
                                  "weather": "snow", "precipitation_rate": 30.0},
        "sensor": {"proj_H": 32, "proj_W": 256, "fov_up": 3.0, "fov_down": -25.0},
        "normalization": {"source": "computed", "frames": 1},
        "output": {"checkpoint_dir": str(tmp_path / "ckpt")},
    })
    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        stats = compute_from_config(config)  # no the weather model available
    assert "phy" in stats.fallbacks
    assert any("phy statistics could not be measured" in str(w.message) for w in caught)
