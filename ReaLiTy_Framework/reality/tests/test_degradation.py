"""the weather model degradation plugin: wiring, phy production, and dropped-point handling.

the weather model itself is GPL-3.0 and not vendored, so these tests drive the plugin with a
stand-in that implements the upstream contract exactly as read from
``atmos_models.py``: ``augment(pc, Rr) -> (N, 5)`` of ``[x, y, z, ref_new, label]``
with label 0 lost / 1 scattered / 2 unchanged, and lost points moved to the
origin with zero reflectivity. Tests against the real package are skip-guarded.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from reality.core.config import Config
from reality.core.context import Sample, SampleMeta
from reality.core.registry import DEGRADATIONS
from reality.degradation import (
    LABEL_COLUMN,
    PHYSICS_COLUMN,
    DegradationError,
    GeometricDegradation,
    LearnedDegradation,
    WeatherModelUnavailable,
    PhysicsWeatherDegradation,
)
from reality.degradation.physics_weather import find_weather_model
from reality.postprocessing.backprojection import backproject
from reality.preprocessing.projection import project
from reality.tests.test_projection import KITTI_SENSOR, make_cloud


class FakeWeatherModel:
    """Stand-in implementing upstream the weather model's documented augment contract."""

    def __init__(self, lost_fraction=0.2, scattered_fraction=0.2, seed=0, **kwargs):
        self.kwargs = kwargs
        self.calls = 0
        self.lost_fraction = lost_fraction
        self.scattered_fraction = scattered_fraction
        self.rng = np.random.default_rng(seed)

    def augment(self, pc, Rr):
        self.calls += 1
        self.last_rain_rate = Rr
        n = len(pc)
        out = np.zeros((n, 5))
        out[:, :4] = pc
        draw = self.rng.random(n)
        lost = draw < self.lost_fraction
        scattered = (draw >= self.lost_fraction) & (
            draw < self.lost_fraction + self.scattered_fraction)
        unchanged = ~(lost | scattered)

        out[unchanged, 4] = 2
        out[unchanged, 3] = pc[unchanged, 3] * 0.8          # attenuated
        out[scattered, 4] = 1
        out[scattered, 3] = pc[scattered, 3] * 0.3          # scattered
        out[scattered, :3] = pc[scattered, :3] * 0.9        # pulled closer
        # Lost points: upstream sets range and reflectivity to zero, which puts
        # them at the origin.
        out[lost, :4] = 0.0
        out[lost, 4] = 0
        return out


def weather_config(tmp_path=None, **overrides):
    data = {
        "source": {"dataset": "kitti", "path": "/tmp/src"},
        "target": {"dataset": "cadc", "path": "/tmp/tgt"},
        "task": {"type": "weather"},
        "geometric_degradation": {"enabled": True, "type": "physics",
                                  "weather": "snow", "precipitation_rate": 30.0,
                                  **overrides},
        "sensor": {"proj_H": 64, "proj_W": 1024, "fov_up": 3.0, "fov_down": -25.0},
        "normalization": {"source": "picgan_default"},
    }
    if tmp_path is not None:
        data["output"] = {"checkpoint_dir": str(tmp_path / "ckpt")}
    return Config.from_dict(data)


def kitti_like(n=2000, seed=0):
    """A labelled KITTI-style cloud: x, y, z, intensity, label, reflectance."""
    sample = make_cloud(n, seed=seed,
                        columns=("x", "y", "z", "intensity", "label", "reflectance"),
                        has_reflectance=True, sensor=KITTI_SENSOR)
    sample.points[:, 3] = np.clip(sample.points[:, 3], 0.0, 1.0)
    sample.points[:, 5] = np.clip(sample.points[:, 5], 0.0, 1.0)
    return sample


@pytest.fixture
def plugin():
    return PhysicsWeatherDegradation(weather_config(), weather_model=FakeWeatherModel())


# --------------------------------------------------------------------------- #
# Registration and configuration
# --------------------------------------------------------------------------- #


def test_registered_as_a_degradation():
    assert DEGRADATIONS.get("physics") is PhysicsWeatherDegradation
    assert DEGRADATIONS.get("learned") is LearnedDegradation
    assert issubclass(PhysicsWeatherDegradation, GeometricDegradation)


def test_learned_is_an_interface_stub_only():
    stub = LearnedDegradation(weather_config(type="learned"))
    with pytest.raises(NotImplementedError, match="interface stub"):
        stub.apply(kitti_like(10))


def test_weather_selects_the_atmospheric_model():
    assert PhysicsWeatherDegradation(weather_config(), weather_model=FakeWeatherModel()).atm_model == "snow"
    rain = weather_config(weather="rain")
    assert PhysicsWeatherDegradation(rain, weather_model=FakeWeatherModel()).atm_model == "rain"


def test_precipitation_rate_reaches_weather_model(plugin):
    plugin.apply(kitti_like(500))
    assert plugin.weather_model.last_rain_rate == 30.0


def test_weather_model_parameters_come_from_config():
    config = weather_config(mode="last", rmax=120.0, rmin=2.0, bdiv=5.0e-3)
    spec = config.geometric_degradation
    assert (spec.mode, spec.rmax, spec.rmin, spec.bdiv) == ("last", 120.0, 2.0, 5.0e-3)


def test_unsupported_weather_is_refused():
    config = weather_config()
    config.geometric_degradation.weather = "fog"
    with pytest.raises(DegradationError, match="supports"):
        PhysicsWeatherDegradation(config, weather_model=FakeWeatherModel())


def test_disabled_degradation_is_refused():
    config = Config.from_dict({
        "source": {"dataset": "kitti"}, "target": {"dataset": "kitti"},
        "task": {"type": "sensor"}})
    with pytest.raises(DegradationError, match="not enabled"):
        PhysicsWeatherDegradation(config, weather_model=FakeWeatherModel())


def test_missing_weather_model_explains_how_to_provide_it():
    plugin = PhysicsWeatherDegradation(weather_config())
    plugin.weather_model_path = None
    with pytest.raises(WeatherModelUnavailable, match="REALITY_WEATHER_MODEL_PATH"):
        _ = plugin.weather_model


def test_weather_model_is_located_by_config_or_environment(tmp_path, monkeypatch):
    """Config path, then $REALITY_WEATHER_MODEL_PATH, then the vendored directories."""
    from reality.degradation.physics_weather import REPO_MODEL, VENDORED_MODEL

    (tmp_path / "atmos_models.py").write_text("class the weather model: pass\n")
    assert find_weather_model(tmp_path) == tmp_path
    monkeypatch.setenv("REALITY_WEATHER_MODEL_PATH", str(tmp_path))
    assert find_weather_model(None) == tmp_path

    monkeypatch.delenv("REALITY_WEATHER_MODEL_PATH")
    fallback = find_weather_model(None)
    if (VENDORED_MODEL / "atmos_models.py").is_file():
        assert fallback == VENDORED_MODEL
    elif (REPO_MODEL / "atmos_models.py").is_file():
        assert fallback == REPO_MODEL
    else:
        assert fallback is None


def test_missing_mie_table_is_refused_rather_than_recomputed(tmp_path):
    """Upstream would silently recompute for minutes and not match the published table."""
    (tmp_path / "atmos_models.py").write_text("class the weather model: pass\n")
    from reality.degradation.physics_weather import load_weather_model_class

    with pytest.raises(WeatherModelUnavailable, match="mie_q.npz"):
        load_weather_model_class(tmp_path)


def test_weather_model_is_constructed_once_and_reused():
    """Mie setup is the expensive part; it must not repeat per frame."""
    constructions = []

    class CountingWeatherModel(FakeWeatherModel):
        def __init__(self, **kwargs):
            constructions.append(kwargs)
            super().__init__(**kwargs)

    plugin = PhysicsWeatherDegradation(weather_config(), weather_model=CountingWeatherModel())
    for _ in range(3):
        plugin.apply(kitti_like(200))
    assert len(constructions) == 1
    assert plugin.weather_model.calls == 3


# --------------------------------------------------------------------------- #
# phy wiring
# --------------------------------------------------------------------------- #


def test_physics_intensity_is_weather_models_ref_new(plugin):
    sample = kitti_like(1000)
    clean = sample.raw_cloud().astype(np.float64)
    degraded = plugin.apply(sample)

    assert PHYSICS_COLUMN in degraded.meta.columns
    physics = degraded.point_column(PHYSICS_COLUMN)
    assert physics.shape == (1000,)
    # ref_new also becomes the cloud's intensity: it is the degraded return.
    assert np.allclose(physics, degraded.point_column("intensity"))
    assert not np.allclose(physics, clean[:, 3]), "the weather model must change the reflectivity"


def test_phy_is_populated_by_projecting_the_degraded_cloud(plugin):
    degraded = plugin.apply(kitti_like(2000))
    projected = project(degraded, KITTI_SENSOR)

    assert projected.phy is not None, "phy must come from the weather model, not from PICGAN"
    assert projected.phy.shape == (1, 64, 1024)
    rows, cols = np.nonzero(projected.mapping >= 0)
    owners = projected.mapping[rows, cols]
    assert np.allclose(projected.phy[0][rows, cols],
                       degraded.point_column(PHYSICS_COLUMN)[owners], atol=1e-6)


def test_reflectance_survives_degradation(plugin):
    """A material's reflectance does not change with weather."""
    sample = kitti_like(800)
    before = sample.point_column("reflectance").copy()
    degraded = plugin.apply(sample)
    assert np.allclose(degraded.point_column("reflectance"), before)
    assert degraded.meta.has_reflectance is True


def test_three_channel_source_survives_degradation(plugin):
    projected = project(plugin.apply(kitti_like(1500)), KITTI_SENSOR)
    assert projected.channels == ("range", "incidence", "reflectance", "intensity", "mask")
    assert projected.meta.source_channels == 3


def test_two_channel_source_also_works():
    """An unlabelled source stays 2-channel through the weather path."""
    sample = make_cloud(800, columns=("x", "y", "z", "intensity"),
                        has_reflectance=False, sensor=KITTI_SENSOR)
    sample.points[:, 3] = np.clip(sample.points[:, 3], 0, 1)
    plugin = PhysicsWeatherDegradation(weather_config(), weather_model=FakeWeatherModel())
    projected = project(plugin.apply(sample), KITTI_SENSOR)
    assert projected.meta.source_channels == 2
    assert projected.phy is not None


def test_degradation_counts_are_recorded(plugin):
    stats = plugin.apply(kitti_like(1000)).meta.extra["degradation"]
    assert stats["weather"] == "snow"
    assert stats["precipitation_rate"] == 30.0
    assert stats["n_lost"] + stats["n_scattered"] + stats["n_unchanged"] == 1000
    assert stats["n_lost"] > 0


# --------------------------------------------------------------------------- #
# Dropped points and the back-projection guard
# --------------------------------------------------------------------------- #


def test_lost_points_are_dropped_not_treated_as_returns(plugin):
    """the weather model puts lost points at the origin; they must never own a pixel."""
    degraded = plugin.apply(kitti_like(2000))
    labels = degraded.point_column(LABEL_COLUMN)
    lost = labels == 0
    assert lost.any()
    assert np.allclose(degraded.points[lost, :3], 0.0), "upstream zeroes lost points"

    projected = project(degraded, KITTI_SENSOR)
    owners = set(projected.mapping[projected.mapping >= 0].tolist())
    assert not owners & set(np.nonzero(lost)[0].tolist()), (
        "a zero-range point at the origin is not a return and must own no pixel"
    )


def test_backprojection_guard_passes_against_the_degraded_cloud(plugin):
    """the weather model preserves N, so the alignment guard is satisfied."""
    sample = kitti_like(1200)
    degraded = plugin.apply(sample)
    assert degraded.points.shape[0] == sample.points.shape[0]

    projected = project(degraded, KITTI_SENSOR)
    result = backproject(projected, projected.channel("range"), cloud=degraded.points)
    assert np.array_equal(result.points[:, :3], degraded.points[:, :3])
    assert np.allclose(result.points[result.written, 3],
                       np.linalg.norm(degraded.points[result.written, :3], axis=1),
                       rtol=1e-4, atol=1e-3)


def test_backprojection_refuses_the_undegraded_cloud(plugin):
    """Mapping built on the degraded cloud must not be applied to a different one."""
    sample = kitti_like(1000)
    projected = project(plugin.apply(sample), KITTI_SENSOR)
    from reality.postprocessing.backprojection import BackProjectionError

    with pytest.raises(BackProjectionError, match="same one that was projected"):
        backproject(projected, projected.channel("range"), cloud=sample.points[:500])


def test_lost_points_are_never_written(plugin):
    degraded = plugin.apply(kitti_like(1500))
    projected = project(degraded, KITTI_SENSOR)
    result = backproject(projected, projected.channel("range"), fill="nan")
    lost = degraded.point_column(LABEL_COLUMN) == 0
    assert not result.written[lost].any()
    assert np.isnan(result.points[lost, 3]).all(), "explicitly marked, not written"


# --------------------------------------------------------------------------- #
# Contract enforcement
# --------------------------------------------------------------------------- #


def test_projected_sample_is_refused(plugin):
    projected = project(kitti_like(200), KITTI_SENSOR)
    with pytest.raises(DegradationError, match="must run before projection"):
        plugin.apply(projected)


def test_out_of_range_reflectivity_is_refused(plugin):
    """the weather model documents reflectivity in [0, 1] and range in metres."""
    sample = kitti_like(100)
    sample.points[:, 3] = 42.0
    with pytest.raises(DegradationError, match=r"normalised to \[0, 1\]"):
        plugin.apply(sample)


def test_wrong_augment_shape_is_reported():
    class BadWeatherModel(FakeWeatherModel):
        def augment(self, pc, Rr):
            return np.zeros((len(pc), 4))

    plugin = PhysicsWeatherDegradation(weather_config(), weather_model=BadWeatherModel())
    with pytest.raises(DegradationError, match=r"expected \(\d+, 5\)"):
        plugin.apply(kitti_like(50))


# --------------------------------------------------------------------------- #
# Against the bundled scattering model
# --------------------------------------------------------------------------- #

BUNDLED_MODEL = find_weather_model(os.environ.get("REALITY_WEATHER_MODEL_PATH"))
bundled_only = pytest.mark.skipif(
    BUNDLED_MODEL is None,
    reason="no weather model available (bundled model missing and none configured)")


def test_weather_model_is_bundled_with_the_package():
    """The weather path works out of the box: no external setup step."""
    from reality.degradation.physics_weather import MIE_TABLE, VENDORED_MODEL

    assert (VENDORED_MODEL / "atmos_models.py").is_file()
    assert (VENDORED_MODEL / "atmos_models_cpu.py").is_file(), "CPU fallback ships too"
    assert (VENDORED_MODEL / MIE_TABLE).is_file(), "the Mie table must travel with it"
    assert (VENDORED_MODEL / "LICENSE.md").is_file(), "GPL text travels with the code"
    assert find_weather_model(None) == VENDORED_MODEL


@bundled_only
def test_bundled_model_augments_a_cloud():
    plugin = PhysicsWeatherDegradation(weather_config())
    degraded = plugin.apply(kitti_like(500))

    labels = degraded.point_column(LABEL_COLUMN)
    assert set(np.unique(labels)) <= {0.0, 1.0, 2.0}
    physics = degraded.point_column(PHYSICS_COLUMN)
    assert np.isfinite(physics).all()
    assert 0.0 <= physics.min() and physics.max() <= 1.0


@bundled_only
def test_cpu_implementation_is_selectable():
    """atmos_models_cpu is the unmodified upstream implementation, kept usable."""
    plugin = PhysicsWeatherDegradation(
        weather_config(weather_model_module="atmos_models_cpu"))
    assert plugin.weather_model_module == "atmos_models_cpu"
    degraded = plugin.apply(kitti_like(200))
    labels = degraded.point_column(LABEL_COLUMN)
    assert set(np.unique(labels)) <= {0.0, 1.0, 2.0}
    assert np.isfinite(degraded.point_column(PHYSICS_COLUMN)).all()


@bundled_only
def test_external_model_path_still_overrides_the_bundled_one():
    """Bundling is the default, not a lock-in."""
    from reality.degradation.physics_weather import VENDORED_MODEL

    plugin = PhysicsWeatherDegradation(
        weather_config(weather_model_path=str(VENDORED_MODEL)))
    assert plugin.weather_model_path == VENDORED_MODEL
