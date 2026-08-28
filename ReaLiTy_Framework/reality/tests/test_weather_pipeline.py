"""End-to-end weather pipeline: KITTI -> the weather model -> PICGAN -> CADC.

The point of these tests is "runs clean", not convergence: no shape, dtype or
index errors anywhere, phy actually produced by the degradation stage, a real
optimizer update past AMP warm-up, and output that reloads through the adapter.

the weather model is GPL-3.0 and not vendored, so the default run uses the contract-faithful
stand-in from test_degradation; with $REALITY_WEATHER_MODEL_PATH set the same tests run
against the real package.
"""

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reality.core.config import Config
from reality.core.determinism import seed_everything
from reality.datasets import CadcAdapter, KittiAdapter
from reality.degradation import LABEL_COLUMN, PHYSICS_COLUMN, PhysicsWeatherDegradation
from reality.degradation.physics_weather import find_weather_model
from reality.io import OutputWriter
from reality.models import PicganAdapter
from reality.postprocessing.backprojection import backproject
from reality.preprocessing.statistics import ChannelStats, NormalizationStats
from reality.tests.test_degradation import FakeWeatherModel
from reality.training import WeatherPipeline

DATA_ROOT = Path(os.environ.get("REALITY_DATA_ROOT", "data"))
real_data_only = pytest.mark.skipif(
    not (DATA_ROOT / "KITTI").is_dir() or not (DATA_ROOT / "CADC").is_dir(),
    reason="KITTI/CADC data not present")

REAL_WEATHER_MODEL = find_weather_model(os.environ.get("REALITY_WEATHER_MODEL_PATH"))

#: A small image keeps the tiny-subset run quick without changing any code path.
TINY_SENSOR = {"proj_H": 32, "proj_W": 256, "fov_up": 3.0, "fov_down": -25.0}


def usable_cuda() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.mm(torch.randn(4, 4, device="cuda"), torch.randn(4, 4, device="cuda"))
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


DEVICE = "cuda" if usable_cuda() else "cpu"


def tiny_config(tmp_path, **overrides):
    data = {
        "source": {"dataset": "kitti", "path": str(DATA_ROOT / "KITTI")},
        "target": {"dataset": "cadc", "path": str(DATA_ROOT / "CADC")},
        "task": {"type": "weather"},
        "geometric_degradation": {"enabled": True, "type": "physics",
                                  "weather": "snow", "precipitation_rate": 30.0,
                                  **overrides.pop("degradation", {})},
        "sensor": dict(TINY_SENSOR),
        "model": {"type": "picgan"},
        "normalization": {"source": "computed", "frames": 2, "seed": 0},
        "training": {"batch_size": 1, "epochs": 1, "learning_rate": 1.0e-5,
                     "lambda_cycle": 10, "lambda_physics": 10, "seed": 42},
        "output": {"checkpoint_dir": str(tmp_path / "run")},
    }
    data.update(overrides)
    return Config.from_dict(data)


def measured_stats():
    """Stand-in for measured constants, so the run does not re-scan the datasets."""
    return NormalizationStats(
        channels={"range": ChannelStats(mean=10.79, std=9.12),
                  "incidence": ChannelStats(mean=0.3616, std=0.2379),
                  "reflectance": ChannelStats(mean=0.3387, std=0.1592),
                  "phy": ChannelStats(mean=0.05, std=0.06),
                  "intensity": ChannelStats(mean=0.0511, std=0.0749)},
        mode="computed", source_dataset="kitti", target_dataset="cadc")


def build_pipeline(tmp_path, weather_model=None, stats=None, device=DEVICE):
    config = tiny_config(tmp_path)
    degradation = PhysicsWeatherDegradation(
        config, weather_model=weather_model if weather_model is not None else FakeWeatherModel(),
        weather_model_path=str(REAL_WEATHER_MODEL) if (weather_model is None and REAL_WEATHER_MODEL) else None,
    ) if weather_model is not False else None
    if REAL_WEATHER_MODEL and weather_model is None:
        degradation = PhysicsWeatherDegradation(config, weather_model_path=str(REAL_WEATHER_MODEL))
    model = PicganAdapter(config, workspace=tmp_path / "ws", device=device,
                          stats=stats or measured_stats())
    return WeatherPipeline(config, model=model, degradation=degradation,
                           stats=stats or measured_stats())


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def test_weather_pipeline_needs_a_weather_task(tmp_path):
    from reality.training import TrainingError

    config = tiny_config(tmp_path)
    config.task.type = "sensor"
    with pytest.raises(TrainingError, match="task.type='weather'"):
        WeatherPipeline(config)


def test_shipped_config_describes_the_real_experiment():
    config = Config.load("reality/configs/weather/kitti_to_cadc.yaml")
    assert config.run_name == "kitti_to_cadc"
    assert config.task.is_weather
    assert config.geometric_degradation.type == "physics"
    assert config.geometric_degradation.weather == "snow"
    assert config.normalization.source == "computed"
    assert config.sensor.proj_H == 64 and config.sensor.fov_down == -25.0


def test_stage_order_puts_degradation_before_projection(tmp_path):
    """the weather model consumes a 3D cloud, so it cannot run after projection."""
    from reality.core.pipeline import plan_stages

    roles = [s.role for s in plan_stages(tiny_config(tmp_path))]
    assert roles == ["degradation", "projection", "model", "backprojection"]


# --------------------------------------------------------------------------- #
# The tiny-subset end-to-end run
# --------------------------------------------------------------------------- #


@real_data_only
def test_tiny_subset_runs_clean_end_to_end(tmp_path):
    """The deliverable: 6 steps over a few real frames with no errors anywhere."""
    seed_everything(42, deterministic=True)
    pipeline = build_pipeline(tmp_path)

    results = pipeline.run_steps(n_steps=6, n_source=4, n_target=4, batch_size=1)

    assert len(results) == 6
    assert all(r.source_channels == 3 for r in results), "KITTI labels -> 3 channels"
    assert all(r.image_shape == (32, 256) for r in results)

    # AMP skips the first steps while the loss scale calibrates down; a real
    # update must land within the run.
    updated = [r.step for r in results if r.weights_changed]
    assert updated, f"no optimizer update in 6 steps (loss scales: {[r.loss_scale for r in results]})"
    assert min(updated) <= 5, f"first update at step {min(updated)}, expected by step 5"
    assert results[-1].loss_scale < 65536.0, "the scaler must have calibrated"

    # Every parameter must be finite after training.
    for name in ("gen_R", "gen_S", "disc_R", "disc_S"):
        for parameter in getattr(pipeline.model, name).parameters():
            assert torch.isfinite(parameter).all(), f"{name} went non-finite"

    stats = results[0].degradation
    assert stats["weather"] == "snow"
    assert stats["n_lost"] + stats["n_scattered"] + stats["n_unchanged"] > 0


@real_data_only
def test_the_three_tuple_reaches_the_model_with_correct_shapes(tmp_path):
    pipeline = build_pipeline(tmp_path)
    source_adapter = pipeline.adapter(pipeline.config.source)
    target_adapter = pipeline.adapter(pipeline.config.target)

    source = pipeline.prepare_source(source_adapter[0])
    target = pipeline.prepare_target(target_adapter[0])

    assert source.phy is not None, "phy must come from the weather model"
    assert source.meta.has_reflectance is True
    assert target.meta.has_reflectance is False, "CADC has no per-point labels"

    sim, real, phy = pipeline.build_batch([source], [target])
    assert sim.shape == (1, 3, 32, 256)
    assert real.shape == (1, 1, 32, 256)
    assert phy.shape == (1, 1, 32, 256)
    assert sim.dtype == real.dtype == phy.dtype == torch.float32
    assert torch.isfinite(sim).all() and torch.isfinite(real).all()
    assert torch.isfinite(phy).all()
    assert pipeline.model.stats.mode == "computed"


@real_data_only
def test_labelled_source_and_unlabelled_target_combination(tmp_path):
    """KITTI has labels (3-channel source); CADC has none (intensity-only target)."""
    pipeline = build_pipeline(tmp_path)
    source = pipeline.prepare_source(pipeline.adapter(pipeline.config.source)[0])
    target = pipeline.prepare_target(pipeline.adapter(pipeline.config.target)[0])

    assert source.channels == ("range", "incidence", "reflectance", "intensity", "mask")
    assert target.channels == ("range", "incidence", "intensity", "mask")
    assert pipeline.model.channels_for(source) == 3
    result = pipeline.train_step([source], [target], step=1)
    assert result.source_channels == 3


@real_data_only
def test_phy_comes_from_weather_model_not_from_the_source_intensity(tmp_path):
    pipeline = build_pipeline(tmp_path)
    adapter = pipeline.adapter(pipeline.config.source)
    clean = adapter[0]
    degraded = pipeline.prepare_source(clean)

    assert PHYSICS_COLUMN in degraded.meta.columns
    physics = degraded.point_column(PHYSICS_COLUMN)
    assert not np.allclose(physics, clean.point_column("intensity")), (
        "the physics intensity must be the weather model's ref_new, not the clear-weather value"
    )
    occupied = degraded.mapping >= 0
    owners = degraded.mapping[occupied]
    assert np.allclose(degraded.phy[0][occupied], physics[owners], atol=1e-6)


@real_data_only
def test_backprojection_round_trips_against_the_degraded_cloud(tmp_path):
    """Geometry byte-intact, and the alignment guard satisfied."""
    pipeline = build_pipeline(tmp_path)
    source = pipeline.prepare_source(pipeline.adapter(pipeline.config.source)[0])

    if not pipeline.model.is_built:
        pipeline.model.build_model(3)
    generated = pipeline.model.denormalize_real_intensity(
        pipeline.model.generate(source))
    intensity = generated.detach().cpu().numpy()[0]

    result = backproject(source, intensity, fill="nan")
    assert np.array_equal(result.points[:, :3], source.points[:, :3])
    assert result.n_written == int((source.mapping >= 0).sum())
    lost = source.point_column(LABEL_COLUMN) == 0
    assert not result.written[lost].any(), "lost points must never be written"
    assert np.isfinite(result.points[result.written, 3]).all()


@real_data_only
def test_output_writes_and_reloads_through_the_adapter(tmp_path):
    pipeline = build_pipeline(tmp_path)
    source = pipeline.prepare_source(pipeline.adapter(pipeline.config.source)[0])
    if not pipeline.model.is_built:
        pipeline.model.build_model(3)
    generated = pipeline.model.denormalize_real_intensity(
        pipeline.model.generate(source)).detach().cpu().numpy()[0]
    result = backproject(source, generated)

    velodyne = tmp_path / "out/data_odometry_velodyne/dataset/sequences/00/velodyne"
    writer = OutputWriter(velodyne, fmt="bin", columns=("x", "y", "z", "intensity"))
    path = writer.write(source, result.points, frame_id="000000")

    from reality.core.config import DataSpec

    reloaded = KittiAdapter(DataSpec(dataset="kitti", path=str(tmp_path / "out")))[0]
    assert reloaded.points.shape == (source.points.shape[0], 4)
    assert np.allclose(reloaded.points[:, :3], source.points[:, :3])
    assert path.stat().st_size == source.points.shape[0] * 4 * 4


@real_data_only
def test_checkpoint_written_after_the_tiny_run_reloads(tmp_path):
    """What travels to a cluster and back must round-trip."""
    from reality.training import checkpoint as ckpt

    pipeline = build_pipeline(tmp_path)
    pipeline.run_steps(n_steps=5, n_source=2, n_target=2)
    paths = ckpt.save(pipeline.model, tmp_path / "ckpt", pipeline.config, epoch=1)

    loaded = ckpt.load(paths["slim"])
    assert loaded.stats.mode == "computed"
    assert loaded.metadata["weather"] == "snow"
    assert loaded.metadata["degradation"] == "physics"
    assert loaded.in_channels_s == 3


@real_data_only
@pytest.mark.skipif(DEVICE != "cuda", reason="no usable CUDA device")
def test_the_tiny_run_happens_on_the_gpu(tmp_path):
    pipeline = build_pipeline(tmp_path, device="cuda")
    results = pipeline.run_steps(n_steps=5, n_source=2, n_target=2)
    assert pipeline.model.device.type == "cuda"
    assert next(pipeline.model.gen_R.parameters()).is_cuda
    assert any(r.weights_changed for r in results)
