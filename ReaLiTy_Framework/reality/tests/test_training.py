"""Training orchestration: cached prepare, full-set statistics, epochs, resume."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reality.core.config import Config
from reality.degradation import PhysicsWeatherDegradation
from reality.models import PicganAdapter
from reality.preprocessing.cache import CacheError, PrepareCache, config_key
from reality.preprocessing.statistics import PICGAN_DEFAULT_STATS, compute_from_cache
from reality.tests.test_dataset_layout import build_dataset
from reality.tests.test_degradation import FakeWeatherModel
from reality.training import Trainer, checkpoint as ckpt

TINY_SENSOR = {"proj_H": 32, "proj_W": 128, "fov_up": 3.0, "fov_down": -25.0}


def cloud_in_fov(n=400, seed=0, columns=4):
    """A cloud that lands inside the sensor's field of view."""
    rng = np.random.default_rng(seed)
    azimuth = rng.uniform(-np.pi, np.pi, n)
    radius = rng.uniform(4.0, 45.0, n)
    elevation = np.radians(rng.uniform(-24.0, 2.5, n))
    points = np.zeros((n, columns), dtype=np.float32)
    points[:, 0] = radius * np.cos(elevation) * np.cos(azimuth)
    points[:, 1] = radius * np.cos(elevation) * np.sin(azimuth)
    points[:, 2] = radius * np.sin(elevation)
    points[:, 3] = rng.uniform(0, 1, n)
    return points


def released_dataset(root, train=4, test=2, seed=0):
    """Write a domain in the released train/ + test/ layout."""
    for split, count in (("train", train), ("test", test)):
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            cloud_in_fov(seed=seed + i).tofile(directory / f"{i:06d}.bin")
    return root


def tiny_config(tmp_path, **overrides):
    source = released_dataset(tmp_path / "kitti_std", train=4, test=2, seed=0)
    target = released_dataset(tmp_path / "cadc_std", train=3, test=2, seed=100)
    data = {
        "source": {"dataset": "kitti", "path": str(source)},
        "target": {"dataset": "cadc", "path": str(target)},
        "task": {"type": "weather"},
        "geometric_degradation": {"enabled": True, "type": "physics",
                                  "weather": "snow", "precipitation_rate": 30.0},
        "sensor": dict(TINY_SENSOR),
        "model": {"type": "picgan"},
        "normalization": {"source": "computed"},
        "training": {"batch_size": 2, "epochs": 2, "learning_rate": 1.0e-5,
                     "lambda_cycle": 10, "lambda_physics": 10, "seed": 42,
                     "num_workers": 0},
        "output": {"checkpoint_dir": str(tmp_path / "run")},
    }
    data.update(overrides)
    return Config.from_dict(data)


def make_trainer(tmp_path, config=None, device="cpu"):
    config = config or tiny_config(tmp_path)
    degradation = PhysicsWeatherDegradation(config, weather_model=FakeWeatherModel())
    model = PicganAdapter(config, workspace=tmp_path / "ws", device=device)
    return Trainer(config, model=model, degradation=degradation,
                   cache_root=tmp_path / "cache")


# --------------------------------------------------------------------------- #
# The cache
# --------------------------------------------------------------------------- #


def test_cache_key_depends_on_what_changes_the_tensors(tmp_path):
    config = tiny_config(tmp_path)
    baseline = config_key(config)

    config.geometric_degradation.precipitation_rate = 45.0
    assert config_key(config) != baseline, "weather intensity changes the tensors"

    config.geometric_degradation.precipitation_rate = 30.0
    config.sensor.proj_W = 512
    assert config_key(config) != baseline, "projection width changes the tensors"


def test_cache_key_ignores_settings_that_do_not(tmp_path):
    """Changing the learning rate must not invalidate thousands of prepared frames."""
    config = tiny_config(tmp_path)
    baseline = config_key(config)
    config.training.learning_rate = 1.0e-3
    config.training.epochs = 999
    config.normalization.source = "picgan_default"
    assert config_key(config) == baseline


def test_prepare_writes_the_picgan_stack_layout(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.prepare()
    cache = trainer.cache

    assert cache.is_valid()
    assert len(cache.source_files()) == 4 and len(cache.target_files()) == 3
    source = np.load(cache.source_files()[0])
    target = np.load(cache.target_files()[0])
    assert source.shape == (3, 32, 128), "range, incidence, phy (no labels here)"
    assert target.shape == (2, 32, 128), "range, intensity"
    assert cache.source_channels == ("range", "incidence", "phy")
    assert cache.image_shape == (32, 128)


def test_prepare_is_skipped_when_a_valid_cache_exists(tmp_path):
    trainer = make_trainer(tmp_path)
    first = trainer.prepare()
    assert first.source_written == 4

    second = make_trainer(tmp_path).prepare()
    assert second.source_written == 0, "a valid cache must not be rebuilt"
    assert second.source_skipped == 4
    assert second.complete


def test_prepare_is_resumable(tmp_path):
    """A killed pass resumes rather than starting over."""
    trainer = make_trainer(tmp_path)
    trainer.prepare()
    cache = trainer.cache

    # Simulate an interrupted run: drop two frames and the completion flag.
    for path in cache.source_files()[:2]:
        path.unlink()
    cache.write_manifest(complete=False)
    assert not cache.is_valid()

    report = make_trainer(tmp_path).prepare()
    assert report.source_written == 2, "only the missing frames are redone"
    assert report.source_skipped == 2
    assert report.complete


def test_a_different_config_gets_a_different_cache(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.prepare()

    other = tiny_config(tmp_path)
    other.geometric_degradation.precipitation_rate = 10.0
    second = make_trainer(tmp_path, config=other)
    assert second.cache.directory != trainer.cache.directory
    assert not second.cache.is_valid(), "a new configuration must prepare afresh"


def test_force_rebuilds(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.prepare()
    assert trainer.prepare(force=True).source_written == 4


def test_incomplete_cache_is_refused(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.prepare()
    trainer.cache.source_files()[0].unlink()
    trainer.cache.write_manifest(complete=False)
    with pytest.raises(CacheError, match="not complete"):
        trainer.cache.require_complete()


def test_empty_source_is_reported(tmp_path):
    (tmp_path / "empty" / "train").mkdir(parents=True)
    (tmp_path / "empty" / "test").mkdir(parents=True)
    config = tiny_config(tmp_path)
    config.source.path = str(tmp_path / "empty")
    with pytest.raises(CacheError, match="nothing to prepare"):
        make_trainer(tmp_path, config=config).prepare()


# --------------------------------------------------------------------------- #
# Statistics over the complete prepared set
# --------------------------------------------------------------------------- #


def test_statistics_cover_every_prepared_frame(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.prepare()
    stats = compute_from_cache(trainer.cache, trainer.config)

    assert stats.mode == "computed"
    assert stats.n_source_frames == 4 and stats.n_target_frames == 3
    assert stats.seed is None, "every frame is used, so nothing is sampled"
    for name in ("range", "incidence", "phy", "intensity"):
        assert name in stats.channels, f"{name} must be measured"
    assert stats.pair("phy") != PICGAN_DEFAULT_STATS["phy"]


def test_statistics_refuse_a_partial_cache(tmp_path):
    """Measuring half a preparation would bake in the wrong constants."""
    trainer = make_trainer(tmp_path)
    trainer.prepare()
    trainer.cache.write_manifest(complete=False)
    with pytest.raises(CacheError, match="not complete"):
        compute_from_cache(trainer.cache, trainer.config)


def test_statistics_are_measured_on_occupied_pixels(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.prepare()
    stats = compute_from_cache(trainer.cache, trainer.config)

    stack = np.load(trainer.cache.source_files()[0])
    occupied = stack[0] > 0
    assert occupied.sum() < occupied.size, "the tiny frames leave empty pixels"
    assert stats.channels["range"].mean > float(stack[0].mean()), (
        "excluding empty pixels must raise the mean range"
    )


def test_statistics_are_in_range_image_form(tmp_path):
    """Constants describe the tensors the model sees, not the raw clouds."""
    trainer = make_trainer(tmp_path)
    trainer.prepare()
    stats = compute_from_cache(trainer.cache, trainer.config)
    total = sum(int((np.load(p)[0] > 0).sum()) for p in trainer.cache.source_files())
    assert stats.channels["range"].count == total


# --------------------------------------------------------------------------- #
# Training, checkpointing and resume
# --------------------------------------------------------------------------- #


def test_one_command_prepares_measures_and_trains(tmp_path):
    trainer = make_trainer(tmp_path)
    result = trainer.train(epochs=2)

    assert result["epochs_run"] == 2
    assert trainer.cache.is_valid()
    assert (trainer.run_dir / "config.snapshot.yaml").is_file()
    assert (trainer.run_dir / "normalization_stats.json").is_file()
    assert (trainer.run_dir / "train.log").is_file()
    assert (trainer.run_dir / ckpt.FULL_SUFFIX).is_file()
    assert (trainer.run_dir / ckpt.SLIM_SUFFIX).is_file()
    assert (trainer.run_dir / "metadata.json").is_file()

    log = (trainer.run_dir / "train.log").read_text()
    assert "normalization (computed)" in log
    assert "epoch" in log


def test_statistics_are_baked_into_both_checkpoints(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.train(epochs=1)
    measured = trainer.stats.pair("intensity")

    for name in (ckpt.FULL_SUFFIX, ckpt.SLIM_SUFFIX):
        loaded = ckpt.load(trainer.run_dir / name)
        assert loaded.stats.mode == "computed"
        assert loaded.stats.pair("intensity") == measured
        assert loaded.stats.n_source_frames == 4


def test_resume_continues_rather_than_restarting(tmp_path):
    first = make_trainer(tmp_path)
    first.train(epochs=2)
    assert ckpt.load(first.run_dir / ckpt.FULL_SUFFIX).epoch == 1

    second = make_trainer(tmp_path)
    result = second.train(epochs=4)
    assert result["start_epoch"] == 2, "must continue from the saved epoch"
    assert result["epochs_run"] == 2, "only the remaining epochs"
    assert ckpt.load(second.run_dir / ckpt.FULL_SUFFIX).epoch == 3


def test_resume_is_a_no_op_when_already_finished(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.train(epochs=2)
    assert make_trainer(tmp_path).train(epochs=2)["epochs_run"] == 0


def test_no_resume_starts_over(tmp_path):
    make_trainer(tmp_path).train(epochs=2)
    result = make_trainer(tmp_path).train(epochs=1, resume=False)
    assert result["start_epoch"] == 0


def test_second_run_reuses_the_cache(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.train(epochs=1)
    mtimes = {p: p.stat().st_mtime_ns for p in trainer.cache.source_files()}

    make_trainer(tmp_path).train(epochs=2)
    assert {p: p.stat().st_mtime_ns for p in trainer.cache.source_files()} == mtimes, (
        "a second run must not re-prepare a single frame"
    )


def test_losses_stay_finite_and_weights_update(tmp_path):
    trainer = make_trainer(tmp_path)
    result = trainer.train(epochs=6)
    model = trainer.pipeline.model
    for name in ("gen_R", "gen_S", "disc_R", "disc_S"):
        for parameter in getattr(model, name).parameters():
            assert torch.isfinite(parameter).all(), f"{name} went non-finite"
    assert any(entry["updated"] for entry in result["history"]), (
        "a real optimizer update must land past AMP warm-up"
    )


def test_training_scale_is_not_baked_in(tmp_path):
    """Whatever is in train/ is the training set."""
    config = tiny_config(tmp_path)
    released_dataset(tmp_path / "bigger", train=9, test=2, seed=500)
    config.source.path = str(tmp_path / "bigger")
    trainer = make_trainer(tmp_path, config=config)
    trainer.prepare()
    assert len(trainer.cache.source_files()) == 9


def test_run_directory_snapshot_reproduces_the_config(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.train(epochs=1)
    from reality.core.config import load_config

    assert load_config(trainer.run_dir / "config.snapshot.yaml") == trainer.config


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_training_runs_on_the_gpu(tmp_path):
    trainer = make_trainer(tmp_path, device="cuda")
    result = trainer.train(epochs=2)
    assert trainer.pipeline.model.device.type == "cuda"
    assert result["epochs_run"] == 2
