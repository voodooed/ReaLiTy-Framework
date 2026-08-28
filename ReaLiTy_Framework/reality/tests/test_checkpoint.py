"""Checkpoints: full resumable state, slim inference file, and ONNX export."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reality.core.config import Config
from reality.models import PicganAdapter
from reality.preprocessing.statistics import ChannelStats, NormalizationStats
from reality.training import checkpoint as ckpt

H, W = 32, 64


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    return tmp_path_factory.mktemp("ckpt_ws")


def measured_stats():
    """Statistics as a cluster run would have measured them over the full set."""
    return NormalizationStats(
        channels={
            "range": ChannelStats(mean=10.7928, std=9.1174),
            "incidence": ChannelStats(mean=0.3616, std=0.2379),
            "reflectance": ChannelStats(mean=0.3387, std=0.1592),
            "intensity": ChannelStats(mean=0.0511, std=0.0749),
        },
        mode="computed", source_dataset="kitti", target_dataset="cadc",
        n_source_frames=23201, n_target_frames=5701, seed=0,
    )


@pytest.fixture
def model(workspace):
    torch.manual_seed(7)
    model = PicganAdapter(workspace=workspace, device="cpu", stats=measured_stats())
    model.build_model(3)
    return model


def weather_config(tmp_path):
    return Config.from_dict({
        "source": {"dataset": "kitti", "path": "/tmp/k"},
        "target": {"dataset": "cadc", "path": "/tmp/c"},
        "task": {"type": "weather"},
        "geometric_degradation": {"enabled": True, "type": "physics",
                                  "weather": "snow", "precipitation_rate": 30.0},
        "normalization": {"source": "computed"},
        "output": {"checkpoint_dir": str(tmp_path / "run")},
    })


# --------------------------------------------------------------------------- #
# Full checkpoint
# --------------------------------------------------------------------------- #


def test_full_checkpoint_round_trips(model, workspace, tmp_path):
    config = weather_config(tmp_path)
    path = ckpt.save_full(model, tmp_path / "full.pt", config, epoch=12)

    restored = PicganAdapter(workspace=workspace, device="cpu")
    epoch = ckpt.restore(restored, path)

    assert epoch == 12
    assert restored.in_channels_s == 3
    for a, b in zip(model.gen_R.state_dict().values(),
                    restored.gen_R.state_dict().values()):
        assert torch.equal(a, b)
    for name in ("gen_S", "disc_R", "disc_S"):
        for a, b in zip(getattr(model, name).state_dict().values(),
                        getattr(restored, name).state_dict().values()):
            assert torch.equal(a, b)


def test_full_checkpoint_carries_optimizers_and_scaler(model, tmp_path):
    loaded = ckpt.load(ckpt.save_full(model, tmp_path / "full.pt", epoch=3))
    for key in ("gen_R", "gen_S", "disc_R", "disc_S", "opt_gen", "opt_disc",
                "g_scaler", "d_scaler"):
        assert key in loaded.state, f"a resumable checkpoint needs {key}"
    assert loaded.is_slim is False


def test_resuming_restores_optimizer_state(model, workspace, tmp_path):
    batch = (torch.randn(1, 3, H, W), torch.randn(1, 1, H, W), torch.randn(1, 1, H, W))
    for _ in range(5):
        model.train_step(batch)
    path = ckpt.save_full(model, tmp_path / "full.pt", epoch=1)

    restored = PicganAdapter(workspace=workspace, device="cpu")
    ckpt.restore(restored, path)
    assert (restored.opt_gen.state_dict()["state"].keys()
            == model.opt_gen.state_dict()["state"].keys())
    assert restored.g_scaler.get_scale() == model.g_scaler.get_scale()


# --------------------------------------------------------------------------- #
# Slim checkpoint
# --------------------------------------------------------------------------- #


def test_slim_checkpoint_holds_only_gen_r_and_stats(model, tmp_path):
    loaded = ckpt.load(ckpt.save_slim(model, tmp_path / "slim.pt"))
    assert loaded.is_slim is True
    assert "gen_R" in loaded.state
    for absent in ("gen_S", "disc_R", "disc_S", "opt_gen", "opt_disc"):
        assert absent not in loaded.state
    assert loaded.stats.mode == "computed"


def test_slim_checkpoint_is_smaller(model, tmp_path):
    full = ckpt.save_full(model, tmp_path / "full.pt")
    slim = ckpt.save_slim(model, tmp_path / "slim.pt")
    assert slim.stat().st_size < full.stat().st_size / 2


def test_slim_checkpoint_reproduces_gen_r_exactly(model, workspace, tmp_path):
    """The file that comes back from the cluster must generate identical output."""
    path = ckpt.save_slim(model, tmp_path / "slim.pt")
    sample = torch.randn(1, 3, H, W)
    model.gen_R.eval()
    with torch.no_grad():
        expected = model.gen_R(sample)

    restored = PicganAdapter(workspace=workspace, device="cpu")
    ckpt.restore(restored, path)
    restored.gen_R.eval()
    with torch.no_grad():
        assert torch.equal(restored.gen_R(sample), expected)


# --------------------------------------------------------------------------- #
# Statistics travel inside the checkpoint
# --------------------------------------------------------------------------- #


def test_statistics_live_inside_the_checkpoint(model, tmp_path):
    loaded = ckpt.load(ckpt.save_slim(model, tmp_path / "slim.pt"))
    assert loaded.stats.pair("intensity") == (pytest.approx(0.0511),
                                              pytest.approx(0.0749))
    assert loaded.stats.n_source_frames == 23201


def test_denormalization_uses_the_checkpoints_statistics(model, workspace, tmp_path):
    """Weights trained on cluster statistics must denormalise with those, locally.

    A machine with different data present must not silently substitute its own.
    """
    path = ckpt.save_slim(model, tmp_path / "slim.pt")

    local = PicganAdapter(workspace=workspace, device="cpu")  # defaults to PICGAN's
    assert local.normaweather_modeltion("intensity_real_transform") == (0.0158, 0.0462)
    ckpt.restore(local, path)
    assert local.normaweather_modeltion("intensity_real_transform") == (pytest.approx(0.0511),
                                                               pytest.approx(0.0749))
    assert torch.allclose(local.denormalize_real_intensity(torch.zeros(1, 1, 4, 4)),
                          torch.full((1, 1, 4, 4), 0.0511))


def test_a_checkpoint_without_statistics_is_refused(tmp_path):
    torch.save({"gen_R": {}, "metadata": {}}, tmp_path / "bad.pt")
    with pytest.raises(ckpt.CheckpointError, match="no normalization statistics"):
        ckpt.load(tmp_path / "bad.pt")


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #


def test_save_writes_both_files_and_a_metadata_sidecar(model, tmp_path):
    config = weather_config(tmp_path)
    paths = ckpt.save(model, tmp_path / "run", config, epoch=4)
    assert paths["full"].is_file() and paths["slim"].is_file()

    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["source_dataset"] == "kitti"
    assert metadata["target_dataset"] == "cadc"
    assert metadata["task"] == "weather"
    assert metadata["weather"] == "snow"
    assert metadata["degradation"] == "physics"
    assert metadata["normalization_mode"] == "computed"
    assert metadata["in_channels_s"] == 3
    assert metadata["framework_version"]
    assert metadata["config"]["geometric_degradation"]["precipitation_rate"] == 30.0


def test_metadata_snapshot_reproduces_the_run_config(model, tmp_path):
    config = weather_config(tmp_path)
    loaded = ckpt.load(ckpt.save_full(model, tmp_path / "full.pt", config))
    assert Config.from_dict(loaded.metadata["config"]) == config


def test_two_channel_checkpoint_rebuilds_for_two_channels(workspace, tmp_path):
    torch.manual_seed(3)
    model = PicganAdapter(workspace=workspace, device="cpu", stats=measured_stats())
    model.build_model(2)
    path = ckpt.save_slim(model, tmp_path / "slim2.pt")

    restored = PicganAdapter(workspace=workspace, device="cpu")
    ckpt.restore(restored, path)
    assert restored.in_channels_s == 2


def test_unbuilt_model_cannot_be_saved(workspace, tmp_path):
    model = PicganAdapter(workspace=workspace, device="cpu")
    with pytest.raises(ckpt.CheckpointError, match="build the model"):
        ckpt.save_full(model, tmp_path / "x.pt")


def test_missing_checkpoint_is_reported(tmp_path):
    with pytest.raises(ckpt.CheckpointError, match="not found"):
        ckpt.load(tmp_path / "nope.pt")


# --------------------------------------------------------------------------- #
# ONNX export (optional)
# --------------------------------------------------------------------------- #

onnx = pytest.importorskip("onnx", reason="onnx not installed")


def test_onnx_export_matches_the_torch_generator(model, tmp_path):
    from reality.inference.export_onnx import export_gen_r, verify

    path = export_gen_r(ckpt.save_slim(model, tmp_path / "slim.pt"),
                        tmp_path / "gen_r.onnx", image_shape=(H, W))
    assert path.is_file()
    onnx.checker.check_model(onnx.load(str(path)))

    model.gen_R.eval()
    difference = verify(path, model.gen_R, image_shape=(H, W), in_channels=3)
    assert difference < 1e-4, f"ONNX output diverges from torch by {difference}"


def test_onnx_export_takes_channels_from_the_checkpoint(workspace, tmp_path):
    from reality.inference.export_onnx import export_gen_r

    torch.manual_seed(5)
    two = PicganAdapter(workspace=workspace, device="cpu", stats=measured_stats())
    two.build_model(2)
    path = export_gen_r(ckpt.save_slim(two, tmp_path / "slim2.pt"),
                        tmp_path / "gen_r2.onnx", image_shape=(H, W))
    graph_input = onnx.load(str(path)).graph.input[0]
    assert graph_input.type.tensor_type.shape.dim[1].dim_value == 2


def test_missing_onnx_gives_an_install_message(monkeypatch):
    import builtins

    from reality.inference.export_onnx import OnnxUnavailable, require_onnx

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "onnx":
            raise ImportError("no onnx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(OnnxUnavailable, match="pip install onnx"):
        require_onnx()
