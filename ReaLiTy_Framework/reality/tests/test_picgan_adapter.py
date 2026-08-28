"""PICGAN adapter, including the parity test against the frozen original.

The adapter is the only bridge to ``models/PICGAN/``. If a parity assertion here
fails, the adapter is wrong -- PICGAN is never changed to make it pass.
"""

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reality.core.config import Config
from reality.core.context import Sample, SampleMeta
from reality.core.registry import MODELS
from reality.models import IntensityModel, PicganAdapter, PicganAdapterError
from reality.models.picgan_runtime import PICGAN_DIR, inject_config, load_picgan

H, W = 32, 64  # divisible by 4: the generator downsamples twice


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """A throwaway root, so PICGAN's import-time makedirs never touch the repo."""
    return tmp_path_factory.mktemp("picgan_ws")


@pytest.fixture(scope="module")
def picgan(workspace):
    return load_picgan(workspace)


@pytest.fixture
def adapter(workspace):
    return PicganAdapter(workspace=workspace, device="cpu")


def sim_array(channels=4, seed=0, with_nans=False):
    """A synthetic simulated stack: [range, incidence, reflectance, phy] or no reflectance."""
    rng = np.random.default_rng(seed)
    arr = rng.random((channels, H, W), dtype=np.float32)
    if with_nans:
        arr[0, 0, 0] = np.nan
        arr[1, 1, 1] = np.inf
        arr[-1, 2, 2] = -np.inf
    return arr


def real_array(seed=1):
    """A synthetic real stack: [range, intensity]."""
    return np.random.default_rng(seed).random((2, H, W), dtype=np.float32)


def make_source_sample(arr, has_reflectance=True):
    """Wrap a sim stack as a projected Sample: range image + phy set upstream."""
    channels = ("range", "incidence", "reflectance") if has_reflectance else ("range", "incidence")
    return Sample(
        points=np.zeros((4, 4), dtype=np.float32),
        meta=SampleMeta(dataset="kitti", task="weather", has_reflectance=has_reflectance),
        range_image=arr[:-1].copy(), phy=arr[-1:].copy(), channels=channels,
    )


def make_target_sample(arr):
    return Sample(
        points=np.zeros((4, 4), dtype=np.float32),
        meta=SampleMeta(dataset="cadc", task="weather", has_reflectance=False),
        range_image=arr.copy(), channels=("range", "intensity"),
    )


# --------------------------------------------------------------------------- #
# THE PARITY TEST
# --------------------------------------------------------------------------- #


def original_tuple(picgan, tmp_path, sim, real):
    """What PICGAN's own dataset.py produces from the equivalent .npy files.

    LidarDataset is the frozen loader, driven with PICGAN's own transform objects
    exactly as main.py wires them.
    """
    sim_dir, real_dir = tmp_path / "sim", tmp_path / "real"
    sim_dir.mkdir(); real_dir.mkdir()
    np.save(sim_dir / "000000.npy", sim)
    np.save(real_dir / "000000.npy", real)
    tu = picgan.transform_utils
    dataset = picgan.LidarDataset(
        lidar_real_dir=str(real_dir), lidar_sim_adverse_dir=str(sim_dir),
        lidar_transform=tu.lidar_transform, incidence_transform=tu.incidence_transform,
        reflectance_transform=tu.reflectance_transform,
        intensity_sim_transform=tu.intensity_sim_transform,
        intensity_real_transform=tu.intensity_real_transform,
    )
    return dataset[0]


@pytest.mark.parametrize("with_nans", [False, True], ids=["clean", "with-nan-and-inf"])
def test_parity_tensors_are_identical_to_the_original(adapter, picgan, tmp_path, with_nans):
    """The adapter must produce byte-identical (sim, real, phy) to dataset.py."""
    sim, real = sim_array(4, with_nans=with_nans), real_array()

    mine = adapter.to_tensors(sim, real)
    theirs = original_tuple(picgan, tmp_path, sim, real)

    for name, a, b in zip(("sim", "real", "phy"), mine, theirs):
        assert a.shape == b.shape, f"{name}: shape {a.shape} != {b.shape}"
        assert a.dtype == b.dtype, f"{name}: dtype {a.dtype} != {b.dtype}"
        assert torch.equal(a, b), f"{name}: values differ from PICGAN's dataset.py"


def test_parity_from_a_sample_not_just_arrays(adapter, picgan, tmp_path):
    """The whole Sample -> tensor path matches, not only the array entry point."""
    sim, real = sim_array(4, seed=7), real_array(seed=8)
    mine = adapter.sample_to_tensors(make_source_sample(sim), make_target_sample(real))
    theirs = original_tuple(picgan, tmp_path, sim, real)
    for a, b in zip(mine, theirs):
        assert torch.equal(a, b)


def test_parity_channel_order_and_shapes(adapter, picgan, tmp_path):
    """Channel order is range, incidence, reflectance -- verified plane by plane."""
    sim, real = sim_array(4, seed=3), real_array(seed=4)
    mine_sim, mine_real, mine_phy = adapter.to_tensors(sim, real)
    assert mine_sim.shape == (3, H, W)
    assert mine_real.shape == (1, H, W) and mine_phy.shape == (1, H, W)

    tu = picgan.transform_utils
    assert torch.equal(mine_sim[0:1], tu.lidar_transform(sim[0]))
    assert torch.equal(mine_sim[1:2], tu.incidence_transform(sim[1]))
    assert torch.equal(mine_sim[2:3], tu.reflectance_transform(sim[2]))
    # phy is the sim stack's last channel under intensity_sim_transform...
    assert torch.equal(mine_phy, tu.intensity_sim_transform(sim[3]))
    # ...and real intensity is the real stack's channel 1, not channel 0.
    assert torch.equal(mine_real, tu.intensity_real_transform(real[1]))
    assert not torch.equal(mine_real, tu.intensity_real_transform(real[0]))


def test_parity_uses_picgans_own_normaweather_modeltion_constants(adapter):
    """Constants come from transform_utils.py; they are not restated in ReaLiTy."""
    assert adapter.normaweather_modeltion("lidar_transform") == (pytest.approx(0.0965),
                                                        pytest.approx(0.1068))
    assert adapter.normaweather_modeltion("incidence_transform") == (pytest.approx(0.7156),
                                                            pytest.approx(0.6352))
    assert adapter.normaweather_modeltion("reflectance_transform") == (pytest.approx(0.2979),
                                                              pytest.approx(0.2743))
    assert adapter.normaweather_modeltion("intensity_sim_transform") == (pytest.approx(0.1745),
                                                                pytest.approx(0.1515))
    assert adapter.normaweather_modeltion("intensity_real_transform") == (pytest.approx(0.0158),
                                                                 pytest.approx(0.0462))
    source = Path(PICGAN_DIR / "transform_utils.py").read_text()
    for value in ("0.0965", "0.1068", "0.7156", "0.2979", "0.1745", "0.0158"):
        assert value in source, "constants must still live in PICGAN, not be copied out"


def test_parity_forward_pass_matches(adapter, picgan, tmp_path):
    """gen_R driven through the adapter equals gen_R driven the original way."""
    sim, real = sim_array(4, seed=11), real_array(seed=12)

    torch.manual_seed(1234)
    adapter.build_model(in_channels_s=3, in_channels_r=1)

    mine_sim = adapter.to_tensors(sim, real)[0].unsqueeze(0)
    theirs_sim = original_tuple(picgan, tmp_path, sim, real)[0].unsqueeze(0)

    adapter.gen_R.eval()
    with torch.no_grad():
        mine_out = adapter.gen_R(mine_sim)
        theirs_out = adapter.gen_R(theirs_sim)
    assert torch.equal(mine_out, theirs_out)

    # And an independently constructed generator with the same seed agrees.
    torch.manual_seed(1234)
    reference = picgan.Generator(img_channels=3, out_channels=1, num_residuals=9).eval()
    with torch.no_grad():
        assert torch.allclose(reference(theirs_sim), theirs_out, atol=0, rtol=0)


# --------------------------------------------------------------------------- #
# Channel-count paths
# --------------------------------------------------------------------------- #


def test_builds_with_reflectance_3_channel_source(adapter, picgan):
    adapter.build_model(in_channels_s=3)
    assert picgan.config.IN_CHANNELS_S == 3 and picgan.config.IN_CHANNELS_R == 1
    out = adapter.gen_R(torch.randn(1, 3, H, W))
    assert out.shape == (1, 1, H, W)
    assert adapter.gen_S(torch.randn(1, 1, H, W)).shape == (1, 3, H, W)
    assert adapter.disc_S(torch.randn(1, 3, H, W)).shape[1] == 1


def test_builds_without_reflectance_2_channel_source(adapter, picgan):
    """The no-labels path needs no PICGAN edit: channel counts are constructor args."""
    adapter.build_model(in_channels_s=2)
    assert picgan.config.IN_CHANNELS_S == 2
    assert adapter.gen_R(torch.randn(1, 2, H, W)).shape == (1, 1, H, W)
    assert adapter.gen_S(torch.randn(1, 1, H, W)).shape == (1, 2, H, W)
    assert adapter.disc_S(torch.randn(1, 2, H, W)).shape[1] == 1


def test_2_channel_stack_from_a_sample_without_labels(adapter):
    sim = sim_array(3, seed=5)  # [range, incidence, phy]
    sample = make_source_sample(sim, has_reflectance=False)
    stack = adapter.source_stack(sample)
    assert stack.shape == (3, H, W)
    assert adapter.channels_for(sample) == 2

    sim_t, _, phy_t = adapter.to_tensors(stack, real_array())
    assert sim_t.shape == (2, H, W), "reflectance must be dropped from the source stack"
    assert phy_t.shape == (1, H, W), "phy is still the last channel"
    # phy is not silently consumed as reflectance.
    tu = adapter.transforms
    assert torch.equal(phy_t, tu.intensity_sim_transform(sim[2]))


def test_2_channel_batch_flows_through_the_frozen_train_step(adapter):
    """train_fn never indexes channels, so a 2-channel source passes through."""
    adapter.build_model(in_channels_s=2)
    source = make_source_sample(sim_array(3, seed=9), has_reflectance=False)
    target = make_target_sample(real_array(seed=10))
    batch = adapter.build_batch([source], [target])
    assert batch[0].shape == (1, 2, H, W)
    stats = adapter.train_step(batch)
    assert stats["source_channels"] == 2


def test_build_for_picks_the_channel_count_from_the_sample(adapter):
    adapter.build_for(make_source_sample(sim_array(3), has_reflectance=False))
    assert adapter.in_channels_s == 2


def test_rejects_unsupported_channel_count(adapter):
    """2, 3 and 4 are the supported widths (4 adds the retro channel)."""
    with pytest.raises(PicganAdapterError, match="in_channels_s must be"):
        adapter.build_model(in_channels_s=5)
    adapter.build_model(in_channels_s=4)
    assert adapter.gen_R(torch.randn(1, 4, H, W)).shape == (1, 1, H, W)


# --------------------------------------------------------------------------- #
# Training delegation and inference
# --------------------------------------------------------------------------- #


def test_train_step_delegates_and_updates_weights(adapter):
    adapter.build_model(in_channels_s=3)
    before = adapter.gen_R.last.weight.detach().clone()
    batch = adapter.build_batch([make_source_sample(sim_array(4))],
                                [make_target_sample(real_array())])
    adapter.train_step(batch)
    assert not torch.equal(before, adapter.gen_R.last.weight), "the step must optimise"


def test_train_step_requires_a_built_model(adapter):
    with pytest.raises(RuntimeError, match="build_model"):
        adapter.train_step((torch.zeros(1, 3, H, W),) * 3)


def test_generate_returns_single_channel_intensity(adapter):
    adapter.build_model(in_channels_s=3)
    out = adapter.generate(make_source_sample(sim_array(4, seed=21)))
    assert out.shape == (1, 1, H, W)
    assert torch.isfinite(out).all()
    assert out.min() >= -1.0 and out.max() <= 1.0, "gen_R ends in tanh"


def test_denormalize_uses_the_real_intensity_constants(adapter):
    mean, std = adapter.normaweather_modeltion("intensity_real_transform")
    normalized = torch.zeros(1, 1, 4, 4)
    assert torch.allclose(adapter.denormalize_real_intensity(normalized),
                          torch.full((1, 1, 4, 4), mean))
    assert torch.allclose(adapter.denormalize_real_intensity(torch.ones(1, 1, 4, 4)),
                          torch.full((1, 1, 4, 4), mean + std))


# --------------------------------------------------------------------------- #
# Sample contract enforcement
# --------------------------------------------------------------------------- #


def test_missing_phy_is_refused_not_invented(adapter):
    """PICGAN never computes phy; a Sample without it is an upstream error."""
    sample = make_source_sample(sim_array(4))
    sample.phy = None
    with pytest.raises(PicganAdapterError, match="PICGAN never computes"):
        adapter.source_stack(sample)


def test_unprojected_sample_is_refused(adapter):
    sample = make_source_sample(sim_array(4))
    sample.range_image = None
    with pytest.raises(PicganAdapterError, match="project the sample first"):
        adapter.source_stack(sample)


def test_mismatched_batch_lengths(adapter):
    with pytest.raises(PicganAdapterError, match="paired batch"):
        adapter.build_batch([make_source_sample(sim_array(4))], [])


def test_real_stack_must_have_two_channels(adapter):
    with pytest.raises(PicganAdapterError, match=r"must be \(2, H, W\)"):
        adapter.to_tensors(sim_array(4), np.zeros((1, H, W), dtype=np.float32))


# --------------------------------------------------------------------------- #
# Weights, registry, interface
# --------------------------------------------------------------------------- #


def test_weights_round_trip(adapter, workspace, tmp_path):
    adapter.build_model(in_channels_s=3)
    path = adapter.save_weights(tmp_path / "model.pt")
    other = PicganAdapter(workspace=workspace, device="cpu")
    other.load_weights(path)
    assert other.in_channels_s == 3
    for a, b in zip(adapter.gen_R.state_dict().values(), other.gen_R.state_dict().values()):
        assert torch.equal(a, b)


def test_loading_a_2_channel_checkpoint_rebuilds_for_2_channels(adapter, workspace, tmp_path):
    adapter.build_model(in_channels_s=2)
    path = adapter.save_weights(tmp_path / "m2.pt")
    other = PicganAdapter(workspace=workspace, device="cpu")
    other.load_weights(path)
    assert other.in_channels_s == 2


def test_channel_mismatch_on_load_is_reported(adapter, workspace, tmp_path):
    adapter.build_model(in_channels_s=2)
    path = adapter.save_weights(tmp_path / "m2.pt")
    other = PicganAdapter(workspace=workspace, device="cpu")
    other.build_model(in_channels_s=3)
    with pytest.raises(PicganAdapterError, match="in_channels_s"):
        other.load_weights(path)


def test_registered_as_a_swappable_model():
    assert MODELS.get("picgan") is PicganAdapter
    assert issubclass(PicganAdapter, IntensityModel)


def test_pipeline_can_resolve_picgan_from_config(sensor_cfg_dict, workspace):
    """The model is selected by name from config, not baked into the pipeline."""
    from reality.core.pipeline import plan_stages
    cfg = Config.from_dict(sensor_cfg_dict)
    model_stage = [s for s in plan_stages(cfg) if s.role == "model"][0]
    assert MODELS.get(model_stage.name) is PicganAdapter


def test_training_config_is_injected_from_reality(workspace, weather_cfg_dict):
    weather_cfg_dict["training"]["lambda_physics"] = 7.5
    weather_cfg_dict["training"]["learning_rate"] = 3.0e-5
    cfg = Config.from_dict(weather_cfg_dict)
    model = PicganAdapter(cfg, workspace=workspace, device="cpu")
    assert model.picgan.config.LAMBDA_Physics == 7.5
    assert model.picgan.config.LEARNING_RATE == pytest.approx(3.0e-5)
    assert model.picgan.config.LAMBDA_CYCLE == 10


def test_device_selection_refuses_an_unusable_cuda(monkeypatch):
    """CUDA that reports available but has no kernels for the GPU must not be chosen."""
    from reality.models import picgan_runtime

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (7, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_80", "sm_90"])
    with pytest.warns(RuntimeWarning, match="no kernels for compute capability 7.0"):
        assert picgan_runtime.select_device() == "cpu"

    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_70", "sm_80"])
    assert picgan_runtime.select_device() == "cuda"
    assert picgan_runtime.select_device("cpu") == "cpu"


def test_no_stray_directories_are_created_at_import(workspace, tmp_path, monkeypatch):
    """PICGAN's config.py makedirs must land under the workspace, not the CWD."""
    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    model = PicganAdapter(workspace=workspace / "run", device="cpu")

    assert list(elsewhere.iterdir()) == [], "importing PICGAN must not litter the CWD"
    assert Path(os.getcwd()) == elsewhere, "the CWD must be restored"
    assert Path(model.picgan.config.CHECKPOINT_GEN_R).is_absolute()
    assert Path(model.picgan.config.Trial_Path).is_absolute()
    assert str(workspace) in str(model.picgan.config.CHECKPOINT_GEN_R)


def test_picgan_paths_are_cwd_independent(workspace, tmp_path, monkeypatch):
    """Nothing may depend on where the process happens to be running from."""
    model = PicganAdapter(workspace=workspace / "run2", device="cpu")
    before = dict(gen_r=model.picgan.config.CHECKPOINT_GEN_R,
                  out=str(model.picgan.config.OUTPUT_FOLDER))
    monkeypatch.chdir(tmp_path)
    again = PicganAdapter(workspace=workspace / "run2", device="cpu")
    assert again.picgan.config.CHECKPOINT_GEN_R == before["gen_r"]
    assert str(again.picgan.config.OUTPUT_FOLDER) == before["out"]
    for path in (again.picgan.config.Trial_Path, again.picgan.config.OUTPUT_FOLDER):
        assert Path(path).is_dir(), "injected output directories must exist"


# --------------------------------------------------------------------------- #
# : data-derived normalization
# --------------------------------------------------------------------------- #


def test_adapter_defaults_to_the_published_constants(adapter):
    """Parity above depends on this: the default path is PICGAN's own numbers."""
    from reality.preprocessing.statistics import PICGAN_DEFAULT_STATS

    assert adapter.stats.mode == "picgan_default"
    assert adapter.normaweather_modeltion("incidence_transform") == PICGAN_DEFAULT_STATS["incidence"]


def test_computed_statistics_replace_the_constants(workspace):
    """Measured statistics drive the transforms instead of the VoxelScape literals."""
    from reality.preprocessing.statistics import ChannelStats, NormalizationStats

    stats = NormalizationStats(
        channels={"incidence": ChannelStats(mean=0.3616, std=0.2379),
                  "intensity": ChannelStats(mean=0.0511, std=0.0749)},
        mode="computed", source_dataset="kitti", target_dataset="cadc",
    )
    model = PicganAdapter(workspace=workspace, device="cpu", stats=stats)
    assert model.normaweather_modeltion("incidence_transform") == (pytest.approx(0.3616),
                                                          pytest.approx(0.2379))
    # An unmeasured channel still falls back to PICGAN's constant.
    assert model.normaweather_modeltion("reflectance_transform") == (pytest.approx(0.2979),
                                                            pytest.approx(0.2743))


def test_computed_statistics_centre_the_data_that_default_constants_do_not(workspace):
    """The point of the correction: real data lands near zero mean after z-scoring."""
    from reality.preprocessing.statistics import ChannelStats, NormalizationStats

    rng = np.random.default_rng(0)
    incidence = rng.normal(0.36, 0.24, (H, W)).astype(np.float32)
    sim = np.stack([np.zeros((H, W), np.float32), incidence,
                    np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)])
    real = np.zeros((2, H, W), dtype=np.float32)

    default = PicganAdapter(workspace=workspace, device="cpu")
    computed = PicganAdapter(workspace=workspace, device="cpu", stats=NormalizationStats(
        channels={"incidence": ChannelStats(mean=0.36, std=0.24)}, mode="computed"))

    default_mean = float(default.to_tensors(sim, real)[0][1].mean())
    computed_mean = float(computed.to_tensors(sim, real)[0][1].mean())
    assert abs(default_mean) > 0.4, "VoxelScape constants leave KITTI incidence off-centre"
    assert abs(computed_mean) < 0.05, "measured constants centre it"


def test_denormalization_uses_the_targets_intensity_statistics(workspace):
    """The asymmetry fix: outputs return to units via the target's own statistics."""
    from reality.preprocessing.statistics import ChannelStats, NormalizationStats

    stats = NormalizationStats(
        channels={"intensity": ChannelStats(mean=0.0511, std=0.0749)},
        mode="computed", target_dataset="cadc")
    model = PicganAdapter(workspace=workspace, device="cpu", stats=stats)

    assert torch.allclose(model.denormalize_real_intensity(torch.zeros(1, 1, 4, 4)),
                          torch.full((1, 1, 4, 4), 0.0511))
    assert torch.allclose(model.denormalize_real_intensity(torch.ones(1, 1, 4, 4)),
                          torch.full((1, 1, 4, 4), 0.0511 + 0.0749))
    # Not the hard-coded CADC constant it used to be.
    default = PicganAdapter(workspace=workspace, device="cpu")
    assert not torch.allclose(model.denormalize_real_intensity(torch.zeros(1, 1, 4, 4)),
                              default.denormalize_real_intensity(torch.zeros(1, 1, 4, 4)))


def test_normalization_is_a_round_trip(workspace):
    from reality.preprocessing.statistics import ChannelStats, NormalizationStats

    stats = NormalizationStats(channels={"intensity": ChannelStats(mean=0.05, std=0.07)})
    model = PicganAdapter(workspace=workspace, device="cpu", stats=stats)
    values = np.linspace(0, 1, 32, dtype=np.float32).reshape(1, 4, 8)
    real = np.concatenate([np.zeros_like(values), values])
    normalized = model.to_tensors(np.zeros((4, 4, 8), np.float32), real)[1]
    assert torch.allclose(model.denormalize_real_intensity(normalized),
                          torch.from_numpy(values), atol=1e-5)


def test_switching_statistics_rebuilds_the_transforms(adapter):
    from reality.preprocessing.statistics import ChannelStats, NormalizationStats

    before = adapter.normaweather_modeltion("incidence_transform")
    adapter.use_statistics(NormalizationStats(
        channels={"incidence": ChannelStats(mean=0.1, std=0.2)}, mode="computed"))
    assert adapter.normaweather_modeltion("incidence_transform") == (pytest.approx(0.1),
                                                            pytest.approx(0.2))
    assert before != adapter.normaweather_modeltion("incidence_transform")


def test_importing_picgan_creates_no_scaffolding(tmp_path, monkeypatch):
    """config.py no longer runs makedirs at import ( plumbing fix)."""
    from reality.models.picgan_runtime import PICGAN_DIR

    source = (PICGAN_DIR / "config.py").read_text()
    assert "def ensure_output_dirs" in source
    body = source.split("def ensure_output_dirs")[0]
    assert "os.makedirs" not in body, "no directory creation may run at import time"

    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    PicganAdapter(workspace=workspace, device="cpu")
    assert not (workspace / "Trial").exists(), "importing PICGAN must create no Trial/"
    assert not (Path.cwd() / "Trial").exists()


def test_run_directory_holds_only_realitys_own_structure(tmp_path):
    """A run directory should contain checkpoints and logs, not legacy scaffolding."""
    run_dir = tmp_path / "run"
    model = PicganAdapter(workspace=run_dir, device="cpu")
    assert not (run_dir / "Trial").exists()
    # The injected paths are still absolute and usable.
    assert Path(model.picgan.config.Trial_Path).is_absolute()
    assert Path(model.picgan.config.OUTPUT_FOLDER).is_dir()


def test_standalone_picgan_can_still_create_its_directories(tmp_path, monkeypatch):
    """main.py runs PICGAN on its own, so the helper must still work."""
    from reality.models.picgan_runtime import load_picgan

    picgan = load_picgan(tmp_path / "ws")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(picgan.config, "Trial_Path", tmp_path / "Trial/Output/M")
    monkeypatch.setattr(picgan.config, "OUTPUT_FOLDER", tmp_path / "Trial/Output/O")
    picgan.config.ensure_output_dirs()
    assert (tmp_path / "Trial/Output/M").is_dir()
    assert (tmp_path / "Trial/Output/O").is_dir()


# --------------------------------------------------------------------------- #
# : sigmoid head, physics option (iii), distributional term
# --------------------------------------------------------------------------- #


def sigmoid_config(tmp_path, **model):
    return Config.from_dict({
        "source": {"dataset": "kitti", "path": "/tmp/k"},
        "target": {"dataset": "cadc", "path": "/tmp/c"},
        "task": {"type": "sensor"},
        "normalization": {"source": "picgan_default"},
        "model": {"type": "picgan", **model},
        "output": {"checkpoint_dir": str(tmp_path / "run")},
    })


def test_tanh_remains_the_default(adapter):
    """the released model must stay reproducible: nothing changes unless asked for."""
    adapter.build_model(3)
    assert adapter.output_activation == "tanh"
    assert adapter.gen_R.output_activation == "tanh"
    assert adapter.physics_transform is None, "no physics rescaling with a tanh head"
    assert adapter.distributional_loss is None, "the extra term is off by default"


def test_sigmoid_head_applies_to_gen_r_only(workspace, tmp_path):
    model = PicganAdapter(sigmoid_config(tmp_path, output_activation="sigmoid"),
                          workspace=workspace, device="cpu")
    model.build_model(3)
    assert model.gen_R.output_activation == "sigmoid"
    assert model.gen_S.output_activation == "tanh", (
        "gen_S emits z-scored source channels that are legitimately negative"
    )


def test_sigmoid_output_is_bounded_to_the_unit_interval(workspace, tmp_path):
    model = PicganAdapter(sigmoid_config(tmp_path, output_activation="sigmoid"),
                          workspace=workspace, device="cpu")
    model.build_model(3)
    out = model.gen_R(torch.randn(2, 3, H, W) * 5)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out.max() > 0.5, "the head must be able to reach the bright end"


def test_tanh_head_cannot_exceed_one_sigma(adapter):
    """the ceiling, pinned as a regression."""
    adapter.build_model(3)
    mean, std = adapter.normaweather_modeltion("intensity_real_transform")
    out = adapter.gen_R(torch.randn(2, 3, H, W) * 5)
    denormalised = adapter.denormalize_real_intensity(out)
    assert denormalised.max() <= mean + std + 1e-6
    assert denormalised.min() >= mean - std - 1e-6


def test_sigmoid_target_transform_is_the_identity(workspace, tmp_path):
    """With a sigmoid head the target is used in native [0, 1]."""
    model = PicganAdapter(sigmoid_config(tmp_path, output_activation="sigmoid"),
                          workspace=workspace, device="cpu")
    assert model.normaweather_modeltion("intensity_real_transform") == (0.0, 1.0)
    values = np.linspace(0, 1, 32, dtype=np.float32).reshape(1, 4, 8)
    real = np.concatenate([np.zeros_like(values), values])
    _, normalised, _ = model.to_tensors(np.zeros((4, 4, 8), np.float32), real)
    assert torch.allclose(normalised, torch.from_numpy(values), atol=1e-6)
    # ... and denormaweather_modeltion is then also the identity.
    assert torch.allclose(model.denormalize_real_intensity(normalised),
                          torch.from_numpy(values), atol=1e-6)


def test_physics_transform_preserves_the_comparison_space(workspace, tmp_path):
    """Option (iii): the physics term keeps comparing z-scores to z-scores."""
    from reality.preprocessing.statistics import ChannelStats, NormalizationStats

    stats = NormalizationStats(
        channels={"intensity": ChannelStats(mean=0.0494, std=0.0755)}, mode="computed")
    model = PicganAdapter(sigmoid_config(tmp_path, output_activation="sigmoid"),
                          workspace=workspace, device="cpu", stats=stats)
    transform = model.physics_transform
    assert transform is not None

    # A data-unit intensity maps to the z-score the tanh model would have emitted.
    data_units = torch.tensor([0.0494, 0.0494 + 0.0755, 1.0])
    z = transform(data_units)
    assert torch.allclose(z[:2], torch.tensor([0.0, 1.0]), atol=1e-4)
    assert float(z[2]) > 12.0, "the bright tail is now reachable in physics space"


def test_distributional_term_is_off_unless_weighted(workspace, tmp_path):
    off = PicganAdapter(sigmoid_config(tmp_path, output_activation="sigmoid"),
                        workspace=workspace, device="cpu")
    assert off.distributional_loss is None
    on = PicganAdapter(sigmoid_config(tmp_path, output_activation="sigmoid",
                                      lambda_wasserstein=10.0),
                       workspace=workspace, device="cpu")
    assert on.distributional_loss is not None


def test_distributional_term_scores_occupied_pixels(workspace, tmp_path):
    """It must compare like with like: occupied pixels on both sides."""
    model = PicganAdapter(sigmoid_config(tmp_path, output_activation="sigmoid",
                                         lambda_wasserstein=1.0),
                          workspace=workspace, device="cpu")
    loss = model.distributional_loss
    rng = np.random.default_rng(0)
    occupied = torch.from_numpy((rng.random((1, 1, H, W)) < 0.4).astype(np.float32))
    range_mean, range_std = model.normaweather_modeltion("lidar_transform")
    sim = torch.full((1, 3, H, W), (0.0 - range_mean) / range_std)
    sim[:, :1] = torch.where(occupied > 0, torch.tensor(5.0), sim[:, :1])

    real = occupied * 0.05
    matched = loss(occupied * 0.05, real, sim, real)
    mismatched = loss(occupied * 0.9, real, sim, real)
    assert float(matched) < float(mismatched), "a matching distribution must score lower"


def test_checkpoint_records_the_head(workspace, tmp_path):
    model = PicganAdapter(sigmoid_config(tmp_path, output_activation="sigmoid"),
                          workspace=workspace, device="cpu")
    model.build_model(3)
    assert model.state_dict()["output_activation"] == "sigmoid"


def test_config_rejects_an_unknown_activation(tmp_path):
    from reality.core.config import ConfigError

    with pytest.raises(ConfigError, match="output_activation"):
        sigmoid_config(tmp_path, output_activation="relu")
