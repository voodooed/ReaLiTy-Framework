"""GPU execution: parity, round-trip and determinism on CUDA.

Skipped unless a CUDA device is actually usable -- ``cuda.is_available()`` alone
is not enough, since a torch build without kernels for the installed GPU reports
True and then fails on the first op.
"""

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reality.core.config import DataSpec
from reality.models import PicganAdapter
from reality.postprocessing.backprojection import backproject
from reality.preprocessing.projection import project


def cuda_really_works() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.mm(torch.randn(8, 8, device="cuda"), torch.randn(8, 8, device="cuda"))
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


gpu_only = pytest.mark.skipif(not cuda_really_works(),
                              reason="no usable CUDA device (kernels or driver)")
DATA_ROOT = Path(os.environ.get("REALITY_DATA_ROOT", "data"))
kitti_only = pytest.mark.skipif(not (DATA_ROOT / "KITTI").is_dir(),
                                reason="KITTI data not present")

H, W = 32, 64


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    return tmp_path_factory.mktemp("gpu_ws")


def sim_array(channels=4, seed=0):
    return np.random.default_rng(seed).random((channels, H, W), dtype=np.float32)


def real_array(seed=1):
    return np.random.default_rng(seed).random((2, H, W), dtype=np.float32)


@gpu_only
def test_the_selected_device_is_cuda(workspace):
    """With the cu126 build the adapter must choose the GPU, not fall back."""
    model = PicganAdapter(workspace=workspace)
    assert model.device.type == "cuda"
    assert "sm_70" in torch.cuda.get_arch_list(), "the build must cover the V100s"


@gpu_only
def test_gpu_and_cpu_tensors_are_identical(workspace):
    """The Sample -> tensor path is device-independent; only the model moves."""
    cpu = PicganAdapter(workspace=workspace, device="cpu")
    gpu = PicganAdapter(workspace=workspace, device="cuda")
    sim, real = sim_array(), real_array()
    for a, b in zip(cpu.to_tensors(sim, real), gpu.to_tensors(sim, real)):
        assert torch.equal(a, b)


@gpu_only
def test_parity_against_picgans_dataset_holds_on_gpu(workspace, tmp_path):
    """the parity guarantee, with the batch built on CUDA."""
    from reality.models.picgan_runtime import load_picgan

    picgan = load_picgan(workspace)
    sim, real = sim_array(seed=5), real_array(seed=6)

    sim_dir, real_dir = tmp_path / "sim", tmp_path / "real"
    sim_dir.mkdir(); real_dir.mkdir()
    np.save(sim_dir / "000000.npy", sim)
    np.save(real_dir / "000000.npy", real)
    tu = picgan.transform_utils
    theirs = picgan.LidarDataset(
        lidar_real_dir=str(real_dir), lidar_sim_adverse_dir=str(sim_dir),
        lidar_transform=tu.lidar_transform, incidence_transform=tu.incidence_transform,
        reflectance_transform=tu.reflectance_transform,
        intensity_sim_transform=tu.intensity_sim_transform,
        intensity_real_transform=tu.intensity_real_transform,
    )[0]

    model = PicganAdapter(workspace=workspace, device="cuda")
    mine = model.to_tensors(sim, real)
    for a, b in zip(mine, theirs):
        assert torch.equal(a.cpu(), b), "parity must not depend on the device"


@gpu_only
def test_seeded_forward_pass_matches_between_cpu_and_gpu(workspace):
    """Same seed, same weights: gen_R agrees across devices to float tolerance."""
    sim = torch.from_numpy(sim_array(seed=7)[:3]).unsqueeze(0)

    torch.manual_seed(4242)
    cpu = PicganAdapter(workspace=workspace, device="cpu")
    cpu.build_model(3)
    torch.manual_seed(4242)
    gpu = PicganAdapter(workspace=workspace, device="cuda")
    gpu.build_model(3)

    for a, b in zip(cpu.gen_R.state_dict().values(), gpu.gen_R.state_dict().values()):
        assert torch.equal(a, b.cpu()), "seeded initiaweather_modeltion must match"

    cpu.gen_R.eval(); gpu.gen_R.eval()
    with torch.no_grad():
        assert torch.allclose(cpu.gen_R(sim), gpu.gen_R(sim.cuda()).cpu(), atol=1e-5)


@gpu_only
def test_repeated_gpu_forward_passes_agree_to_tolerance(workspace):
    """Without deterministic kernels, cuDNN's algorithm choice varies the result.

    Measured on a V100 this is ~3e-7 -- reproducible in the numerical sense, but
    not bit-identical, which matters when comparing runs exactly.
    """
    from reality.core.determinism import seed_everything

    seed_everything(11, deterministic=False)
    model = PicganAdapter(workspace=workspace, device="cuda")
    model.build_model(3)
    sim = torch.from_numpy(sim_array(seed=8)[:3]).unsqueeze(0).cuda()
    model.gen_R.eval()
    with torch.no_grad():
        first, second = model.gen_R(sim), model.gen_R(sim)
    assert torch.allclose(first, second, atol=1e-5)


@gpu_only
def test_deterministic_mode_makes_gpu_passes_bit_identical(workspace):
    """seed_everything(deterministic=True) buys exact reproducibility on CUDA."""
    from reality.core.determinism import seed_everything

    seed_everything(11, deterministic=True)
    model = PicganAdapter(workspace=workspace, device="cuda")
    model.build_model(3)
    sim = torch.from_numpy(sim_array(seed=8)[:3]).unsqueeze(0).cuda()
    model.gen_R.eval()
    with torch.no_grad():
        assert torch.equal(model.gen_R(sim), model.gen_R(sim))


@gpu_only
def test_seeded_builds_match_across_runs(workspace):
    from reality.core.determinism import seed_everything

    weights = []
    for _ in range(2):
        seed_everything(2024, deterministic=True)
        model = PicganAdapter(workspace=workspace, device="cuda")
        model.build_model(3)
        weights.append(model.gen_R.last.weight.detach().clone())
    assert torch.equal(weights[0], weights[1])


@gpu_only
def test_training_step_runs_on_gpu_with_amp(workspace):
    """AMP is enabled on CUDA; the frozen train_fn must still step cleanly."""
    from reality.core.context import Sample, SampleMeta

    model = PicganAdapter(workspace=workspace, device="cuda")
    model.build_model(3)
    assert model.g_scaler.is_enabled(), "AMP should be on for CUDA"

    sim = torch.from_numpy(sim_array(seed=9)).unsqueeze(0)
    real = torch.from_numpy(real_array(seed=10)).unsqueeze(0)
    batch = (sim[:, :3].cuda(), real[:, 1:2].cuda(), sim[:, 3:4].cuda())
    before = model.gen_R.last.weight.detach().clone()

    # GradScaler starts at a high loss scale and skips the optimiser step while it
    # calibrates down: on this box steps 1-3 are skipped and the first real update
    # lands on step 4. That is standard AMP warm-up, not a stalled optimiser.
    updated_at = None
    for step in range(1, 7):
        model.train_step(batch)
        if updated_at is None and not torch.equal(before, model.gen_R.last.weight):
            updated_at = step
    assert updated_at is not None, "AMP training must start updating within a few steps"
    assert updated_at <= 5
    assert model.g_scaler.get_scale() < 65536, "the scaler must have calibrated down"
    assert torch.isfinite(model.gen_R.last.weight).all()


@gpu_only
@kitti_only
def test_real_frame_round_trips_through_the_gpu(workspace):
    """A real KITTI frame: project, generate on CUDA, back-project, geometry intact."""
    from reality.datasets import KittiAdapter

    adapter = KittiAdapter(DataSpec(dataset="kitti", path=str(DATA_ROOT / "KITTI")),
                           sequences=["00"])
    sample = adapter[0]
    projected = project(sample)
    projected.phy = np.zeros((1,) + projected.mapping.shape, dtype=np.float32)

    model = PicganAdapter(workspace=workspace, device="cuda")
    model.build_for(projected)
    generated = model.denormalize_real_intensity(model.generate(projected))
    intensity = generated.detach().cpu().numpy()[0]

    assert generated.device.type == "cuda"
    result = backproject(projected, intensity)
    assert np.array_equal(result.points[:, :3], sample.points[:, :3])
    assert result.n_written == int((projected.mapping >= 0).sum())
    assert np.isfinite(result.points[:, 3]).all()
