"""End-to-end sensor pipeline: projection -> PICGAN -> back-projection -> file."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reality.core.config import Config
from reality.core.context import Sample, SampleMeta
from reality.core.registry import DATASETS, MODELS, TRANSFORMS
from reality.inference import GenerationError, IntensityGenerator
from reality.io import OutputWriter
from reality.models import PicganAdapter
from reality.plugins import register_all
from reality.postprocessing.backprojection import backproject
from reality.preprocessing.projection import project
from reality.tests.test_projection import CADC_SENSOR, KITTI_SENSOR, make_cloud

register_all()

SMALL = dict(proj_H=32, proj_W=64, fov_up=3.0, fov_down=-25.0)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    return tmp_path_factory.mktemp("gen_ws")


@pytest.fixture
def model(workspace):
    return PicganAdapter(workspace=workspace, device="cpu")


def small_sensor():
    from reality.core.config import SensorSpec
    return SensorSpec(**SMALL)


def source_sample(n=1500, seed=0, has_reflectance=True, with_phy=True):
    """A projected source Sample with phy set, as the sensor path requires."""
    columns = ["x", "y", "z", "intensity"]
    if has_reflectance:
        columns.append("reflectance")
    if with_phy:
        columns.append("physics_intensity")
    sample = make_cloud(n, seed=seed, columns=tuple(columns),
                        has_reflectance=has_reflectance, sensor=small_sensor())
    return project(sample, small_sensor())


GENERIC_LAYOUT = {"format": "bin", "columns": ["x", "y", "z", "intensity"]}


def config_for(tmp_path, source="voxelscape", target="kitti",
               source_layout=None, target_layout=None, **overrides):
    """Build a run config. The generic adapter must declare its layout up front."""
    data = {
        "source": {"dataset": source, "path": str(tmp_path / "src"),
                   **(source_layout or (GENERIC_LAYOUT if source == "generic" else {}))},
        "target": {"dataset": target, "path": str(tmp_path / "tgt"),
                   **(target_layout or (GENERIC_LAYOUT if target == "generic" else {}))},
        "task": {"type": "sensor"},
        "sensor": dict(SMALL),
        "model": {"type": "picgan"},
        # These fixtures point at synthetic dirs with nothing to measure, so they
        # ask for the published constants; computed stats are covered in
        # test_statistics.py against real data.
        "normalization": {"source": "picgan_default"},
        "output": {"checkpoint_dir": str(tmp_path / "ckpt")},
    }
    data.update(overrides)
    return Config.from_dict(data)


# --------------------------------------------------------------------------- #
# Channel-order parity with the adapter
# --------------------------------------------------------------------------- #


def test_projection_feeds_the_adapter_without_restating_transforms(model):
    """The stack projection builds is exactly what the adapter expects, by name."""
    projected = source_sample()
    stack = model.source_stack(projected)
    assert stack.shape == (4, SMALL["proj_H"], SMALL["proj_W"])
    assert np.array_equal(stack[0], projected.channel("range"))
    assert np.array_equal(stack[1], projected.channel("incidence"))
    assert np.array_equal(stack[2], projected.channel("reflectance"))
    assert np.array_equal(stack[3], projected.phy[0])


def test_two_channel_projection_feeds_the_adapter(model):
    projected = source_sample(has_reflectance=False)
    stack = model.source_stack(projected)
    assert stack.shape == (3, SMALL["proj_H"], SMALL["proj_W"])
    assert np.array_equal(stack[2], projected.phy[0]), "phy stays the last channel"
    sim, _, phy = model.to_tensors(stack, np.zeros((2,) + stack.shape[1:], dtype=np.float32))
    assert sim.shape[0] == 2 and phy.shape[0] == 1


def test_normalization_still_comes_from_picgan(model):
    """Projection must not normalise: the adapter owns that, using PICGAN's constants."""
    projected = source_sample()
    stack = model.source_stack(projected)
    sim, _, _ = model.to_tensors(stack, np.zeros((2,) + stack.shape[1:], dtype=np.float32))
    mean, std = model.normaweather_modeltion("lidar_transform")
    assert torch.allclose(sim[0], torch.from_numpy((stack[0] - mean) / std), atol=1e-6)


def test_projected_sample_flows_through_generate(model):
    projected = source_sample()
    model.build_for(projected)
    out = model.generate(projected)
    assert out.shape == (1, 1, SMALL["proj_H"], SMALL["proj_W"])


# --------------------------------------------------------------------------- #
# Full round trip through the model
# --------------------------------------------------------------------------- #


def test_end_to_end_round_trip_preserves_geometry(model, tmp_path):
    source = make_cloud(1200, seed=3, columns=("x", "y", "z", "intensity", "reflectance",
                                               "physics_intensity"),
                        has_reflectance=True, sensor=small_sensor())
    generator = IntensityGenerator(config_for(tmp_path), model=model)
    frame = generator.generate_frame(source)

    assert np.array_equal(frame.points[:, :3], source.points[:, :3]), "geometry untouched"
    assert frame.points.shape == source.points.shape
    assert frame.stats["n_written"] + frame.stats["n_dropped"] == source.points.shape[0]
    assert frame.stats["n_written"] > 0


def test_generated_intensity_is_denormalized_to_data_units(model, tmp_path):
    """gen_R emits tanh output; what lands on the cloud must be in data units."""
    generator = IntensityGenerator(config_for(tmp_path), model=model)
    frame = generator.generate_frame(source_sample(seed=4), projected=True)
    mean, std = model.normaweather_modeltion("intensity_real_transform")
    written = frame.points[frame.written, 3]
    assert np.all(np.abs(written - mean) <= std + 1e-6), "tanh output maps into mean +- std"


def test_two_channel_path_runs_end_to_end(model, tmp_path):
    """The CADC-relevant no-labels path, all the way to a written file."""
    source = make_cloud(1000, seed=5,
                        columns=("x", "y", "z", "intensity", "physics_intensity"),
                        has_reflectance=False, sensor=small_sensor())
    writer = OutputWriter(tmp_path / "out", fmt="bin", columns=("x", "y", "z", "intensity"))
    generator = IntensityGenerator(config_for(tmp_path), model=model, writer=writer)
    frame = generator.generate_frame(source)

    assert model.in_channels_s == 2
    assert frame.path.is_file()
    reread = np.fromfile(frame.path, dtype=np.float32).reshape(-1, 4)
    assert reread.shape == (1000, 4)
    assert np.allclose(reread[:, :3], source.points[:, :3])


def test_pipeline_is_deterministic(workspace, tmp_path):
    source = make_cloud(800, seed=6, columns=("x", "y", "z", "intensity", "physics_intensity"),
                        has_reflectance=False, sensor=small_sensor())
    outputs = []
    for _ in range(2):
        torch.manual_seed(99)
        model = PicganAdapter(workspace=workspace, device="cpu")
        model.build_model(2)
        outputs.append(
            IntensityGenerator(config_for(tmp_path), model=model).generate_frame(source).points
        )
    assert np.array_equal(outputs[0], outputs[1])


def test_missing_phy_is_refused_by_the_sensor_path(model, tmp_path):
    """Sensor transfer needs phy from the simulator; nothing invents it."""
    from reality.models import PicganAdapterError

    source = make_cloud(300, seed=7, has_reflectance=False, sensor=small_sensor())
    generator = IntensityGenerator(config_for(tmp_path), model=model)
    with pytest.raises(PicganAdapterError, match="phy is not set"):
        generator.generate_frame(source)


# --------------------------------------------------------------------------- #
# Config-driven assembly: no per-dataset code
# --------------------------------------------------------------------------- #


def test_source_and_target_are_resolved_from_config(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tgt").mkdir()
    generator = IntensityGenerator(config_for(tmp_path, source="generic", target="generic"))
    assert generator.source_adapter().name == "generic"
    assert generator.target_adapter().name == "generic"


@pytest.mark.parametrize("target", ["kitti", "nuscenes", "cadc", "boreas"])
def test_switching_target_needs_no_pipeline_change(tmp_path, target):
    """kitti -> nuscenes is a config edit, not a code path.

    Resolution is what matters here: the target name in config selects the
    adapter class through the registry, with no branch anywhere in the pipeline.
    """
    config = config_for(tmp_path, target=target)
    assert DATASETS.get(config.target.dataset) is DATASETS.get(target)
    assert DATASETS.get(config.target.dataset).name == target
    from reality.core.pipeline import plan_stages
    assert [s.role for s in plan_stages(config)] == [
        "projection", "model", "backprojection"
    ], "the sensor path never gains a degradation stage"


def test_model_is_resolved_by_name_from_the_registry(tmp_path):
    generator = IntensityGenerator(config_for(tmp_path))
    model = generator.build_model(source_sample())
    assert isinstance(model, MODELS.get("picgan"))
    assert model.in_channels_s == 3


def test_model_channel_count_follows_the_sample(tmp_path, workspace):
    generator = IntensityGenerator(config_for(tmp_path))
    generator.build_model(source_sample(has_reflectance=False))
    assert generator.model.in_channels_s == 2


def test_run_over_a_dataset_writes_every_frame(model, tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    rng = np.random.default_rng(8)
    for i in range(3):
        cloud = make_cloud(400, seed=i, columns=("x", "y", "z", "intensity",
                                                 "physics_intensity"),
                           has_reflectance=False, sensor=small_sensor())
        cloud.points.tofile(source_dir / f"{i:06d}.bin")

    config = config_for(tmp_path, source="generic", source_layout={
        "format": "bin", "columns": ["x", "y", "z", "intensity", "physics_intensity"]})
    writer = OutputWriter(tmp_path / "out", columns=("x", "y", "z", "intensity"))
    frames = IntensityGenerator(config, model=model, writer=writer).run()

    assert len(frames) == 3
    assert all(f.path.is_file() for f in frames)
    assert {f.stats["frame_id"] for f in frames} == {"000000", "000001", "000002"}


def test_run_respects_a_limit(model, tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    for i in range(4):
        make_cloud(200, seed=i, columns=("x", "y", "z", "intensity", "physics_intensity"),
                   has_reflectance=False, sensor=small_sensor()).points.tofile(
            source_dir / f"{i:06d}.bin")
    config = config_for(tmp_path, source="generic", source_layout={
        "format": "bin", "columns": ["x", "y", "z", "intensity", "physics_intensity"]})
    assert len(IntensityGenerator(config, model=model).run(limit=2)) == 2


def test_empty_source_is_reported(model, tmp_path):
    (tmp_path / "src").mkdir()
    config = config_for(tmp_path, source="generic")
    with pytest.raises(GenerationError, match="no frames found"):
        IntensityGenerator(config, model=model).run()


def test_stages_are_registered_for_pipeline_assembly(tmp_path):
    from reality.core.pipeline import Pipeline

    pipeline = Pipeline.from_config(config_for(tmp_path))
    assert pipeline.stage_names == ["projection", "picgan", "backprojection"]
    assert "projection" in TRANSFORMS and "backprojection" in TRANSFORMS


# --------------------------------------------------------------------------- #
# Scale independence
# --------------------------------------------------------------------------- #


def test_stream_yields_frames_one_at_a_time(model, tmp_path):
    """Nothing about a run is proportional to the dataset size."""
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    for i in range(5):
        make_cloud(200, seed=i, columns=("x", "y", "z", "intensity", "physics_intensity"),
                   has_reflectance=False, sensor=small_sensor()).points.tofile(
            source_dir / f"{i:06d}.bin")
    config = config_for(tmp_path, source="generic", source_layout={
        "format": "bin", "columns": ["x", "y", "z", "intensity", "physics_intensity"]})

    generator = IntensityGenerator(config, model=model)
    stream = generator.stream()
    import types

    assert isinstance(stream, types.GeneratorType), "the run must stream, not materialise"
    first = next(stream)
    assert first.stats["n_points"] == 200
    assert len(list(stream)) == 4, "the rest arrive lazily"


def test_streaming_does_not_retain_frames(model, tmp_path):
    """A caller that discards frames keeps memory flat regardless of scale."""
    import weakref

    source_dir = tmp_path / "src"
    source_dir.mkdir()
    for i in range(4):
        make_cloud(200, seed=i, columns=("x", "y", "z", "intensity", "physics_intensity"),
                   has_reflectance=False, sensor=small_sensor()).points.tofile(
            source_dir / f"{i:06d}.bin")
    config = config_for(tmp_path, source="generic", source_layout={
        "format": "bin", "columns": ["x", "y", "z", "intensity", "physics_intensity"]})

    references = []
    for frame in IntensityGenerator(config, model=model).stream():
        references.append(weakref.ref(frame))
    del frame  # the loop variable is the last strong reference
    import gc

    gc.collect()
    assert all(reference() is None for reference in references), (
        "frames must be collectable once consumed"
    )


def test_generation_works_without_labels(model, tmp_path):
    """Inference must not require the labels the source was trained with."""
    source = make_cloud(300, seed=2,
                        columns=("x", "y", "z", "intensity", "physics_intensity"),
                        has_reflectance=False, sensor=small_sensor())
    assert source.meta.has_reflectance is False
    frame = IntensityGenerator(config_for(tmp_path), model=model).generate_frame(source)
    assert frame.points.shape == source.points.shape


def test_clamp_bounds_the_written_intensity(model, tmp_path):
    """gen_R ends in tanh, so denormalised output can undershoot zero."""
    source = make_cloud(400, seed=3,
                        columns=("x", "y", "z", "intensity", "physics_intensity"),
                        has_reflectance=False, sensor=small_sensor())
    clamped = IntensityGenerator(config_for(tmp_path), model=model,
                                 clamp=(0.0, 1.0)).generate_frame(source)
    written = clamped.points[clamped.written, 3]
    assert written.min() >= 0.0 and written.max() <= 1.0


def test_unclamped_output_is_left_alone(model, tmp_path):
    source = make_cloud(400, seed=4,
                        columns=("x", "y", "z", "intensity", "physics_intensity"),
                        has_reflectance=False, sensor=small_sensor())
    raw = IntensityGenerator(config_for(tmp_path), model=model).generate_frame(source)
    clamped = IntensityGenerator(config_for(tmp_path), model=model,
                                 clamp=(0.0, 1.0)).generate_frame(source)
    assert not np.array_equal(raw.points[:, 3], clamped.points[:, 3]) or \
        raw.points[raw.written, 3].min() >= 0.0
