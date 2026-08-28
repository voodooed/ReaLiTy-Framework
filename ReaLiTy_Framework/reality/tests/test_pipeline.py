"""Stage planning and pipeline assembly."""

import numpy as np
import pytest

from reality.core.config import Config
from reality.core.context import Sample, SampleMeta
from reality.core.pipeline import Pipeline, Stage, StageSpec, plan_stages
from reality.core.registry import Registry, RegistryError


class RecordingStage(Stage):
    """A no-op stage that records that it ran."""

    name = "recording"

    def apply(self, sample: Sample) -> Sample:
        visited = list(sample.meta.extra.get("visited", []))
        visited.append(self.name)
        sample.meta.extra["visited"] = visited
        return sample


def stage_class(stage_name):
    return type(f"{stage_name.title()}Stage", (RecordingStage,), {"name": stage_name})


@pytest.fixture
def sample():
    return Sample(
        points=np.zeros((4, 4), dtype=np.float32),
        meta=SampleMeta(dataset="voxelscape", task="sensor"),
    )


@pytest.fixture
def registries():
    """Registries populated with stub stages, so assembly can be tested alone."""
    transforms, models, degradations = Registry("transforms"), Registry("models"), Registry(
        "degradations"
    )
    transforms.register("projection", stage_class("projection"))
    transforms.register("backprojection", stage_class("backprojection"))
    models.register("picgan", stage_class("picgan"))
    degradations.register("physics", stage_class("physics"))
    return {"transforms": transforms, "models": models, "degradations": degradations}


# -- planning --------------------------------------------------------------- #


def test_sensor_plan_skips_degradation(sensor_cfg_dict):
    plan = plan_stages(Config.from_dict(sensor_cfg_dict))
    assert [s.role for s in plan] == ["projection", "model", "backprojection"]
    assert [s.name for s in plan] == ["projection", "picgan", "backprojection"]


def test_weather_plan_includes_degradation(weather_cfg_dict):
    plan = plan_stages(Config.from_dict(weather_cfg_dict))
    # Degradation runs on the 3D cloud, before projection (the weather model consumes (N, 4)).
    assert [s.role for s in plan] == ["degradation", "projection", "model", "backprojection"]
    degradation = plan[0]
    assert (degradation.namespace, degradation.name) == ("degradations", "physics")


def test_plan_follows_model_type(sensor_cfg_dict):
    sensor_cfg_dict["model"]["type"] = "some_future_model"
    plan = plan_stages(Config.from_dict(sensor_cfg_dict))
    assert plan[1].namespace == "models"
    assert plan[1].name == "some_future_model"


def test_stage_spec_str():
    assert str(StageSpec("model", "models", "picgan")) == "model(models:picgan)"


# -- assembly and running --------------------------------------------------- #


def test_from_config_builds_sensor_pipeline(sensor_cfg_dict, registries, sample):
    cfg = Config.from_dict(sensor_cfg_dict)
    pipeline = Pipeline.from_config(cfg, registries)
    assert pipeline.stage_names == ["projection", "picgan", "backprojection"]
    out = pipeline.run(sample)
    assert out.meta.extra["visited"] == ["projection", "picgan", "backprojection"]


def test_from_config_builds_weather_pipeline(weather_cfg_dict, registries, sample):
    cfg = Config.from_dict(weather_cfg_dict)
    pipeline = Pipeline.from_config(cfg, registries)
    assert pipeline.run(sample).meta.extra["visited"] == [
        "physics", "projection", "picgan", "backprojection",
    ]


def test_stages_receive_the_config(sensor_cfg_dict, registries):
    cfg = Config.from_dict(sensor_cfg_dict)
    pipeline = Pipeline.from_config(cfg, registries)
    assert all(stage.config is cfg for stage in pipeline.stages)


def test_unregistered_stage_names_the_missing_piece(sensor_cfg_dict, registries):
    registries["models"].unregister("picgan")
    cfg = Config.from_dict(sensor_cfg_dict)
    with pytest.raises(RegistryError, match="picgan"):
        Pipeline.from_config(cfg, registries)


def test_missing_namespace_is_reported(sensor_cfg_dict, registries):
    del registries["transforms"]
    cfg = Config.from_dict(sensor_cfg_dict)
    with pytest.raises(RegistryError, match="no registry for namespace 'transforms'"):
        Pipeline.from_config(cfg, registries)


def test_empty_pipeline_is_a_noop(sample):
    pipeline = Pipeline()
    assert len(pipeline) == 0
    assert pipeline.run(sample) is sample
    assert repr(pipeline) == "Pipeline(<empty>)"


def test_run_all(sample):
    pipeline = Pipeline([stage_class("projection")()])
    out = pipeline.run_all([sample, sample.replace()])
    assert len(out) == 2


def test_stage_must_return_a_sample(sample):
    class BadStage(Stage):
        name = "bad"

        def apply(self, sample):
            return "not a sample"

    with pytest.raises(TypeError, match="returned str, expected Sample"):
        Pipeline([BadStage()]).run(sample)


def test_stage_is_callable(sample):
    stage = stage_class("projection")()
    assert stage(sample).meta.extra["visited"] == ["projection"]


def test_stage_is_abstract():
    with pytest.raises(TypeError):
        Stage()  # apply() is abstract


def test_repr_lists_stages(sensor_cfg_dict, registries):
    pipeline = Pipeline.from_config(Config.from_dict(sensor_cfg_dict), registries)
    assert repr(pipeline) == "Pipeline(projection -> picgan -> backprojection)"
