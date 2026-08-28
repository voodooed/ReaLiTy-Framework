"""Config loading, validation and snapshotting."""

import copy
from pathlib import Path

import pytest
import yaml

from reality.core.config import (
    Config,
    ConfigError,
    DataSpec,
    OutputSpec,
    SensorSpec,
    load_config,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
SHIPPED_CONFIGS = sorted(CONFIG_DIR.rglob("*.yaml"))


# --------------------------------------------------------------------------- #
# Loading valid configs
# --------------------------------------------------------------------------- #


def test_shipped_configs_exist():
    names = {p.name for p in SHIPPED_CONFIGS}
    assert names >= {"kitti_to_cadc.yaml", "voxelscape_to_kitti.yaml",
                     "template_sensor.yaml"}
    # configs are organised by task
    tasks = {p.parent.name for p in SHIPPED_CONFIGS}
    assert tasks == {"sensor", "weather"}, f"unexpected config folders: {tasks}"


@pytest.mark.parametrize("path", SHIPPED_CONFIGS, ids=lambda p: p.name)
def test_every_shipped_config_loads(path):
    cfg = load_config(path)
    assert cfg.source.dataset and cfg.target.dataset
    assert cfg.task.type in ("sensor", "weather")


def test_sensor_config_fields(sensor_cfg_dict):
    cfg = Config.from_dict(sensor_cfg_dict)
    assert cfg.source.dataset == "voxelscape"
    assert cfg.target.dataset == "kitti"
    assert cfg.task.type == "sensor"
    assert cfg.task.is_weather is False
    assert cfg.geometric_degradation.enabled is False
    assert cfg.model.type == "picgan"
    assert cfg.training.batch_size == 8
    assert cfg.training.learning_rate == pytest.approx(1e-5)
    assert cfg.run_name == "voxelscape_to_kitti"


def test_weather_config_fields(weather_cfg_dict):
    cfg = Config.from_dict(weather_cfg_dict)
    deg = cfg.geometric_degradation
    assert (deg.enabled, deg.type, deg.weather) == (True, "physics", "snow")
    assert deg.precipitation_rate == pytest.approx(30.0)
    assert cfg.task.is_weather is True


def test_defaults_are_applied_when_sections_omitted():
    cfg = Config.from_dict(
        {"source": {"dataset": "voxelscape"}, "target": {"dataset": "kitti"},
         "task": {"type": "sensor"}}
    )
    assert cfg.model.type == "picgan"
    assert (cfg.training.batch_size, cfg.training.epochs) == (8, 200)
    assert cfg.training.learning_rate == pytest.approx(1e-5)
    assert (cfg.training.lambda_cycle, cfg.training.lambda_physics) == (10.0, 10.0)
    assert cfg.geometric_degradation.enabled is False
    assert cfg.sensor is None


def test_checkpoint_dir_is_derived_when_output_omitted():
    cfg = Config.from_dict(
        {"source": {"dataset": "voxelscape"}, "target": {"dataset": "nuscenes"},
         "task": {"type": "sensor"}}
    )
    assert cfg.output == OutputSpec(checkpoint_dir="checkpoints/voxelscape_to_nuscenes")


def test_generic_adapter_layout_is_parsed():
    cfg = load_config(CONFIG_DIR / "sensor" / "template_sensor.yaml")
    assert cfg.source.columns == ("x", "y", "z", "intensity")
    assert cfg.target.columns == ("x", "y", "z", "intensity", "ring")
    assert cfg.target.intensity_scale == pytest.approx(255.0)
    assert cfg.source.has_labels is True
    assert cfg.source.labels.format == "semantickitti"
    assert cfg.sensor == SensorSpec(proj_H=64, proj_W=1024, fov_up=2.0, fov_down=-24.9)


def test_labels_optional_marks_no_labels(sensor_cfg_dict):
    sensor_cfg_dict["source"] = {"dataset": "generic", "path": "/d", "format": "bin",
                                 "columns": ["x", "y", "z", "intensity"]}
    cfg = Config.from_dict(sensor_cfg_dict)
    assert cfg.source.has_labels is False


def test_integers_are_accepted_for_float_fields(sensor_cfg_dict):
    sensor_cfg_dict["training"]["learning_rate"] = 1
    cfg = Config.from_dict(sensor_cfg_dict)
    assert isinstance(cfg.training.learning_rate, float)
    assert cfg.training.learning_rate == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Rejecting invalid configs
# --------------------------------------------------------------------------- #


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml(write_yaml):
    path = write_yaml("source: {dataset: voxelscape\n", name="broken.yaml")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(path)


def test_non_mapping_yaml(write_yaml):
    path = write_yaml("- just\n- a list\n", name="list.yaml")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(path)


@pytest.mark.parametrize("missing", ["source", "target", "task"])
def test_required_sections(sensor_cfg_dict, missing):
    del sensor_cfg_dict[missing]
    with pytest.raises(ConfigError, match=f"{missing}: required"):
        Config.from_dict(sensor_cfg_dict)


def test_unknown_top_level_key_rejected(sensor_cfg_dict):
    sensor_cfg_dict["trainng"] = {"batch_size": 4}
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict(sensor_cfg_dict)


def test_unknown_nested_key_rejected(sensor_cfg_dict):
    sensor_cfg_dict["training"]["lr"] = 0.1
    with pytest.raises(ConfigError, match=r"training: unknown key\(s\) \['lr'\]"):
        Config.from_dict(sensor_cfg_dict)


def test_bad_task_type(sensor_cfg_dict):
    sensor_cfg_dict["task"]["type"] = "fog"
    with pytest.raises(ConfigError, match="task.type"):
        Config.from_dict(sensor_cfg_dict)


def test_weather_task_requires_degradation(weather_cfg_dict):
    weather_cfg_dict["geometric_degradation"] = {"enabled": False}
    with pytest.raises(ConfigError, match="must be true for task.type='weather'"):
        Config.from_dict(weather_cfg_dict)


def test_sensor_task_rejects_degradation(sensor_cfg_dict):
    sensor_cfg_dict["geometric_degradation"] = {
        "enabled": True, "type": "physics", "weather": "snow", "precipitation_rate": 10.0,
    }
    with pytest.raises(ConfigError, match="must be false for task.type='sensor'"):
        Config.from_dict(sensor_cfg_dict)


@pytest.mark.parametrize(
    "patch, match",
    [
        ({"type": None}, "type: required when enabled"),
        ({"type": "magic"}, "geometric_degradation.type"),
        ({"weather": "fog"}, "geometric_degradation.weather"),
        ({"weather": None}, "geometric_degradation.weather"),
        ({"precipitation_rate": None}, "precipitation_rate: required"),
        ({"precipitation_rate": 0.0}, "must be > 0"),
        ({"precipitation_rate": -5.0}, "must be > 0"),
    ],
)
def test_degradation_validation(weather_cfg_dict, patch, match):
    weather_cfg_dict["geometric_degradation"].update(patch)
    with pytest.raises(ConfigError, match=match):
        Config.from_dict(weather_cfg_dict)


@pytest.mark.parametrize(
    "patch, match",
    [
        ({"batch_size": 0}, "batch_size: must be > 0"),
        ({"epochs": -1}, "epochs: must be > 0"),
        ({"learning_rate": 0}, "learning_rate: must be > 0"),
        ({"lambda_cycle": -1}, "lambda_cycle: must be >= 0"),
        ({"lambda_physics": -0.5}, "lambda_physics: must be >= 0"),
        ({"num_workers": -1}, "num_workers: must be >= 0"),
    ],
)
def test_training_validation(sensor_cfg_dict, patch, match):
    sensor_cfg_dict["training"].update(patch)
    with pytest.raises(ConfigError, match=match):
        Config.from_dict(sensor_cfg_dict)


@pytest.mark.parametrize(
    "value, match",
    [
        ("eight", "expected an integer"),
        (True, "expected an integer"),
        (8.5, "expected an integer"),
    ],
)
def test_type_errors_are_reported_with_path(sensor_cfg_dict, value, match):
    sensor_cfg_dict["training"]["batch_size"] = value
    with pytest.raises(ConfigError, match="training.batch_size"):
        Config.from_dict(sensor_cfg_dict)
    sensor_cfg_dict["training"]["batch_size"] = value
    with pytest.raises(ConfigError, match=match):
        Config.from_dict(sensor_cfg_dict)


def test_enabled_must_be_boolean(weather_cfg_dict):
    weather_cfg_dict["geometric_degradation"]["enabled"] = "yes-please"
    with pytest.raises(ConfigError, match="expected a boolean"):
        Config.from_dict(weather_cfg_dict)


def test_section_must_be_mapping(sensor_cfg_dict):
    sensor_cfg_dict["training"] = 8
    with pytest.raises(ConfigError, match="training: expected a mapping"):
        Config.from_dict(sensor_cfg_dict)


def test_columns_must_be_a_list(sensor_cfg_dict):
    sensor_cfg_dict["source"] = {"dataset": "generic", "path": "/d", "format": "bin",
                                 "columns": "x,y,z"}
    with pytest.raises(ConfigError, match="source.columns: expected a list"):
        Config.from_dict(sensor_cfg_dict)


@pytest.mark.parametrize("drop", ["path", "format", "columns"])
def test_generic_adapter_requires_declared_layout(sensor_cfg_dict, drop):
    source = {"dataset": "generic", "path": "/d", "format": "bin",
              "columns": ["x", "y", "z", "intensity"]}
    del source[drop]
    sensor_cfg_dict["source"] = source
    with pytest.raises(ConfigError, match=f"source.{drop}"):
        Config.from_dict(sensor_cfg_dict)


def test_empty_dataset_name_rejected(sensor_cfg_dict):
    sensor_cfg_dict["target"]["dataset"] = ""
    with pytest.raises(ConfigError, match="target.dataset"):
        Config.from_dict(sensor_cfg_dict)


def test_intensity_scale_must_be_positive(sensor_cfg_dict):
    sensor_cfg_dict["target"]["intensity_scale"] = 0
    with pytest.raises(ConfigError, match="target.intensity_scale"):
        Config.from_dict(sensor_cfg_dict)


def test_sensor_geometry_validation(sensor_cfg_dict):
    sensor_cfg_dict["sensor"] = {"proj_H": 64, "proj_W": 1024, "fov_up": -24.9, "fov_down": 2.0}
    with pytest.raises(ConfigError, match="fov_up"):
        Config.from_dict(sensor_cfg_dict)
    sensor_cfg_dict["sensor"] = {"proj_H": 0, "proj_W": 1024, "fov_up": 2.0, "fov_down": -24.9}
    with pytest.raises(ConfigError, match="proj_H/proj_W"):
        Config.from_dict(sensor_cfg_dict)


def test_config_must_be_a_mapping():
    with pytest.raises(ConfigError, match="must be a mapping"):
        Config.from_dict(["source"])


# --------------------------------------------------------------------------- #
# Snapshot
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", SHIPPED_CONFIGS, ids=lambda p: p.name)
def test_snapshot_round_trips(path, tmp_path):
    original = load_config(path)
    snap = original.snapshot(tmp_path / "snap" / "config.yaml")
    assert snap.is_file()
    assert load_config(snap) == original


def test_snapshot_is_plain_yaml(weather_cfg_dict, tmp_path):
    cfg = Config.from_dict(weather_cfg_dict)
    data = yaml.safe_load(cfg.snapshot(tmp_path / "c.yaml").read_text())
    assert data["task"] == {"type": "weather"}
    assert data["geometric_degradation"]["weather"] == "snow"
    assert data["output"]["checkpoint_dir"] == "checkpoints/voxelscape_to_cadc"


def test_to_dict_round_trips(sensor_cfg_dict):
    cfg = Config.from_dict(sensor_cfg_dict)
    assert Config.from_dict(cfg.to_dict()) == cfg


def test_snapshot_records_resolved_defaults(tmp_path):
    cfg = Config.from_dict(
        {"source": {"dataset": "voxelscape"}, "target": {"dataset": "kitti"},
         "task": {"type": "sensor"}}
    )
    data = yaml.safe_load(cfg.snapshot(tmp_path / "c.yaml").read_text())
    # A snapshot must be complete on its own: defaults are written out, not implied.
    assert data["training"]["epochs"] == 200
    assert data["output"]["checkpoint_dir"] == "checkpoints/voxelscape_to_kitti"
    assert data["model"]["type"] == "picgan"


def test_snapshot_omits_unset_optional_sections(sensor_cfg_dict, tmp_path):
    cfg = Config.from_dict(sensor_cfg_dict)
    data = yaml.safe_load(cfg.snapshot(tmp_path / "c.yaml").read_text())
    assert "sensor" not in data
    assert "labels" not in data["source"]


def test_dataspec_equality_ignores_construction_route():
    a = DataSpec(dataset="kitti")
    b = Config.from_dict(
        {"source": {"dataset": "kitti"}, "target": {"dataset": "kitti"},
         "task": {"type": "sensor"}}
    ).source
    assert a == b
