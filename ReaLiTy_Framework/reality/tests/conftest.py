"""Shared fixtures: minimal valid configs, built as dicts so tests stay explicit."""

import copy

import pytest
import yaml

SENSOR_CONFIG = {
    "source": {"dataset": "voxelscape"},
    "target": {"dataset": "kitti"},
    "task": {"type": "sensor"},
    "geometric_degradation": {"enabled": False},
    "model": {"type": "picgan"},
    "training": {
        "batch_size": 8,
        "epochs": 200,
        "learning_rate": 1.0e-5,
        "lambda_cycle": 10,
        "lambda_physics": 10,
    },
    "output": {"checkpoint_dir": "checkpoints/voxelscape_to_kitti"},
}

WEATHER_CONFIG = {
    "source": {"dataset": "voxelscape"},
    "target": {"dataset": "cadc"},
    "task": {"type": "weather"},
    "geometric_degradation": {
        "enabled": True,
        "type": "physics",
        "weather": "snow",
        "precipitation_rate": 30.0,
    },
    "model": {"type": "picgan"},
    "training": {"batch_size": 8, "epochs": 200, "learning_rate": 1.0e-5,
                 "lambda_cycle": 10, "lambda_physics": 10},
    "output": {"checkpoint_dir": "checkpoints/voxelscape_to_cadc"},
}


@pytest.fixture
def sensor_cfg_dict():
    return copy.deepcopy(SENSOR_CONFIG)


@pytest.fixture
def weather_cfg_dict():
    return copy.deepcopy(WEATHER_CONFIG)


@pytest.fixture
def write_yaml(tmp_path):
    """Write a dict to a YAML file and return its path."""

    def _write(data, name="run.yaml"):
        path = tmp_path / name
        path.write_text(yaml.safe_dump(data) if not isinstance(data, str) else data)
        return path

    return _write
