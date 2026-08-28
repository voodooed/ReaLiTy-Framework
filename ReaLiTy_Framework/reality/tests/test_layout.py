"""Repository layout checks."""

import hashlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "reality"
PICGAN = PACKAGE / "models" / "PICGAN"


def test_package_layout():
    for name in [
        "core", "datasets", "preprocessing", "degradation", "physics", "models",
        "training", "inference", "evaluation", "postprocessing", "io", "configs",
    ]:
        assert (PACKAGE / name).is_dir(), f"missing package directory: {name}"
    for name in ["__init__.py", "__main__.py", "cli.py"]:
        assert (PACKAGE / name).is_file(), f"missing module: {name}"
    for name in ["config.py", "registry.py", "pipeline.py", "context.py", "version.py"]:
        assert (PACKAGE / "core" / name).is_file(), f"missing core module: {name}"


def test_picgan_lives_under_models():
    assert PICGAN.is_dir(), "PICGAN must live at reality/models/PICGAN"
    assert not (REPO_ROOT / "model").exists(), "the old model/ directory should be gone"
