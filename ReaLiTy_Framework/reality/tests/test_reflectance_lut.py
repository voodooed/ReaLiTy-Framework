"""The reflectance LUT: every id resolves, values are physical, provenance is real."""

import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

from reality.physics.reflectance import (
    DEFAULT_LUT_PATH,
    Material,
    ReflectanceLUT,
    ReflectanceLUTError,
    default_lut,
)

# Every SemanticKITTI id (config/semantic-kitti.yaml, Behley et al. ICCV 2019).
SEMANTICKITTI_IDS = [0, 1, 10, 11, 13, 15, 16, 18, 20, 30, 31, 32, 40, 44, 48, 49, 50, 51,
                     52, 60, 70, 71, 72, 80, 81, 99, 252, 253, 254, 255, 256, 257, 258, 259]
VALID_SOURCES = ("ECOSTRESS:", "Sozzi945nm:", "repo-existing", "default-assumed",
                 "informed-estimate")


@pytest.fixture(scope="module")
def lut():
    return default_lut()


def test_ships_with_the_package():
    assert DEFAULT_LUT_PATH.is_file()


def test_every_semantickitti_id_resolves(lut):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a fallback warning here means a missing id
        for class_id in SEMANTICKITTI_IDS:
            material = lut.material_for(class_id)
            assert isinstance(material, Material)
            assert lut.class_name(class_id), f"id {class_id} has no class name"


def test_all_values_are_physical(lut):
    for name, material in lut.materials.items():
        assert 0.0 <= material.reflectance <= 1.0, f"{name} outside [0, 1]"


def test_every_material_has_a_truthful_source(lut):
    for name, material in lut.materials.items():
        assert material.source.strip(), f"{name} has an empty source"
        assert material.source.startswith(VALID_SOURCES), (
            f"{name} has source {material.source!r}, which is not one of {VALID_SOURCES}"
        )
        assert material.justification, f"{name} has no justification"


def test_estimates_declare_their_anchor(lut):
    """A value we did not read from a library must say what evidence it rests on."""
    for name, material in lut.materials.items():
        if material.source == "informed-estimate":
            assert material.anchor, f"{name} is an estimate but names no anchor"


def test_measured_values_match_their_reflectance(lut):
    """measured_pct is the raw reading; reflectance is it/100, clamped at 1.0."""
    for name, material in lut.materials.items():
        if material.measured_pct is None or material.source == "informed-estimate":
            continue
        assert material.reflectance == pytest.approx(
            min(material.measured_pct / 100.0, 1.0), abs=0.01
        ), f"{name}: {material.reflectance} does not follow from {material.measured_pct}%"


def test_known_values(lut):
    """Spot-check the numbers that flow into the physics loss."""
    assert lut.reflectance_for(40) == pytest.approx(0.10)   # road / asphalt
    assert lut.reflectance_for(48) == pytest.approx(0.35)   # sidewalk / concrete
    assert lut.reflectance_for(70) == pytest.approx(0.49)   # vegetation
    assert lut.reflectance_for(72) == pytest.approx(0.55)   # terrain / grass
    assert lut.reflectance_for(51) == pytest.approx(0.32)   # fence / mixed materials
    assert lut.reflectance_for(80) == pytest.approx(0.75)   # pole / structural metal
    assert lut.reflectance_for(81) == pytest.approx(1.00)   # traffic sign / retroreflector


def test_road_is_darker_than_vegetation_and_signs_are_brightest(lut):
    assert lut.reflectance_for(40) < lut.reflectance_for(70) < lut.reflectance_for(81)


def test_moving_classes_match_their_static_counterparts(lut):
    for moving, static in [(252, 10), (253, 31), (254, 30), (255, 32),
                           (256, 16), (257, 13), (258, 18), (259, 20)]:
        assert lut.material_for(moving).name == lut.material_for(static).name


def test_vectorised_lookup(lut):
    labels = np.array([40, 70, 80, 81, 10, 0])
    values = lut.lookup(labels)
    assert values.shape == labels.shape
    assert values.dtype == np.float32
    assert values[0] == pytest.approx(0.10)
    assert values[3] == pytest.approx(1.00)
    assert np.all((values >= 0) & (values <= 1))


def test_unknown_id_falls_back_with_a_warning(lut):
    with pytest.warns(RuntimeWarning, match="not in the 'semantickitti' class map"):
        value = lut.lookup(np.array([40, 9999]))
    assert value[1] == pytest.approx(lut.materials["unknown"].reflectance)


def test_lookup_warns_once_per_missing_id_not_per_point(lut):
    labels = np.full(1000, 9999)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lut.lookup(labels)
    assert len(caught) == 1


def test_unknown_dataset_is_reported(lut):
    with pytest.raises(ReflectanceLUTError, match="no class map for dataset 'waymo'"):
        lut.lookup(np.array([1]), dataset="waymo")


def test_generic_material_table_is_dataset_independent(lut):
    """Materials are keyed by material, so a new dataset only needs a class map."""
    assert "semantickitti" in lut.class_maps
    assert set(lut.class_maps["semantickitti"].values()) <= set(lut.materials)


def test_round_trips(lut, tmp_path):
    path = tmp_path / "lut.yaml"
    path.write_text(yaml.safe_dump(lut.to_dict(), sort_keys=False))
    reloaded = ReflectanceLUT.load(path)
    assert reloaded.to_dict() == lut.to_dict()
    for class_id in SEMANTICKITTI_IDS:
        assert reloaded.reflectance_for(class_id) == lut.reflectance_for(class_id)


# -- malformed LUTs are rejected -------------------------------------------- #


def write_lut(tmp_path, materials, class_maps=None, default="unknown"):
    data = {"default_material": default, "materials": materials,
            "class_maps": class_maps or {}}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_rejects_out_of_range_value(tmp_path):
    path = write_lut(tmp_path, {"unknown": {"reflectance": 1.4, "source": "x"}})
    with pytest.raises(ReflectanceLUTError, match="outside"):
        ReflectanceLUT.load(path)


def test_rejects_missing_source(tmp_path):
    path = write_lut(tmp_path, {"unknown": {"reflectance": 0.2, "source": "  "}})
    with pytest.raises(ReflectanceLUTError, match="source"):
        ReflectanceLUT.load(path)


def test_rejects_unknown_material_reference(tmp_path):
    path = write_lut(tmp_path, {"unknown": {"reflectance": 0.2, "source": "x"}},
                     {"semantickitti": {40: {"material": "moon-cheese"}}})
    with pytest.raises(ReflectanceLUTError, match="unknown material"):
        ReflectanceLUT.load(path)


def test_rejects_missing_default_material(tmp_path):
    path = write_lut(tmp_path, {"asphalt": {"reflectance": 0.1, "source": "x"}}, default="nope")
    with pytest.raises(ReflectanceLUTError, match="default_material"):
        ReflectanceLUT.load(path)


def test_missing_file(tmp_path):
    with pytest.raises(ReflectanceLUTError, match="not found"):
        ReflectanceLUT.load(tmp_path / "nope.yaml")
