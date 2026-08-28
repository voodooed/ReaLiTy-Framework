"""The Sample contract passed between stages."""

import numpy as np
import pytest

from reality.core.config import SensorSpec
from reality.core.context import Sample, SampleMeta

H, W = 4, 8
CHANNELS = ("range", "incidence", "reflectance", "intensity", "mask")


def make_sample(has_reflectance=True, with_image=True, with_phy=True):
    meta = SampleMeta(
        dataset="voxelscape",
        task="sensor",
        sensor="kitti",
        fov=SensorSpec(proj_H=H, proj_W=W, fov_up=2.0, fov_down=-24.9),
        intensity_scale=1.0,
        has_reflectance=has_reflectance,
        columns=("x", "y", "z", "intensity"),
    )
    channels = CHANNELS if has_reflectance else tuple(c for c in CHANNELS if c != "reflectance")
    return Sample(
        points=np.zeros((16, 4), dtype=np.float32),
        meta=meta,
        range_image=np.zeros((len(channels), H, W), dtype=np.float32) if with_image else None,
        phy=np.zeros((1, H, W), dtype=np.float32) if with_phy else None,
        mapping=np.arange(16, dtype=np.int64) if with_image else None,
        channels=channels if with_image else (),
    )


def test_sample_shapes():
    s = make_sample().validate()
    assert s.num_points == 16
    assert s.image_shape == (H, W)
    assert s.has_phy is True


def test_sample_before_projection():
    s = make_sample(with_image=False, with_phy=False).validate()
    assert s.image_shape is None and s.has_phy is False


def test_source_channels_follows_reflectance():
    assert make_sample(has_reflectance=True).meta.source_channels == 3
    assert make_sample(has_reflectance=False).meta.source_channels == 2


def test_channel_lookup():
    s = make_sample()
    assert s.channel("intensity").shape == (H, W)
    with pytest.raises(KeyError, match="no channel 'reflectance'"):
        make_sample(has_reflectance=False).channel("reflectance")


def test_channel_requires_projection():
    with pytest.raises(ValueError, match="no range_image"):
        make_sample(with_image=False, with_phy=False).channel("range")


def test_replace_is_a_copy():
    s = make_sample()
    phy = np.ones((1, H, W), dtype=np.float32)
    t = s.replace(phy=phy)
    assert t is not s
    assert np.array_equal(t.phy, phy)
    assert np.array_equal(s.phy, np.zeros((1, H, W)))
    assert t.meta is s.meta


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("points", np.zeros(16), "points must be"),
        ("range_image", np.zeros((H, W)), "range_image must be"),
        ("phy", np.zeros((2, H, W)), "phy must be"),
        ("phy", np.zeros((1, H + 1, W)), "spatial shape"),
    ],
)
def test_validate_rejects_bad_shapes(field, value, match):
    s = make_sample().replace(**{field: value})
    with pytest.raises(ValueError, match=match):
        s.validate()


def test_validate_rejects_channel_mismatch():
    s = make_sample().replace(channels=("range", "incidence"))
    with pytest.raises(ValueError, match="do not match range_image"):
        s.validate()
