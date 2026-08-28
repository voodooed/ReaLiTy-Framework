"""Per-channel normalization for PICGAN's inputs.

The method is unchanged: every channel is z-scored with a fixed mean and standard
deviation, exactly as before. What changed is where those numbers come from.

The original constants below were fitted on VoxelScape-derived data. They do not
describe other sources -- measured on real KITTI, the incidence channel has mean
0.346 against the 0.7156 assumed here -- so ReaLiTy measures the statistics from
the configured source and target datasets and builds the transforms from those
(see reality/preprocessing/statistics.py).

PICGAN_DEFAULT_STATS is kept, and the module-level transforms below are still
built from it, so `normalization: {source: picgan_default}` and the original
main.py reproduce the published behaviour exactly.
"""

from torchvision import transforms

#: The original VoxelScape-fitted constants: channel -> (mean, std).
PICGAN_DEFAULT_STATS = {
    "range": (0.0965, 0.1068),
    "incidence": (0.7156, 0.6352),
    "reflectance": (0.2979, 0.2743),
    "phy": (0.1745, 0.1515),
    "intensity": (0.0158, 0.0462),
}

#: Which transform name each channel drives, for callers building from statistics.
CHANNEL_TRANSFORMS = {
    "range": "lidar_transform",
    "incidence": "incidence_transform",
    "reflectance": "reflectance_transform",
    "phy": "intensity_sim_transform",
    "intensity": "intensity_real_transform",
}


def make_transform(mean, std):
    """A channel transform: to tensor, then z-score with the given statistics."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),
    ])


def build_transforms(stats=None):
    """Build the five transforms from ``{channel: (mean, std)}``.

    Channels absent from ``stats`` fall back to the original constants, so a
    partially measured run (for example before the weather model supplies phy) still works.
    """
    stats = dict(stats or {})
    return {
        name: make_transform(*stats.get(channel, PICGAN_DEFAULT_STATS[channel]))
        for channel, name in CHANNEL_TRANSFORMS.items()
    }


# Module-level transforms, built from the original constants. main.py imports
# these by name, and they are what `picgan_default` reproduces.
_DEFAULTS = build_transforms()

lidar_transform = _DEFAULTS["lidar_transform"]
incidence_transform = _DEFAULTS["incidence_transform"]
reflectance_transform = _DEFAULTS["reflectance_transform"]
intensity_sim_transform = _DEFAULTS["intensity_sim_transform"]
intensity_real_transform = _DEFAULTS["intensity_real_transform"]
