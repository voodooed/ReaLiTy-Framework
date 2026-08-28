"""nuScenes sensor-target adapter.

nuScenes clouds are 5-column ``x, y, z, intensity, ring`` float32 with intensity
on a 0-255 scale (README -> *Datasets*), so ``intensity_scale`` is 255.0. No
nuScenes data is present on this machine yet; its tests are skipped pending data.
"""

from __future__ import annotations

from typing import List

import numpy as np

from reality.core.config import SensorSpec
from reality.core.config import dataset_folder
from reality.core.registry import DATASETS
from reality.datasets.base import DatasetAdapter, DatasetError, FrameRef, read_bin

#: Velodyne HDL-32E, 32 beams, +10 to -30 degrees.
NUSCENES_SENSOR = SensorSpec(proj_H=32, proj_W=1024, fov_up=10.0, fov_down=-30.0)


@DATASETS.register("nuscenes")
class NuScenesAdapter(DatasetAdapter):
    """nuScenes LiDAR sweeps used as a sensor-transfer target."""

    name = "nuscenes"
    columns = ("x", "y", "z", "intensity", "ring")
    intensity_scale = 255.0
    default_sensor = NUSCENES_SENSOR

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.root is None:
            raise DatasetError(
                "nuscenes: no location. Set data_root so 'nuscenes' resolves to "
                "<data_root>/" + dataset_folder("nuscenes") + ", or give an explicit path."
            )

    def list_frames(self) -> List[FrameRef]:
        if self.uses_released_layout:
            return self.list_split_frames()
        return [FrameRef(id=p.stem, path=p) for p in sorted(self.root.rglob("*.bin"))]

    def load_points(self, frame: FrameRef) -> np.ndarray:
        if self.uses_released_layout:
            return self.load_split_points(frame)
        return read_bin(frame.path, len(self.columns))
