"""Boreas weather-target adapter (rain).

Boreas is the rain counterpart to CADC in the weather-transfer path. No Boreas
data is present on this machine yet, so the column layout and intensity scale are
declared in config and this adapter's tests are skipped pending data.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from reality.core.config import dataset_folder
from reality.core.registry import DATASETS
from reality.datasets.base import DatasetAdapter, DatasetError, FrameRef, read_bin

LIDAR_DIRS = ("lidar", "velodyne")


@DATASETS.register("boreas")
class BoreasAdapter(DatasetAdapter):
    """Boreas rain-condition clouds, used as the real/target domain."""

    name = "boreas"
    #: Boreas ships x, y, z, intensity plus a per-point time stamp; declare it.
    columns = ("x", "y", "z", "intensity", "timestamp")
    intensity_scale = 1.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.root is None:
            raise DatasetError(
                "boreas: no location. Set data_root so 'boreas' resolves to "
                "<data_root>/" + dataset_folder("boreas") + ", or give an explicit path."
            )

    def list_frames(self) -> List[FrameRef]:
        if self.uses_released_layout:
            return self.list_split_frames()
        frames: List[FrameRef] = []
        for sequence in sorted(p for p in self.root.iterdir() if p.is_dir()):
            for name in LIDAR_DIRS:
                cloud_dir = sequence / name
                if cloud_dir.is_dir():
                    frames += [FrameRef(id=f"{sequence.name}/{p.stem}", path=p,
                                        extra={"sequence": sequence.name})
                               for p in sorted(cloud_dir.glob("*.bin"))]
                    break
        return frames

    def load_points(self, frame: FrameRef) -> np.ndarray:
        if self.uses_released_layout:
            return self.load_split_points(frame)
        return read_bin(frame.path, len(self.columns))

    def load_labels(self, frame: FrameRef) -> Optional[np.ndarray]:
        return None
