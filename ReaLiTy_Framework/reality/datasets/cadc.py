"""CADC (Canadian Adverse Driving Conditions) weather-target adapter.

Expected raw layout::

    <root>/cadcd/{2018_03_06,2018_03_07,2019_02_27}/{drive}/labeled/lidar_points/data/*.bin
                                                          /labeled/lidar_points/timestamps.txt
                                                          /labeled/novatel/data/*.txt
                                                   {drive}/3d_ann.json
                                   {date}/calib/{00..07}.yaml, extrinsics.yaml

Clouds are headerless float32, 4 columns (x, y, z, intensity). The 4-column
reading is the only one under which column 2 is a plausible z (-3.8 to 20.0 m)
while x and y span +-190 m; intensity is 0-1 on an exact k/255 grid, i.e. the
8-bit sensor value already normalised.

CADC has no per-point semantic labels — ``3d_ann.json`` holds object-level 3D
cuboids — so this adapter always takes the no-reflectance path. That is correct
for a target domain, which PICGAN consumes as a single intensity channel.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from reality.core.config import SensorSpec
from reality.core.config import dataset_folder
from reality.core.registry import DATASETS
from reality.datasets.base import DatasetAdapter, DatasetError, FrameRef, read_bin

#: Velodyne VLP-32C. Beam clustering on 2018_03_06/0001 frame 0 recovers exactly
#: 32 beams spanning -24.99 deg to +14.99 deg.
CADC_SENSOR = SensorSpec(proj_H=32, proj_W=1024, fov_up=15.0, fov_down=-25.0)

LIDAR_REL = Path("labeled/lidar_points/data")


@DATASETS.register("cadc")
class CadcAdapter(DatasetAdapter):
    """CADC snow-condition clouds, used as the real/target domain."""

    name = "cadc"
    columns = ("x", "y", "z", "intensity")
    intensity_scale = 1.0  # already 0-1; the underlying sensor value is 8-bit / 255
    default_sensor = CADC_SENSOR

    def __init__(self, *args, dates=None, drives=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.root is None:
            raise DatasetError(
                "cadc: no location. Set data_root so 'cadc' resolves to "
                "<data_root>/" + dataset_folder("cadc") + ", or give an explicit path."
            )
        self.data_root = self.root / "cadcd" if (self.root / "cadcd").is_dir() else self.root
        self.dates = set(dates) if dates else None
        self.drives = set(drives) if drives else None

    def list_frames(self) -> List[FrameRef]:
        # Released layout wins when present; raw cadcd drives are the fallback.
        if self.uses_released_layout:
            return self.list_split_frames()
        frames: List[FrameRef] = []
        for date_dir in sorted(p for p in self.data_root.iterdir() if p.is_dir()):
            if self.dates and date_dir.name not in self.dates:
                continue
            for drive_dir in sorted(p for p in date_dir.iterdir() if p.is_dir()):
                if drive_dir.name == "calib":
                    continue
                if self.drives and drive_dir.name not in self.drives:
                    continue
                cloud_dir = drive_dir / LIDAR_REL
                if not cloud_dir.is_dir():
                    # e.g. 2019_02_27/0061 is listed but ships no lidar data.
                    continue
                annotations = drive_dir / "3d_ann.json"
                for bin_path in sorted(cloud_dir.glob("*.bin")):
                    frames.append(FrameRef(
                        id=f"{date_dir.name}/{drive_dir.name}/{bin_path.stem}",
                        path=bin_path,
                        extra={"date": date_dir.name, "drive": drive_dir.name,
                               "annotations": str(annotations) if annotations.is_file() else None},
                    ))
        return frames

    def load_points(self, frame: FrameRef) -> np.ndarray:
        if self.uses_released_layout:
            return self.load_split_points(frame)
        return read_bin(frame.path, len(self.columns))

    def load_labels(self, frame: FrameRef) -> Optional[np.ndarray]:
        """CADC ships no per-point labels; reflectance is unavailable by design."""
        return None
