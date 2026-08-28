"""KITTI odometry / SemanticKITTI source adapter.

Expected raw layout::

    <root>/data_odometry_velodyne/dataset/sequences/{00..21}/velodyne/*.bin
    <root>/data_odometry_labels/dataset/sequences/{00..21}/labels/*.label
                                                  {00..21}/poses.txt

Clouds are headerless float32, 4 columns (x, y, z, intensity), intensity already
on 0-1. Labels exist for sequences 00-10 only (23,201 frames); 11-21 are the
unlabelled test split (20,351 frames) and exercise the 2-channel path.

The raw x, y, z, intensity columns are preserved unchanged so the downstream the weather model
stage gets the native cloud; label-derived reflectance is appended as an extra
column rather than replacing anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from reality.core.config import SensorSpec
from reality.core.config import dataset_folder
from reality.core.registry import DATASETS
from reality.datasets.base import (
    DatasetAdapter,
    DatasetError,
    FrameRef,
    read_bin,
    read_semantickitti_labels,
)

#: Velodyne HDL-64E. Vertical extent measured on sequence 00 frame 000000:
#: elevation spans -25.16 deg to +4.10 deg; these are the SemanticKITTI devkit values.
KITTI_SENSOR = SensorSpec(proj_H=64, proj_W=1024, fov_up=3.0, fov_down=-25.0)

VELODYNE_ROOTS = ("data_odometry_velodyne/dataset/sequences", "dataset/sequences", "sequences")
LABEL_ROOTS = ("data_odometry_labels/dataset/sequences", "dataset/sequences", "sequences")


def _resolve(root: Path, candidates) -> Optional[Path]:
    """First existing candidate below ``root``, else ``root`` if it holds sequences."""
    for rel in candidates:
        path = root / rel
        if path.is_dir():
            return path
    if root.is_dir() and any(p.is_dir() and p.name.isdigit() for p in root.iterdir()):
        return root
    return None


@DATASETS.register("kitti")
class KittiAdapter(DatasetAdapter):
    """KITTI odometry clouds with optional SemanticKITTI labels."""

    name = "kitti"
    columns = ("x", "y", "z", "intensity")
    intensity_scale = 1.0  # KITTI intensity is already 0-1 (observed max 0.99)
    default_sensor = KITTI_SENSOR
    label_scheme = "semantickitti"

    def __init__(self, *args, sequences=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.root is None:
            raise DatasetError(
                "kitti: no location. Set data_root so 'kitti' resolves to "
                "<data_root>/" + dataset_folder("kitti") + ", or give an explicit path."
            )
        self.velodyne_root = _resolve(self.root, VELODYNE_ROOTS)
        if self.velodyne_root is None and not self.uses_released_layout:
            raise DatasetError(
                f"kitti: {self.root} holds neither a '{self.split}' folder (the released "
                f"layout) nor raw odometry sequences (one of {list(VELODYNE_ROOTS)})"
            )
        # Raw odometry keeps labels in a parallel sequence tree; the released
        # layout keeps them at <root>/labels, which the base class finds.
        self.raw_label_root = (Path(self.spec.labels.path) if self.spec.labels is not None
                               else _resolve(self.root, LABEL_ROOTS))
        declared = sequences or getattr(self.spec, "sequences", ())
        self.sequences = [str(s).zfill(2) for s in declared] if declared else None

    @property
    def label_root(self):
        """Released layout: ``<root>/labels``. Raw layout: the sequence tree."""
        if self.uses_released_layout:
            return super().label_root
        return self.raw_label_root

    def list_frames(self) -> List[FrameRef]:
        # Released layout wins when present; raw odometry is the fallback.
        if self.uses_released_layout:
            return self.list_split_frames()
        frames: List[FrameRef] = []
        for seq_dir in sorted(p for p in self.velodyne_root.iterdir() if p.is_dir()):
            if self.sequences and seq_dir.name not in self.sequences:
                continue
            cloud_dir = seq_dir / "velodyne" if (seq_dir / "velodyne").is_dir() else seq_dir
            for bin_path in sorted(cloud_dir.glob("*.bin")):
                label_path = None
                if self.raw_label_root is not None:
                    candidate = (self.raw_label_root / seq_dir.name / "labels"
                                 / f"{bin_path.stem}.label")
                    if candidate.is_file():
                        label_path = candidate
                frames.append(FrameRef(id=f"{seq_dir.name}/{bin_path.stem}", path=bin_path,
                                       label_path=label_path,
                                       extra={"sequence": seq_dir.name}))
        return frames

    def load_points(self, frame: FrameRef) -> np.ndarray:
        if self.uses_released_layout:
            return self.load_split_points(frame)
        return read_bin(frame.path, len(self.columns))

    def load_labels(self, frame: FrameRef) -> Optional[np.ndarray]:
        if not frame.has_labels:
            return None
        return read_semantickitti_labels(Path(frame.label_path))
