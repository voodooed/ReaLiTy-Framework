"""VoxelScape simulated-source adapter.

VoxelScape is the simulated source domain: it provides the physics-based
intensity directly, so the sensor-transfer path takes ``phy`` from the simulator
rather than from a degradation stage (see the contributor guide).

Layout is declared in config rather than assumed, because no VoxelScape export is
present on this machine yet; the tests for this adapter are skipped pending data.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from reality.core.config import dataset_folder
from reality.core.registry import DATASETS
from reality.datasets.base import DatasetAdapter, DatasetError, FrameRef, read_bin, read_npy

READERS = {"bin": read_bin, "npy": read_npy}


@DATASETS.register("voxelscape")
class VoxelScapeAdapter(DatasetAdapter):
    """Simulated source clouds carrying a physics intensity column."""

    name = "voxelscape"
    #: Declared default; override in config when an export differs.
    columns = ("x", "y", "z", "intensity", "physics_intensity")
    intensity_scale = 1.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.root is None:
            raise DatasetError(
                "voxelscape: no location. Set data_root so 'voxelscape' resolves to "
                "<data_root>/" + dataset_folder("voxelscape") + ", or give an explicit path."
            )
        self.format = (self.spec.format or "bin").lower()
        if self.format not in READERS:
            raise DatasetError(f"voxelscape: unsupported format {self.format!r}")

    @property
    def provides_physics_intensity(self) -> bool:
        """True when the declared layout carries the simulator's physics intensity."""
        return "physics_intensity" in self.columns

    def list_frames(self) -> List[FrameRef]:
        if self.uses_released_layout:
            return self.list_split_frames()
        return [FrameRef(id=p.stem, path=p)
                for p in sorted(self.root.rglob(f"*.{self.format}"))]

    def load_points(self, frame: FrameRef) -> np.ndarray:
        if self.uses_released_layout:
            return self.load_split_points(frame)
        return READERS[self.format](frame.path, len(self.columns))
