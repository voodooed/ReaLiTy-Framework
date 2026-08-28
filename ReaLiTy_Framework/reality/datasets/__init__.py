"""Dataset adapters. Importing this package registers every adapter by name."""

from reality.datasets.base import DatasetAdapter, DatasetError, FrameRef
from reality.datasets.boreas import BoreasAdapter
from reality.datasets.cadc import CadcAdapter
from reality.datasets.generic import GenericAdapter
from reality.datasets.kitti import KittiAdapter
from reality.datasets.nuscenes import NuScenesAdapter
from reality.datasets.voxelscape import VoxelScapeAdapter

__all__ = [
    "DatasetAdapter", "DatasetError", "FrameRef",
    "BoreasAdapter", "CadcAdapter", "GenericAdapter", "KittiAdapter",
    "NuScenesAdapter", "VoxelScapeAdapter",
]
