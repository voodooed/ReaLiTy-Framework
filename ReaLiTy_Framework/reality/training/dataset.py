"""Torch Dataset over the prepared cache.

Reads the cached ``(C, H, W)`` stacks and applies the run's normalization, giving
PICGAN's ``train_fn`` exactly the ``(sim, real, phy)`` tuple it expects. Source and
target are unpaired domains, so the two folders are indexed independently and the
target is cycled -- the same thing PICGAN's own ``LidarDataset`` does by taking the
maximum of the two directory lengths.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from reality.preprocessing.cache import PrepareCache


class PreparedDataset(Dataset):
    """Cached stacks -> the (sim, real, phy) tuple, normalized."""

    def __init__(self, cache: PrepareCache, model) -> None:
        cache.require_complete()
        self.cache = cache
        self.model = model
        self.source_files = cache.source_files()
        self.target_files = cache.target_files()
        self.channels = list(cache.source_channels)
        if not self.source_files or not self.target_files:
            raise ValueError(f"prepared cache {cache.directory} holds no frames")

    def __len__(self) -> int:
        return max(len(self.source_files), len(self.target_files))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source = np.load(self.source_files[index % len(self.source_files)])
        target = np.load(self.target_files[index % len(self.target_files)])
        return self.model.to_tensors(source, target)
