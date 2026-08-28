"""Bring-your-own-data adapter: every assumption comes from config.

Format, column layout, intensity scale, optional labels and sensor geometry are
declared in YAML (README -> *Bring-your-own data*). Nothing is inferred: a file
that does not match the declared layout is an error, not a cue to guess.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from reality.core.config import dataset_folder
from reality.core.registry import DATASETS
from reality.datasets.base import (
    DatasetAdapter,
    DatasetError,
    FrameRef,
    read_bin,
    read_npy,
    read_semantickitti_labels,
)

READERS = {"bin": read_bin, "npy": read_npy}
LABEL_SUFFIX = {"semantickitti": ".label", "npy": ".npy"}


@DATASETS.register("generic")
class GenericAdapter(DatasetAdapter):
    """Config-described dataset, for source or target, with or without labels."""

    name = "generic"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.root is None:
            raise DatasetError(
                "generic: no location. Set data_root so 'generic' resolves to "
                "<data_root>/" + dataset_folder("generic") + ", or give an explicit path."
            )
        self.format = (self.spec.format or "").lower()
        if self.format not in READERS:
            raise DatasetError(
                f"generic: unsupported format {self.spec.format!r}; "
                f"expected one of {sorted(READERS)}"
            )
        self.label_format = (self.spec.labels.format.lower() if self.spec.labels else None)
        if self.label_format is not None and self.label_format not in LABEL_SUFFIX:
            raise DatasetError(
                f"generic: unsupported labels.format {self.label_format!r}; "
                f"expected one of {sorted(LABEL_SUFFIX)}"
            )
        # Flat layout keeps labels wherever config says; the released layout has
        # them at <root>/labels, which the base class discovers.
        self.flat_label_root = Path(self.spec.labels.path) if self.spec.labels else None
        if self.label_format == "semantickitti":
            self.label_scheme = "semantickitti"

    @property
    def label_root(self):
        return super().label_root if self.uses_released_layout else self.flat_label_root

    def list_frames(self) -> List[FrameRef]:
        if self.uses_released_layout:
            return self.list_split_frames()
        if not self.root.is_dir():
            raise DatasetError(f"generic: path is not a directory: {self.root}")
        frames = []
        for path in sorted(self.root.rglob(f"*.{self.format}")):
            label_path = None
            if self.flat_label_root is not None:
                candidate = (self.flat_label_root
                             / f"{path.stem}{LABEL_SUFFIX[self.label_format]}")
                if candidate.is_file():
                    label_path = candidate
            frames.append(FrameRef(id=path.stem, path=path, label_path=label_path))
        return frames

    def load_points(self, frame: FrameRef) -> np.ndarray:
        if self.uses_released_layout:
            return self.load_split_points(frame)
        return READERS[self.format](frame.path, len(self.columns))

    def load_labels(self, frame: FrameRef) -> Optional[np.ndarray]:
        if not frame.has_labels:
            return None
        path = Path(frame.label_path)
        if self.label_format == "npy":
            return np.load(str(path)).reshape(-1).astype(np.int32)
        return read_semantickitti_labels(path)
