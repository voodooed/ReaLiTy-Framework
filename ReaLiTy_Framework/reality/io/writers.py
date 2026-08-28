"""OutputWriter — write transformed clouds back in the target's native format.

Byte layout follows what we confirmed on the real data: KITTI and CADC
clouds are headerless float32 with the columns the adapter declares, so a written
frame is a drop-in replacement for the original file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from reality.core.context import Sample


class WriterError(ValueError):
    """Raised when a cloud cannot be written in the requested format."""


class OutputWriter:
    """Writes clouds as native ``.bin`` or ``.npy``."""

    FORMATS = ("bin", "npy")

    def __init__(self, output_dir: Union[str, Path], fmt: str = "bin",
                 columns: Optional[Sequence[str]] = None,
                 dtype: np.dtype = np.float32) -> None:
        fmt = fmt.lower()
        if fmt not in self.FORMATS:
            raise WriterError(f"unsupported output format {fmt!r}; expected {self.FORMATS}")
        self.output_dir = Path(output_dir)
        self.fmt = fmt
        self.columns = tuple(columns) if columns else None
        self.dtype = dtype

    def select_columns(self, sample: Sample, points: np.ndarray) -> np.ndarray:
        """Reduce a cloud to the declared output columns, in declared order."""
        if self.columns is None:
            return points
        available = list(sample.meta.columns)
        missing = [c for c in self.columns if c not in available]
        if missing:
            raise WriterError(
                f"cannot write columns {list(self.columns)}: {missing} not in the cloud "
                f"({available})"
            )
        return points[:, [available.index(c) for c in self.columns]]

    def path_for(self, frame_id: str) -> Path:
        """Output path for a frame id, mirroring any sub-directories in the id."""
        return self.output_dir / f"{frame_id}.{self.fmt}"

    def write(self, sample: Sample, points: Optional[np.ndarray] = None,
              frame_id: Optional[str] = None) -> Path:
        """Write one cloud and return the path."""
        cloud = np.asarray(sample.points if points is None else points, dtype=self.dtype)
        cloud = np.ascontiguousarray(self.select_columns(sample, cloud))
        name = frame_id or sample.meta.extra.get("frame_id") or "frame"
        path = self.path_for(str(name))
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.fmt == "bin":
            cloud.tofile(str(path))
        else:
            np.save(str(path), cloud)
        return path

    def write_all(self, samples: Iterable[Tuple[Sample, np.ndarray]]) -> list:
        return [self.write(sample, points) for sample, points in samples]

    def __repr__(self) -> str:
        return f"OutputWriter({self.output_dir}, format={self.fmt}, columns={self.columns})"
