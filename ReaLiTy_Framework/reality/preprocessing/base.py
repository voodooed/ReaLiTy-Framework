"""Transform — a preprocessing stage operating on a Sample."""

from __future__ import annotations

from abc import abstractmethod

from reality.core.context import Sample
from reality.core.pipeline import Stage


class Transform(Stage):
    """A preprocessing step. Same contract as any pipeline stage."""

    @abstractmethod
    def apply(self, sample: Sample) -> Sample:
        """Transform ``sample`` and return the result."""
