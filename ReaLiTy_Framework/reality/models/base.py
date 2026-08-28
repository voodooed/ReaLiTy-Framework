"""IntensityModel — the interface every intensity-transfer model implements.

PICGAN is registered as one implementation, not baked into the pipeline: a future
network only has to implement this interface and register itself to be selectable
via ``model: {type: ...}`` in config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from reality.core.config import Config
from reality.core.context import Sample


class IntensityModel(ABC):
    """A model that maps a source modality stack to target-domain intensity."""

    #: Registry name, set by subclasses.
    name: str = "base"

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config
        self._built = False

    @property
    def is_built(self) -> bool:
        return self._built

    @abstractmethod
    def build_model(self, in_channels_s: int, in_channels_r: int = 1) -> None:
        """Construct the networks for the given channel counts.

        ``in_channels_s`` is 3 when reflectance is available and 2 when it is not;
        the model must accept both without being edited.
        """

    @abstractmethod
    def train_step(self, batch: Tuple[Any, Any, Any]) -> Dict[str, float]:
        """Run one optimisation step on a ``(source, real, phy)`` batch."""

    @abstractmethod
    def generate(self, source: Union[Sample, Any]) -> Any:
        """Map a source stack to target-domain intensity (inference)."""

    @abstractmethod
    def load_weights(self, path: Union[str, Path]) -> None:
        """Load weights previously written by :meth:`save_weights`."""

    @abstractmethod
    def save_weights(self, path: Union[str, Path]) -> Path:
        """Persist the model's weights and return the path written."""

    def _require_built(self) -> None:
        if not self._built:
            raise RuntimeError(
                f"{type(self).__name__}: call build_model() before using the model"
            )
