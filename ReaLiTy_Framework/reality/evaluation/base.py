"""Evaluator — the interface every metric set implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable

import numpy as np


class Evaluator(ABC):
    """Scores generated intensity against a reference."""

    name: str = "evaluator"

    @abstractmethod
    def evaluate(self, generated: Iterable[np.ndarray],
                 reference: Iterable[np.ndarray]) -> Dict[str, float]:
        """Return a dict of metric name -> value."""
