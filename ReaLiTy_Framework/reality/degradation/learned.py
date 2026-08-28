"""LearnedDegradation — interface stub only. Deliberately not implemented.

Present so the extension point is visible and typed: a learned degradation would
subclass :class:`GeometricDegradation`, populate the same physics-intensity
column, and be selected with ``geometric_degradation: {type: learned}``. Because
it returns the same Sample shape as the physics plugin, PICGAN and every
downstream stage would be unchanged (README -> *Extensibility*).
"""

from __future__ import annotations

from reality.core.context import Sample
from reality.core.registry import DEGRADATIONS
from reality.degradation.base import GeometricDegradation


@DEGRADATIONS.register("learned")
class LearnedDegradation(GeometricDegradation):
    """Placeholder for a learned weather model. Raises if selected."""

    name = "learned"

    def apply(self, sample: Sample) -> Sample:
        raise NotImplementedError(
            "LearnedDegradation is an interface stub: no learned degradation model "
            "is implemented. Use geometric_degradation: {type: physics} for the "
            "the weather model-based path."
        )
