"""ReaLiTy — a physics-informed orchestration framework for LiDAR intensity adaptation.

The framework wraps the frozen PICGAN model (``reality/models/PICGAN``) in a
config-driven pipeline. See README.md for the architecture.
"""

from reality.core.version import __version__

__all__ = ["__version__"]
