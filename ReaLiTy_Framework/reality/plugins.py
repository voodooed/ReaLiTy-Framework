"""Import the plugin packages so their registrations happen.

Registration is a side effect of import, and the core package deliberately does
not import the model or dataset packages itself: that would make ``reality.cli``
depend on torch just to parse a config. Call :func:`register_all` at the point a
run actually needs the registries populated.
"""

from __future__ import annotations

import importlib
from typing import List

#: Packages whose import registers implementations.
PLUGIN_MODULES = (
    "reality.datasets",
    "reality.preprocessing.projection",
    "reality.postprocessing.backprojection",
    "reality.degradation",
    "reality.models",
)

_REGISTERED = False


def register_all(force: bool = False) -> List[str]:
    """Import every plugin package once; returns the modules imported."""
    global _REGISTERED
    if _REGISTERED and not force:
        return []
    imported = [name for name in PLUGIN_MODULES if importlib.import_module(name)]
    _REGISTERED = True
    return imported
